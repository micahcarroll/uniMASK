
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor as tt

import uniMASK.utils
from uniMASK.utils import to_numpy

AVG_METRICS = False


"""
A transformer takes in a certain number of tokens. In NLP, usually the tokens are words: i.e. a sequence of words is 
fed into GPT _simultaneously_ (as it is a transfomer, rather than say an RNN), and a prediction is outputted in the
output "token space".

In our setting, we choose specific abstractions to allow for more flexibility: 

1) We decompose each token we are feeding into the network into various "Factors". As a grounding 
   example, consider a word token in an NLP task. You could consider a word as 1 "word Factor", or as N "letter Factors".
   In either case, they are fed into the network in the same way. The abstraction is there to be able to easily mask 
   individual Factors of a Token rather than have to mask the entire Token, and to keep track of losses for individual
   Factors. In the NLP domain, both of these are not immediately sensical things to do, but they make sense in the RL
   setting.
   
   In RL, one could have 1 Token per timestep, and have factors be "state", "action", "rtg". With the additional
   flexibility provided by this abstraction it's easy to only predict e.g. actions based on states.
   
# TODO: make this more clear
2) We consider the FullTokenSeq fed into the network as a collection of TokenSeqs. This allows for different _types_ of 
   tokens to be fed to the network. They could have different dimensions, etc. In the RL setting, one could also feed 
   "states", "actions", and "rewards" for each timestep as 3 separate tokens (as in the DT paper).
   
   An example of the whole pipeline is below.

This is what the data structures could possibly look like in a multi-agent RL enviroment 

FullTokenSeq:
    TokenSeq "state":
        FactorSeq(s): [agent0_state, agent1_state]

    TokenSeq "rtg":
        FactorSeq(s): [common_rtg]

    TokenSeq "action":
        FactorSeq(s): [agent0_action, agent1_action]
        
NOTE: one current assumption is that the input and the output space are symmetric. The output space is exactly the same
shape as the input one, so that any reconstruction or prediciton task can be made for which inputs are provided (although
potentially masked). This might be something that we want to relax over time.
"""

# If most of the input is close to 0 in value, adding a 0 as the masked value is actually going to cause problems.
# -1 seems better, should do ablations.
MASKED_VALUE = 0

# The value of a missing input. This way if it is used, it causes everything to break
MISSING_VALUE = np.nan


class FactorSeq:
    """
    A factor is a named input to the network. The name should be unique.

    Example factors in the RL setting could be "state", "action", etc.

    In terms of the data, for a generic single-agent RL setup we would expect shapes such as:
    - States will be [num_trajs, state_shape, seq_len]
    - Actions will be [num_trajs, action_shape, seq_len]
    - Rewards will be [num_trajs, reward_shape, seq_len]

    Each FactorSeq instance doesn't have masking or output associated with it until it's passed into a network. In
    particular, each FactorSeq should only be passed into a network and have the loss computed once.
    """

    def __init__(
        self,
        name,
        input_data,
        loss_type,
        output=None,
        output_mask=None,
        input_mask=None,
        **kwargs,
    ):
        assert isinstance(input_data, torch.Tensor)
        self.name = name
        self.input = input_data
        self.loss_type = loss_type
        self.shape = self.input.shape
        self.num_seqs, self.seq_len, self.size = self.shape
        assert len(self.input.shape) == 3, self.input.shape
        assert self.num_seqs > 0, "Making an empty factor is probably a bug?"

        ############################################################################
        # After being passed into the network, these fields will also be populated #
        ############################################################################

        # Output
        self.output = output
        # Mask that was used to generate the output
        self.output_mask = output_mask
        # Mask that was used to generate the input to the model which generated the output
        # NOTE currently removed this as it was not being populated and only useful for debugging
        #  If want to add back in also change on __getitem__
        # self.input_mask = input_mask

    def __getitem__(self, key):
        out, out_m, in_m = None, None, None

        if self.output is not None:
            assert self.output_mask is not None
            # assert self.input_mask is not None
            out = self.output.__getitem__(key)
            out_m = self.output_mask.__getitem__(key)
            # in_m = self.input_mask.__getitem__(key)

        return self.__class__(
            name=self.name,
            input_data=self.input.__getitem__(key),
            loss_type=self.loss_type,
            output=out,
            output_mask=out_m,
            input_mask=in_m,
        )

    @property
    def missing_ts(self):
        if hasattr(self, "_missing_ts"):
            return self._missing_ts

        if self.loss_type == "sce":
            max_vals = self.input.max(axis=2)[0]
            missing_inputs = max_vals.isnan()
        elif self.loss_type == "l2":
            missing_inputs = self.input.isnan()
            assert (missing_inputs.any(dim=2) == missing_inputs.all(dim=2)).all(), (
                "If a part of a factor is nan, the" "entire factor should be nan"
            )
            missing_inputs = missing_inputs.any(dim=2)
        else:
            raise NotImplementedError("Loss type {} not recognized for Factor {}".format(self.loss_type, self.name))

        # Check same missing ts across sequences
        assert (
            missing_inputs.any(dim=0) == missing_inputs.all(dim=0)
        ).all(), "If one input sequence has a specific missing input, all the other ones should have the same"
        missing_inputs = missing_inputs.any(dim=0)
        self._missing_ts = [t for t, missing in enumerate(missing_inputs) if missing]
        return self._missing_ts

    def mask(self, mask, mask_nans=False):
        """
        Will mask the input.

        When missing inputs exist, we want to assert that they are being masked at model input time
        """
        return self.mask_data(self.input, mask, mask_nans)

    @staticmethod
    def mask_data(data, mask, mask_nans=False):
        """
        Will mask the input.

        When missing inputs exist, we want to assert that they are being masked at model input time
        """
        assert mask.shape[:2] == data.shape[:2], f"{mask.shape} vs {data.shape}"
        assert len(mask.shape) in [len(data.shape), len(data.shape) - 1]

        # mask will be [num_seqs, seq_len]
        # self.data will be [num_seqs, seq_len, factor_size]
        masked_input = data.clone()
        masked_input[mask == 0] = MASKED_VALUE
        if mask_nans:
            masked_input[masked_input.isnan()] = MASKED_VALUE
        else:
            assert (
                not masked_input.isnan().any()
            ), f"Some Factor: Mask does not match NaN masking in data, check whether timesteps have been updated in the input"
        return masked_input

    def add_model_output(self, output_data):
        assert output_data.shape == self.input.shape
        assert self.output is None, "You shouldn't be predicting twice for the same set of data"
        self.output = output_data

    def is_structurally_equal(self, other):
        """
        Returns whether other instance of FactorSeq has the same structure (even though it might have different
        number of sequences and sequence contents). Used for checking e.g. whether two instances can be merged
        """
        assert isinstance(other, FactorSeq)
        assert self.__dict__.keys() == other.__dict__.keys()

        # The inputs are allowed to be different. Number of sequences also, so shape should also be ignored.
        keys_to_ignore = ["input", "num_seqs", "shape"]
        # NOTE: timestep will always have loss_weight: np.nan, which always returns false if you check for equality with itself
        if self.name == "timestep":
            keys_to_ignore.append("loss_weight")
        return dict_equal_check(self, other, keys_to_ignore)

    def merge(self, other):
        """Returns a new instance with data from both cases"""
        assert self.is_structurally_equal(other)
        merged_data = torch.cat([self.input, other.input], dim=0)
        assert self.output is None, "For now merging does not support post-output merging"
        return self.__class__(
            name=self.name,
            input_data=merged_data,
            loss_type=self.loss_type,
            output=None,
            output_mask=None,
            input_mask=None,
        )

    @classmethod
    def concatenate(cls, factors):
        assert all(isinstance(f, FactorSeq) for f in factors)
        assert all(factors[0].is_structurally_equal(f) for f in factors)
        assert all(f.output is None for f in factors), "For now merging does not support post-output merging"
        merged_data = torch.cat([f.input for f in factors], dim=0)
        return cls(
            name=factors[0].name,
            input_data=merged_data,
            loss_type=factors[0].loss_type,
            output=None,
            output_mask=None,
            input_mask=None,
        )

    @property
    def inputs_hr(self):
        """
        The inputs in a human readable format. For discretized Factors, this is equivelent to taking the argmax of
        the one hot encoding.
        """
        # NOTE: maybe come up with better name for this. Really what it is a non one-hot version of the inputs.
        if self.loss_type == "sce":
            argmax = self.input.argmax(dim=2).float()
            argmax[self.input.isnan().any(dim=2)] = MISSING_VALUE
            if not argmax.isnan().any():
                argmax = argmax.int()
            return argmax
        else:
            return self.input.squeeze()

    @property
    def predictions(self):
        if self.loss_type == "sce":
            return self.output.argmax(dim=2)
        else:
            raise NotImplementedError()

    @property
    def prediction_probs(self):
        if self.loss_type == "sce":
            return to_numpy(torch.nn.Softmax(dim=2)(self.output))
        else:
            raise NotImplementedError()

    def sample_timestep_predictions(self, num_samples, t, argmax=False):
        if self.loss_type == "sce":
            sample_probs = self.prediction_probs[:, t]
            if argmax:
                assert num_samples == 1, "Why take more than 1 if the same?"
                samples = torch.argmax(tt(sample_probs), dim=-1)
            else:
                samples = torch.multinomial(tt(sample_probs), num_samples=num_samples)
            return samples, sample_probs
        else:
            raise NotImplementedError()

    def get_loss(self, loss_weight, pred_mask=None):
        """
        Computes the loss for the predictions for the current FactorSeq

        This should only be done once, as computing the loss also saves the mask
        """
        if pred_mask is None:
            assert self.output_mask is not None
            pred_mask = self.output_mask
        else:
            self.output_mask = pred_mask.to(int)

        assert pred_mask.shape == self.output.shape[:2], "{} vs {}".format(pred_mask.shape, self.output.shape)

        # pred_mask: [num_trajs, seq_len]
        pred_mask = pred_mask.unsqueeze(2)

        preds = get_masked_items(self.output, pred_mask)
        targets = get_masked_items(self.input, pred_mask).float()
        assert preds.shape == targets.shape

        if len(preds) == 0 or loss_weight == 0:
            # A torch 0, to make it consistent with other loss values which will be tensors
            return tt([0])[0]

        if self.loss_type == "l2":
            loss = nn.MSELoss()(preds, targets)

        elif self.loss_type == "sce":
            # Targets are passed in a one-hot-encoding and have to be flattened
            assert self.missing_ts == [], (
                "You probably shouldn't be computing losses for inputs with missing timesteps. "
                "You don't have targets for them!"
            )
            assert (targets.sum(dim=1) == 1).all(), "{}".format(targets.sum(dim=1) == 1)
            targets = targets.argmax(dim=1)
            loss = nn.CrossEntropyLoss()(preds, targets)
        else:
            raise NotImplementedError()

        return loss * loss_weight

    def get_accuracy(self, mode="avg"):
        assert self.loss_type == "sce", "Accuracy only makes sense in discretized settings"

        targets = self.input.argmax(2)
        predictions = self.output.argmax(2)

        matches = (targets == predictions).to(int)
        acc_by_t = []
        num_by_t = tt([self.output_mask[:, t].sum() for t in range(self.seq_len)])
        for t in range(self.seq_len):
            t_acc = get_masked_items(matches[:, t], self.output_mask[:, t]).to(float).mean()
            acc_by_t.append(t_acc)

        if mode == "avg":
            # Remove timesteps that don't have any predictions from the calculation
            acc_by_t, num_by_t = zip(*[(n, a) for n, a in zip(num_by_t, acc_by_t) if n != 0])
            weight_by_t = tt(num_by_t) / tt(num_by_t).sum()
            return (tt(acc_by_t) * weight_by_t).sum()
        elif mode == "by_t":
            return tt(acc_by_t), num_by_t
        else:
            raise ValueError()


class TokenSeq:
    """
    A factor group is a set of Factors that will be fed into the transformer together (will receive only 1 joint
    embedding). I.e., they will form 1 token.
    """

    def __init__(self, name, factors):
        assert all([isinstance(factor, FactorSeq) for factor in factors])
        self.name = name
        self.factors = factors

        # NOTE: the timestep factor is treated as a special case throughout the code, as
        #  it will not be fed to the model. Rather it will be used as an alternative to the positional encoding
        self.factor_sizes = tt([factor.size for factor in self.factors if factor.name != "timestep"])
        self.factor_names = [f.name for f in self.factors]
        assert len(self.factor_names) == len(set(self.factor_names)), "factor names must be unique"

        # Check factors have same num of trajectories, and of same sequence length
        assert all([self.factors[0].num_seqs == factor.num_seqs for factor in self.factors])
        assert all([self.factors[0].seq_len == factor.seq_len for factor in self.factors])

        self.num_seqs = self.factors[0].num_seqs
        self.seq_len = self.factors[0].seq_len
        self.input = self.model_input()
        self.shape = self.input.shape

    def __repr__(self):
        return str([(f.name, f.shape) for f in self.factors])

    def is_structurally_equal(self, other):
        """
        Returns whether other instance of TokenSeq has the same structure (even though it might have different
        number of sequences). Useful for checking e.g. whether two instances can be merged
        """
        assert isinstance(other, TokenSeq)
        assert self.__dict__.keys() == other.__dict__.keys()
        # The factor seqs are checked separately. Number of sequences are allowed to be different, so shape should also be ignored.
        # Input may have nans (which will never count as equal), and are checked separately within factors, so should also be ignored.
        keys_to_ignore = ["factors", "num_seqs", "shape", "input"]
        factors_eq = [ts0.is_structurally_equal(ts1) for ts0, ts1 in zip(self.factors, other.factors)]
        return dict_equal_check(self, other, keys_to_ignore) and all(factors_eq)

    def merge(self, other):
        """Returns a new instance with data from both cases"""
        assert self.is_structurally_equal(other)
        fs_n = []
        for fs0, fs1 in zip(self.factors, other.factors):
            fs_n.append(fs0.merge(fs1))
        return self.__class__(name=self.name, factors=fs_n, loss_weight=self.loss_weight)

    @classmethod
    def concatenate(cls, token_seqs):
        assert all(isinstance(t, TokenSeq) for t in token_seqs)
        assert all(token_seqs[0].is_structurally_equal(t) for t in token_seqs)
        concatenated_factors_n = []
        for factors in zip(*[t.factors for t in token_seqs]):
            concatenated_factors_n.append(FactorSeq.concatenate(factors))
        return cls(name=token_seqs[0].name, factors=concatenated_factors_n)

    @classmethod
    def add_model_output(cls, output, input_token_seq):
        start_idx = 0
        for factor in input_token_seq.factors:
            # The timestep will not be predicted, as is not fed into the network directly (and thus also not outputted)
            if factor.name == "timestep":
                continue
            end_idx = start_idx + factor.size
            factor.add_model_output(output[:, :, start_idx:end_idx])
            start_idx = end_idx

    @staticmethod
    def get_mask_key(factor_name, mask_dict):
        """
        Given a factor and a mask dict, determine which mask it should be using.

        NOTE: Quite hacky right now. Say you want to have multiple components of the state,
         e.g. state_pos, state_door, etc.
         As long as you name them like this (with an underscore), all masking should work smoothly.
         The current masks that are defined are only for things that look like "action_X", "state_X", "rtg_X"

        TODO: add clear documentation about this somewhere
        """
        if factor_name.split("_")[0] in mask_dict.keys():
            mask_key = factor_name.split("_")[0]
        else:
            mask_key = "*"
        assert mask_key in mask_dict.keys(), f"Key {factor_name} was not among keys for mask_dict: {mask_dict.keys()}"
        return mask_key

    def mask_factors(self, input_mask_dict=None):
        # NOTE: Don't include timestep factor in the return call when querying for masked factors
        if input_mask_dict is None:
            return [f.input for f in self.factors if f.name != "timestep"]

        masked_factors = []
        for f in self.factors:
            if f.name == "timestep":
                continue
            mask_key = self.get_mask_key(f.name, input_mask_dict)
            masked_factors.append(f.mask(input_mask_dict[mask_key]))
        return masked_factors

    def get_unsqueezed_factor_masks(self, input_mask_dict=None):
        masked_factors = []
        for f in self.factors:
            if f.name == "timestep":
                continue
            mask_key = self.get_mask_key(f.name, input_mask_dict)
            mask = input_mask_dict[mask_key]
            masked_factors.append(mask.unsqueeze(-1).expand(-1, -1, f.size))
        return masked_factors

    def masked_model_input(self, mask_dict):
        """Masks the pre-computed model input"""
        # Will return a [num_trajs, seq_len, factor_1_size + ... + factor_n_size] tensor
        masks = torch.cat(self.get_unsqueezed_factor_masks(mask_dict), dim=2)
        assert masks.shape == self.input.shape
        out = FactorSeq.mask_data(self.input, masks)
        desired_shape = (self.num_seqs, self.seq_len, self.factor_sizes.sum())
        assert out.shape == desired_shape, "{} vs {}".format(out.shape, desired_shape)
        return out

    def model_input(self):
        """
        We only do the concatenation of factors once at initialization given that it's an expensive operation
        if factors are high-dimensional.
        We don't include timestep as it's dealt with differently (doesn't have to be concatenated)
        """
        factors = [f.input for f in self.factors if f.name != "timestep"]
        return torch.cat(factors, dim=2)

    def __getitem__(self, key):
        return self.__class__(self.name, [f.__getitem__(key) for f in self.factors])

    def get_loss(self, loss_weights, pred_mask_dict):
        loss_d = {}
        for f in self.factors:
            # When computing losses, don't compute it for the timestep factor (if present) as we won't be predicting it
            if f.name == "timestep":
                continue
            mask_key = self.get_mask_key(f.name, pred_mask_dict)
            loss_d[f.name] = f.get_loss(loss_weights[f.name], pred_mask_dict[mask_key])
        return loss_d

    def get_factor(self, factor_name):
        for f in self.factors:
            if f.name == factor_name:
                return f
        raise ValueError("Factor not found")


class FullTokenSeq:
    """
    Sequences of inputs that will be concatenated horizontally (i.e. in the number of input tokens
    dimension) rather than vertically.
    TODO: Eventually interleave them instead of concatenating (I don't think this matters because it will just change
     the positional encoding ordering). Especially if adding timestep info it definitely doesn't matter

    This is the full sequence of tokens which will be fed to the transformer.

    One important aspect of the input groups is that they might have different dimension.

    Each FullTokenSeq instance doesn't have masking or output associated with it until it's passed into a network.
    """

    def __init__(self, token_seqs):
        assert all([isinstance(ts, TokenSeq) for ts in token_seqs])
        self.token_seqs = token_seqs
        self.num_token_seqs = len(token_seqs)

        # Checking all num trajs across input groups are the same
        assert all([self.token_seqs[0].num_seqs == s.num_seqs for s in self.token_seqs])
        self.num_seqs = self.token_seqs[0].num_seqs

        # Checking seq lens across input groups are the same
        assert all([self.token_seqs[0].seq_len == s.seq_len for s in self.token_seqs])
        self.seq_len = self.token_seqs[0].seq_len

        # Kind of a hack to figure out whether the FullTokenSeq was created with stacked=True or not"""
        self.stacked = len(self.token_seqs) == 1
        self.factor_names = np.concatenate([ts.factor_names for ts in self.token_seqs])
        assert len(self.factor_names) == len(set(self.factor_names)), "Factor names across TokenSeqs have to be unique"
        self.factors = {n: self.get_factor(n) for n in self.factor_names}
        self.loss_types = {f_name: f.loss_type for f_name, f in self.factors.items()}
        self.factor_dims = {f_name: f.shape[2] for f_name, f in self.factors.items()}

        self.shape = self.model_input()[0].shape
        self.num_tokens = self.shape[1]
        # TODO: figure out a better way to deal with variable sizes of input?
        self.max_factor_size = self.shape[-1]

    @property
    def raw_data(self):
        return {k: v.input.clone() for k, v in self.factors.items()}

    def __getitem__(self, key):
        return self.__class__([g.__getitem__(key) for g in self.token_seqs])

    def __repr__(self):
        # NOTE: apparently repr strings should be instance-unique? Figure out why, and fix this here and in other classes, as it currently isn't
        return str({t.name: t.__repr__() for t in self.token_seqs})

    @staticmethod
    def raw_data_to_token_seq(raw_data, stacked, input_keys=None, loss_types=None):
        """
        From raw sequences of states, actions, and rewards, obtain a FullTokenSeq instance

        input_keys: subset of keys. If none, uses all keys found in raw_data
        loss_types: for each input key, what kind of loss is going to be used? This is useful so that human-readable
                    versions of the input can be shown appropriately, etc.
        """
        data_keys = list(raw_data.keys())  # Don't make this a set

        if input_keys is None:
            input_keys = {k for k in data_keys}
        else:
            assert isinstance(input_keys, set)
            assert input_keys.issubset(set(data_keys))

        input_keys = list(input_keys)  # Don't make this a set
        input_keys.sort()

        if loss_types is None:
            loss_types = {k: "sce" for k in input_keys}
        elif type(loss_types) is str:
            loss_types = {k: loss_types for k in input_keys}
        else:
            assert set(loss_types.keys()).issubset(set(raw_data.keys()))

        # Only add to returned token sequences the data for which loss_weights and loss_types is defined!
        assert set(input_keys) == set(loss_types.keys())

        if not stacked:
            # If not stacked, one will have one TokenSeq for each Factor type
            token_sequences = []
            for name in input_keys:
                if name == "timestep":
                    # Deal with timestep separately below
                    continue

                factor = FactorSeq(name, raw_data[name], loss_type=loss_types[name])
                token_sequences.append(TokenSeq(name, [factor]))

            if "timestep" in input_keys:
                # Re-create the last TokenSeq created, to also include the timestep Factor. This is
                # because we have already modified the code hackily to be able to deal with "timestep" as a
                # Factor together with another Factor in a TokenSeq, but more hacks would be required to have
                # timestep as a standalone Factor in a TokenSeq (the whole TokenSeq wouldn't have a shape, model_input, etc.)
                token_sequences.pop(-1)
                name = "timestep"
                timestep_factor = FactorSeq(name, raw_data[name], loss_type=loss_types[name])
                new_token_seq = TokenSeq(name, [factor, timestep_factor])
                token_sequences.append(new_token_seq)
        else:
            # If stacked, one will have only one TokenSeq with multiple Factors
            factors = []
            for name in input_keys:
                factor = FactorSeq(name, raw_data[name], loss_type=loss_types[name])
                factors.append(factor)
            token_sequences = [TokenSeq("all", factors)]

        fts = FullTokenSeq(token_sequences)
        assert set(fts.factor_names) == set(input_keys)
        return fts

    @staticmethod
    def empty_token_seq(names_and_shapes, stacked, input_keys=None, loss_types=None):
        """
        Returns a FullTokenSeq of missing-values with factor names and shapes as specified in `names_and_shapes`
        (additionally subselects factors according to whether they are included in loss_weights and loss_types)
        """
        raw_data = {}
        for factor_name, shape in names_and_shapes.items():
            assert len(shape) == 3, "Should be [num_seqs, seq_len, factor_size]"
            num_seqs, seq_len, factor_size = shape
            missing_data = tt(np.ones((num_seqs, seq_len, factor_size)) * MISSING_VALUE)
            raw_data[factor_name] = missing_data
        return FullTokenSeq.raw_data_to_token_seq(raw_data, stacked, input_keys, loss_types)

    def is_structurally_equal(self, other):
        """
        Returns whether other instance of FullTokenSeq has the same structure (even though it might have different
        number of sequences). Useful for checking e.g. whether two instances can be merged
        """
        assert isinstance(other, FullTokenSeq)
        assert self.__dict__.keys() == other.__dict__.keys()

        # The token seqs and factors are checked separately.
        # Number of sequences are allowed to be different, so shape should also be ignored.
        keys_to_ignore = ["token_seqs", "num_seqs", "shape", "factors", "factor_names"]
        token_seqs_eq = [ts0.is_structurally_equal(ts1) for ts0, ts1 in zip(self.token_seqs, other.token_seqs)]
        return dict_equal_check(self, other, keys_to_ignore) and all(token_seqs_eq)

    def merge(self, other):
        """Returns a new instance with data from both cases"""
        assert self.is_structurally_equal(other)
        ts_n = [ts0.merge(ts1) for ts0, ts1 in zip(self.token_seqs, other.token_seqs)]
        return self.__class__(token_seqs=ts_n)

    @classmethod
    def concatenate(cls, f_token_seqs):
        """
        Returns a FullTokenSeq that concatenates all the inputs.

        This will be EXTREMELY inefficient if trying to merge many full token sequences.
        To do so, we should write a proper concatenate function.
        """
        assert all(isinstance(t, FullTokenSeq) for t in f_token_seqs)
        assert all(f_token_seqs[0].is_structurally_equal(t) for t in f_token_seqs)
        concatenated_ts_n = []
        for token_seqs in zip(*[t.token_seqs for t in f_token_seqs]):
            concatenated_ts_n.append(TokenSeq.concatenate(token_seqs))
        return cls(token_seqs=concatenated_ts_n)

    def get_factor(self, factor_name):
        for ts in self.token_seqs:
            if factor_name in ts.factor_names:
                return ts.get_factor(factor_name)
        raise ValueError(f"Factor {factor_name} not found")

    def get_input_mask_for_factor(self, factor_name):
        # Link this to the batch methods
        factor = self.get_factor(factor_name)
        return factor.input_mask

    def get_prediction_mask_for_factor(self, factor_name):
        factor = self.get_factor(factor_name)
        return factor.output_mask

    def mask_factor_groups(self, mask_dict=None):
        if mask_dict is None:
            # This is just used for debugging. You would always want to pass in a mask_dict for actual computation
            return [ts.input for ts in self.token_seqs]

        masked_input_groups = []
        for ts in self.token_seqs:
            # Selecting masking to use for this token sequence
            mask_key = ts.name if ts.name in mask_dict.keys() else "*"
            masked_input_groups.append(ts.masked_model_input(mask_dict[mask_key]))
        return masked_input_groups

    def model_input(self, mask_dict=None, no_checks=True):
        """
        Puts the data in a format that can be ingested by the model
        """
        masked_input_groups = self.mask_factor_groups(mask_dict)

        max_factor_sum_size = max([g.shape[2] for g in masked_input_groups])

        if self.stacked:
            assert len(masked_input_groups) == 1
            out = masked_input_groups[0]
        else:
            # NOTE: These operations are quite slow so should only do them only if absolutely necessary
            # NOTE: for now we're padding all the input groups which are smaller than the largest one.
            #  if using stacked inputs this should be equivalent to `out = masked_input_groups[0]`
            # TODO: we could make this more flexible with new embedding scheme that embeds each factor separately
            padded_input_groups = []
            for group in masked_input_groups:
                curr_factor_size = group.shape[2]
                padded_zeros = torch.zeros(list(group.shape[:-1]) + [max_factor_sum_size])
                # Making sure you're not losing precision in this step
                padded_zeros = padded_zeros.type(group.type())
                padded_zeros[:, :, :curr_factor_size] = group
                padded_input_groups.append(padded_zeros)

            # Concatenate on the seq_len dimension, leading to a
            # [num_trajs, seq_len * num_input_groups, factor_1_size + ... + factor_n_size]
            # dimensional tensor
            out = torch.cat(padded_input_groups, dim=1)

        if self.stacked and not no_checks:
            # If using no checks, will skip this step that is quite slow
            b = masked_input_groups[0]
            assert np.all(out.numpy()[~out.isnan()] == b.numpy()[~b.isnan()])

        expected_shape = (
            self.num_seqs,
            self.seq_len * self.num_token_seqs,
            max_factor_sum_size,
        )
        assert out.shape == expected_shape, "{} vs {}".format(out.shape, expected_shape)

        if "timestep" in self.factor_names:
            timestep_out = self.get_factor("timestep").input
        else:
            timestep_out = None
        return out, timestep_out

    def add_model_output(self, model_output):
        """Grabs model output and puts it back into FullTokenSeqs"""
        # [num_seqs, seq_len * num_factor_seqs, max_factor_seq_size]
        assert self.seq_len * self.num_token_seqs == model_output.shape[1]

        # NOTE: maybe have a copy be created here so that the method doesn't modify the input
        #  This might be somewhat costly computationally, so currently not doing it
        token_seqs = []
        start_idx = 0
        for i in range(self.num_token_seqs):
            input_token_seq = self.token_seqs[i]
            end_idx = start_idx + self.seq_len
            token_seqs.append(TokenSeq.add_model_output(model_output[:, start_idx:end_idx, :], input_token_seq))
            start_idx = end_idx

    def get_loss(self, loss_weights, pred_mask):
        loss_d = {}
        for ts in self.token_seqs:
            mask_key = ts.name if ts.name in pred_mask.keys() else "*"
            loss_d[ts.name] = ts.get_loss(loss_weights, pred_mask[mask_key])
        return loss_d

    def filter_by(self, factor_name, values, seq_idxs):
        """Filter the data by a specific factor having a specific values at specific timesteps"""
        assert len(values) == len(seq_idxs) > 0

        f = self.get_factor(factor_name)
        assert f.loss_type == "sce", "Currently assuming that data will be in one-hot format"

        indices = np.arange(self.num_seqs)
        for value, seq_idx in zip(values, seq_idxs):
            # For every value, timestep pair, filter the trajectories further
            value_at_timestep = f.input.argmax(2)[:, seq_idx]
            indices = [idx for idx in indices if value_at_timestep[idx] == value]

        if len(indices) == 0:
            raise ValueError("No trajs matched the query")

        return self[indices, :, :], indices

    def split_data(self, train_prop=0.9):
        """Returns split data"""
        num_trajs = self.shape[0]
        train_size = int(num_trajs * train_prop)
        train_data = self[:train_size]
        test_data = self[train_size:]
        return train_data, test_data

    def get_factor_subset(self, input_keys):
        """
        Get FullTokenSequence with just a subset of the factors, as defined by loss_weights (which also
        overwrites the loss_weights values)

        # TODO: this is not general, only works with stacked or non-stacked structurings of the input
        """
        loss_types = {k: self.loss_types[k] for k in input_keys}
        return self.raw_data_to_token_seq(
            self.raw_data,
            stacked=self.stacked,
            input_keys=input_keys,
            loss_types=loss_types,
        )

    def with_changed_input(self, factor_name, seq_idx, x_n, num_classes=None):
        """
        Returns new FullTokenSeq instance, with changed input `x_idx`.

        # TODO: this is not general, only works with stacked or non-stacked structurings of the input
        # TODO: this is a major source of evaluation slowness (probably about 30% of the time is spent on this)
        """
        new_data = self.raw_data
        factor = self.get_factor(factor_name)
        new_inputs_n = deepcopy(factor.input)
        if factor.loss_type == "sce":
            assert num_classes is not None
            # In case of discrete classes, x_n will be a list of indices corresponding to the class of the datapoint
            # we want to add for each sequence in the batch
            new_datapoint_n = F.one_hot(x_n, num_classes=num_classes).squeeze()
        elif factor.loss_type == "l2":
            new_datapoint_n = x_n
        else:
            raise ValueError()
        new_inputs_n[:, seq_idx] = new_datapoint_n
        new_data[factor_name] = new_inputs_n
        return self.raw_data_to_token_seq(new_data, stacked=self.stacked, loss_types=self.loss_types)

    def with_added_missing_input(self, factor_name, seq_idx, x_n, num_classes=None):
        """
        Returns new FullTokenSeq instance, with the input `x_idx` added for a missing index.
        Currently only works for discrete classes
        """
        assert seq_idx in self.get_factor(factor_name).missing_ts
        return self.with_changed_input(factor_name, seq_idx, x_n, num_classes)

    def shift_by_one(self):
        """
        Returns a copy of the current token sequence with all the inputs shifted back by one, and MISSING_VALUE
        replacing the last timestep.

        # TODO: this is not general, only works with stacked or non-stacked structurings of the input
        """
        new_data = self.raw_data
        for factor_name, factor_data in new_data.items():
            num_seqs, seq_len, factor_size = factor_data.shape
            missing_data = tt(np.ones((num_seqs, 1, factor_size)) * MISSING_VALUE)
            new_data[factor_name] = torch.cat([factor_data[:, 1:, :], missing_data], dim=1)
        return self.raw_data_to_token_seq(
            new_data,
            stacked=self.stacked,
            loss_types=self.loss_types,
            input_keys=set(self.factor_names),
        )


#########
# UTILS #
#########


def get_masked_items(items, mask):
    assert items.shape[:-1] == mask.shape[:-1], (items.shape, mask.shape)
    masked_indices = mask.reshape(-1).nonzero().ravel()
    if len(items.shape) == 1:
        masked_items = items[
            masked_indices,
        ]
    elif len(items.shape) == 2:
        masked_items = items.reshape(-1)[
            masked_indices,
        ]
    elif len(items.shape) == 3:
        n1, n2 = items.shape[:2]
        masked_items = items.reshape(n1 * n2, -1)[
            masked_indices,
        ]
    else:
        raise ValueError("{}".format(items.shape))
    return masked_items.to(uniMASK.utils.DEVICE)


def get_accuracy(output_masked, targets_masked):
    """Expecting output_masked to be probabilities (or logits, as argmaxing leads to same result)"""
    return (output_masked.argmax(axis=1).squeeze() == targets_masked.squeeze()).float().mean()


def dict_equal_check(self, other, keys_to_ignore):
    for k in self.__dict__.keys():
        if k in keys_to_ignore:
            continue
        v0, v1 = self.__dict__[k], other.__dict__[k]
        if isinstance(v0, torch.Tensor):
            curr_v_equal = (v0 == v1).all().item()
        else:
            curr_v_equal = v0 == v1

        if not curr_v_equal:
            return False
    return True
