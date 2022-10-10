from abc import ABC

import numpy as np
import torch
from torch import tensor as tt

from uniMASK.sequences import TokenSeq
from uniMASK.utils import get_inheritors, to_numpy


class Batch(ABC):
    """
    A batch needs to have all the information necessary to create it's own component of the model input to the
    transformer:
    [ batch_len , num_tokens, max_token_size ]

    And also needs to keep track of what outputs it should attend

    We want to be able to calculate accuracy, loss, and other statistics by batch, as this will serve us for debugging
    purposes.
    """

    RTG_MASKING_TYPES = [
        "BC",
        "RC_fixed",
        "RC_fixed_first",
        "BCRC_uniform_all",
        "BCRC_uniform_only_zeros",
        "BCRC_uniform_first",
        "Unchanged",
    ]

    def __init__(
        self,
        input_data,
        rtg_masking_type,
        silent=False,
        **kwargs,
    ):
        self.input_data = input_data
        self.mask_shape = input_data.shape
        self.rtg_masking_type = rtg_masking_type

        self.seq_len = self.input_data.seq_len
        self.num_seqs = self.input_data.num_seqs

        self.silent = silent

        # This has to be computed in the subclasses' init methods
        self.input_masks = None
        self.prediction_masks = None

        self.loss = None

        self.computed_output = False

    @classmethod
    def from_params(cls, input_data, batch_params):
        batch_class = batch_params["type"]
        return batch_class(input_data=input_data, **batch_params)

    def get_input_masks(self):
        """Get the input masks for observations and feeds. All logic will be in subclasses"""
        raise NotImplementedError()

    @staticmethod
    def postprocess_rtg_mask(rtg_mask, rtg_masking_type):
        """
        Various kinds of rtg masking.

        - BC: behavior cloning. Will mask all rtg info, always.
        - RC_fixed: reward-conditioning without randomization (all rtg tokens always present).
        - RC_fixed_first: reward-conditioning without randomization (first rtg tokens always present, rest always masked).
        - BCRC_uniform_all: with 50% chance masks all rtg info, and 50% _shows all of it_
        - BCRC_uniform_only_zeros: with 50% chance masks all rtg info, and 50% leaves mask the same as what the batch decided
        - BCRC_uniform_first: with 50% chance mask first rtg. Rest will always be masked
        - Unchanged: just keep whatever masking scheme the batch type generates
        """
        assert rtg_masking_type in Batch.RTG_MASKING_TYPES

        num_seqs, seq_len = rtg_mask.shape
        if rtg_masking_type == "BC":
            rtg_mask[:, :] = 0

        elif rtg_masking_type == "RC_fixed":
            rtg_mask[:, :] = 1

        elif rtg_masking_type == "RC_fixed_first":
            rtg_mask[:, 0] = 1
            rtg_mask[:, 1:] = 0

        elif rtg_masking_type == "BCRC_uniform_all":
            # NOTE: think hard before using this. Sometimes this won't lead to the desired effects
            # Consider using FuturePred batch, with randomized masking. Sometimes you'll want to predict the
            # action 3 with state and rtg up to timestep 3.
            # Using BCRC_uniform in that case will never enable you to do that. When it selects that you should see
            # rtg, it will show you all of the trajectory's RTG, leaking information about the reward.
            # For that case you should use BCRC_uniform_zero_only which only zeros out RTGs (but doesn't reveal any more than the batch class has decided to)
            uniform_rtg_mask = np.random.choice([0, 1], size=[num_seqs])
            rtg_mask[:, :] = tt(np.vstack([uniform_rtg_mask] * seq_len)).T

        elif rtg_masking_type == "BCRC_uniform_only_zeros":
            uniform_rtg_mask = np.random.choice([0, 1], size=[num_seqs])
            for seq_idx, zero_out in enumerate(uniform_rtg_mask):
                if zero_out:
                    rtg_mask[seq_idx, :] = 0

        elif rtg_masking_type == "BCRC_uniform_first":
            uniform_rtg_mask = np.random.choice([0, 1], size=[num_seqs])
            rtg_mask[:, 0] = tt(uniform_rtg_mask)
            rtg_mask[:, 1:] = 0

        elif rtg_masking_type == "Unchanged":
            pass

        else:
            raise ValueError("rtg_masking_type not recognized")

        return rtg_mask

    def get_prediction_masks(self):
        """By default, predict everything that wan't present in the input"""
        # For last item prediction, the prediction masks will be exactly the opposite relative to the input masks
        s_mask = 1 - self.input_masks["*"]["state"]
        a_mask = 1 - self.input_masks["*"]["action"]
        r_mask = 1 - self.input_masks["*"]["rtg"]
        return {"*": {"state": s_mask, "action": a_mask, "rtg": r_mask}}

    @classmethod
    def must_have_size_multiple_of(cls, seq_len):
        """Batch should have a number of sequences multiple of the returned number"""
        return 1

    def num_maskings_per_type(self):
        """
        If the batch allows for more than one masking type, we want to be able to perfectly tile N maskings of each
        type in the batch in order to reduce variance. We use `must_have_size_multiple_of` to determine how many
        sequences we should mask with each masking type.
        """
        num_masking_types = self.must_have_size_multiple_of(self.seq_len)
        assert (
            self.num_seqs % num_masking_types == 0
        ), "Num seqs in the batch {} must be divisible by num_masking_types {}".format(self.num_seqs, num_masking_types)
        num_per_type = self.num_seqs // num_masking_types
        return num_per_type

    @property
    def model_input(self):
        inp, timestep_inp = self.input_data.model_input(self.input_masks)
        return inp, timestep_inp

    def empty_input_masks(self):
        s_in_mask = torch.zeros((self.num_seqs, self.seq_len))
        act_in_mask = torch.zeros((self.num_seqs, self.seq_len))
        rtg_in_mask = torch.zeros((self.num_seqs, self.seq_len))
        return act_in_mask, rtg_in_mask, s_in_mask

    def empty_pred_masks(self):
        s_mask = torch.zeros_like(self.input_masks["*"]["state"])
        a_mask = torch.zeros_like(self.input_masks["*"]["action"])
        r_mask = torch.zeros_like(self.input_masks["*"]["rtg"])
        return a_mask, r_mask, s_mask

    ###############
    # INPUT UTILS #
    ###############

    def get_factor(self, factor_name):
        return self.input_data.get_factor(factor_name)

    def get_input_mask_for_factor(self, factor_name):
        if "*" in self.input_masks:
            mask_key = TokenSeq.get_mask_key(factor_name, self.input_masks["*"])
            return self.input_masks["*"][mask_key]
        else:
            raise NotImplementedError()

    def get_masked_input_factor(self, factor_name, mask_nans=False):
        factor = self.get_factor(factor_name)
        input_mask = self.get_input_mask_for_factor(factor_name)
        return factor.mask(input_mask, mask_nans)

    def get_prediction_mask_for_factor(self, factor_name):
        if "*" in self.prediction_masks:
            return self.prediction_masks["*"][factor_name]
        else:
            raise NotImplementedError()

    ###################

    def add_model_output(self, batch_output):
        self.input_data.add_model_output(batch_output)
        self.computed_output = True

    def compute_loss_and_acc(self, loss_weights):
        """
        We are given the output of the transformer
        We now want to computing losses and accuracy directly on a output head (predicting in behaviour space)

        NOTE: currently does not do accuracies
        """
        assert self.computed_output, "Have to add output with add_model_output before trying to compute loss"

        # need to implement a get_masked_items for Factors
        loss_dict = self.input_data.get_loss(loss_weights, self.prediction_masks)

        total_loss = 0.0
        for ts_name, factors_dict in loss_dict.items():
            for factor_name, v in factors_dict.items():
                total_loss += v.cpu()

        self.loss = total_loss
        loss_dict["total"] = total_loss
        return loss_dict

    @classmethod
    def get_dummy_batch_output(cls, data, batch_params, trainer):
        """
        Based on some data (in FullTokenSeq format), create a dummy batch and return it with the computed predictions

        TODO: have a parameter to do this with the model in eval mode, so as to not accidentally not use eval mode when
         evaluating
        """
        b = cls.from_params(data, batch_params)
        trainer.model(b)
        return b


class RandomPred(Batch):
    """
    Mask which has ======RND===== for actions
    and            ======RND===== for states
    """

    def __init__(self, mask_p=None, random_mask_p=True, **kwargs):
        """
        If using RandomPred, masks will always be random and you'll try to predict the randomly masked items.

        There will be a mask_p chance of having a mask at a position, and 1-mask_p chance of seeing the input.
        """
        super().__init__(**kwargs)
        # TODO: eventually re-introduce this?
        # assert self.rtg_masking_type in [
        #     "BCRC_uniform_all",
        #     "BCRC_uniform_first",
        # ], "If you're doing randomized masking, you probably want to train to do both BC and RC"

        self.random_mask_p = random_mask_p  # If random mask p, ignore mask probs
        if self.random_mask_p:
            assert mask_p is None, "If using random_mask_p, the mask_p should be set to None"
        else:
            assert 0 < mask_p < 1, "Mask p has to be a probability, and 0 or 1 don't make sense"
            self.mask_probs = [mask_p, 1 - mask_p]

        self.input_masks = self.get_input_masks()
        self.prediction_masks = self.get_prediction_masks()

    def get_input_masks(self):
        """
        Each token sequence will be of shape [num_seqs, seq_len, factor_size_sum]

        NOTE: Wherever the mask is _0_, the input will be masked out
        """
        mask_size = (self.num_seqs, self.seq_len)
        if self.random_mask_p:
            s_mask, a_mask, rtg_mask = [], [], []
            for i in range(self.num_seqs):
                # NOTE: There are probably more efficient ways to do this.
                seq_mask_ps = np.random.uniform()
                seq_ps = [seq_mask_ps, 1 - seq_mask_ps]

                seq_s_mask = np.random.choice([0, 1], p=seq_ps, size=[self.seq_len])
                seq_a_mask = np.random.choice([0, 1], p=seq_ps, size=[self.seq_len])
                seq_rtg_mask = np.random.choice([0, 1], p=seq_ps, size=[self.seq_len])

                s_mask.append(seq_s_mask)
                a_mask.append(seq_a_mask)
                rtg_mask.append(seq_rtg_mask)

            s_mask = np.array(s_mask)
            a_mask = np.array(a_mask)
            rtg_mask = np.array(rtg_mask)

        else:
            s_mask = np.random.choice([0, 1], p=self.mask_probs, size=mask_size)
            a_mask = np.random.choice([0, 1], p=self.mask_probs, size=mask_size)
            rtg_mask = np.random.choice([0, 1], p=self.mask_probs, size=mask_size)

        rtg_mask = self.postprocess_rtg_mask(rtg_mask, self.rtg_masking_type)
        return {"*": {"state": tt(s_mask), "action": tt(a_mask), "rtg": tt(rtg_mask)}}

    def get_prediction_masks(self):
        """For item prediction, the prediction masks will be exactly the opposite relative to the input masks"""
        return super().get_prediction_masks()


class CustomPred(Batch):
    """
    Mask should be of the form 10100100101 where 1s are not masked out, and 0s are masked out.

    action_masks: []
    state_masks: []
    """

    def __init__(self, state_masks, action_masks, rtg_masks, **kwargs):
        assert kwargs.get("rtg_masking_type", None) in [
            None,
            "Unchanged",
        ], "Given that we are hand specifying the RTG masks, doing any postprocessing other than Unchanged would not make sense"
        super().__init__(rtg_masking_type="Unchanged", **kwargs)
        assert self.seq_len == len(action_masks) == len(state_masks) == len(rtg_masks)
        self.state_masks = to_numpy(state_masks)
        self.action_masks = to_numpy(action_masks)
        self.rtg_masks = to_numpy(rtg_masks)
        self.input_masks = self.get_input_masks()
        self.prediction_masks = self.get_prediction_masks()

    def get_input_masks(self):
        """
        Each token sequence will be of shape [num_seqs, seq_len, factor_size_sum]
        """
        # We define all masks for the basic next prediction task as [num_seqs, seq_len] ones, and apply them across
        # the entire factor_size_sum dimension.
        s_in_mask = tt(np.repeat([self.state_masks], self.num_seqs, axis=0))
        act_in_mask = tt(np.repeat([self.action_masks], self.num_seqs, axis=0))
        rew_in_mask = tt(np.repeat([self.rtg_masks], self.num_seqs, axis=0))
        return {"*": {"state": s_in_mask, "action": act_in_mask, "rtg": rew_in_mask}}


class SpanPred(Batch):
    """
    Mask which has 1..100..001..1 for actions
    and            1..110..001..1 for states

    where 1s are not masked out, and 0s are masked out. You can sample actions from the back _or_ from the front.

    For span_limit (a,b), will mask out an additional action at the beginning to predict:
        States [a, b) and actions [a-1, b) will be masked out (unless a=0 for first-timestep backwards
        inference, actions [a, b)).

    span_limits examples if seq_len is 10:
    - span_limits=(10, 10) means that you're just training to predict the last action
    - span_limits=(9, 10) means you're training to predict the last state and the last two actions
    - span_limits=(1, 10) means you're training to predict the first action (and later things) given the first state

    State spans can also be predicted in backwards inference by setting span_limits=(0, t_to_predict + 1).

    NOTE: Check input and prediction mask attributes if usure what's going on
    """

    def __init__(self, span_limits=None, **kwargs):
        super().__init__(**kwargs)

        # Masked span to predict at inference time
        # None if training (masks)
        self.span_limits = span_limits
        self.training_spans = self.get_training_spans(self.seq_len)
        self.input_masks = self.get_input_masks()
        self.prediction_masks = self.get_prediction_masks()

        if self.span_limits is not None:
            assert type(span_limits) in [tuple, list]
            span_limits = tuple(span_limits)
            assert self.seq_len >= span_limits[1] >= span_limits[0] >= 0, (
                self.seq_len,
                span_limits[1],
                span_limits[0],
            )
            assert span_limits in self.training_spans, f"{span_limits}vs{self.training_spans}"

    @staticmethod
    def get_training_spans(seq_len):
        possible_span_limits = []
        for start in np.arange(seq_len + 1):
            for end in np.arange(start, seq_len + 1):
                possible_span_limits.append((start, end))
        return possible_span_limits

    @classmethod
    def must_have_size_multiple_of(cls, seq_len):
        """Batch should have a number of sequences multiple of the returned number"""
        return len(cls.get_training_spans(seq_len))

    def generate_training_span_limits(self):
        """Tiles all possible training span combinations to get `self.num_seqs` training masks."""
        num_per_type = self.num_maskings_per_type()
        training_spans = self.get_training_spans(self.seq_len)
        span_limits_n = np.concatenate([[span_limits] * num_per_type for span_limits in training_spans])
        np.random.shuffle(span_limits_n)
        return span_limits_n

    def get_masks_for_span_limits(self, span_limits_n):
        """
        Masks state, ac, rtg tensors according to `span_limits`.
        Each token sequence will be of shape [num_seqs, seq_len, factor_size_sum]
        """
        # We define all masks for the basic next prediction task as [num_seqs, seq_len] ones, and apply them across
        # the entire factor_size_sum dimension.
        s_in_mask = torch.ones((self.num_seqs, self.seq_len))
        act_in_mask = torch.ones((self.num_seqs, self.seq_len))
        rtg_in_mask = torch.ones((self.num_seqs, self.seq_len))

        for traj_idx, span_limits in enumerate(span_limits_n):
            # Mask out the things we want to predict (i.e. zero out)
            a, b = span_limits
            s_in_mask[traj_idx, a:b] = 0
            rtg_in_mask[traj_idx, a:b] = 0

            # If the state input mask starts at 0, have the action mask also start from there.
            # This will happen when doing backward prediction.
            a_in_start_idx = max(a - 1, 0)
            act_in_mask[traj_idx, a_in_start_idx:b] = 0

        rtg_in_mask = self.postprocess_rtg_mask(rtg_in_mask, self.rtg_masking_type)
        return {"*": {"state": s_in_mask, "action": act_in_mask, "rtg": rtg_in_mask}}

    def get_input_masks(self):
        """Get masked s, a, rtg for training or inference."""
        # Inference: predict with a particular span masked
        if self.span_limits is not None:
            span_limits_n = [self.span_limits] * self.num_seqs
        # Training: generate training masks
        else:
            span_limits_n = self.generate_training_span_limits()

        return self.get_masks_for_span_limits(span_limits_n)


class FuturePred(SpanPred):
    """
    Mask which has 1..100..00 for actions
    and            1..110..00 for states

    Differs from SpanPred in that the later states/actions are all zeros always.
    """

    @staticmethod
    def get_training_spans(seq_len):
        possible_span_limits = []
        for start in np.arange(1, seq_len + 1):
            end = seq_len
            possible_span_limits.append((start, end))
        return possible_span_limits


class PastPred(SpanPred):
    """
    Mask which has 00..001..1 for actions
    and            00..001..1 for states

    Differs from SpanPred in that the initial states/actions are all zeros always.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.rtg_masking_type == "BC"

    @staticmethod
    def get_training_spans(seq_len):
        possible_span_limits = []
        for end in np.arange(1, seq_len + 1):
            start = 0
            possible_span_limits.append((start, end))
        return possible_span_limits


class NextActionPred(FuturePred):
    """
    FuturePred, but only predict the next action (instead of all future states and actions).
    Enables comparison to DT's autoregressive loss.

    NOTE: when the rtg_masking_type is BC or Unchanged, this is equivalent to behavior cloning
    NOTE: when the rtg_masking_type is RC_fixed, this is equivalent to reward-conditioned behavior cloning

    If used with span_limits={}, it will take the validation dataset and randomly assign a different masking
    to each sequence. (This is slightly different than what the original version of the code by Jessy did, which
    generated _all_ possible maskings, while here they are randomly selected). To generate all possible maskings,
    use n=seq_len batches, each of which has a different span_limits, and then average the losses.

    Val loss will be average loss on p(a_t | s_1, a_1, ... s_{t-1}, a_{t-1}), *for all t*.
    This can be useful for a apples-to-apples comparison with DT's autoregressive loss.
    """

    def get_prediction_masks(self):
        s_mask, a_mask, r_mask = self.empty_pred_masks()

        # Only look at prediction loss for first missing action
        # First missing action for each seq in the batch
        first_missing_act_to_pred_t_n = self.input_masks["*"]["action"].sum(1).long()
        for seq_idx, first_missing_t in enumerate(first_missing_act_to_pred_t_n):
            a_mask[seq_idx, first_missing_t] = 1

        return {"*": {"state": s_mask, "action": a_mask, "rtg": r_mask}}


class BehaviorCloning(NextActionPred):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.rtg_masking_type in ["BC", "Unchanged"]


class WaypointConditionedBC(BehaviorCloning):
    def __init__(self, waypoints, **kwargs):
        # A list of timesteps for which the state is known
        self.waypoints = waypoints

        # Call super
        super().__init__(**kwargs)

    def get_input_masks(self):
        # Add waypoints to seen
        input_masks = super().get_input_masks()
        for waypoint_t in self.waypoints:
            input_masks["*"]["state"][:, waypoint_t] = 1
        return input_masks

    def get_prediction_masks(self):
        # Remove prediction of waypoints
        pred_masks = super().get_prediction_masks()
        for waypoint_t in self.waypoints:
            pred_masks["*"]["state"][:, waypoint_t] = 0
        return pred_masks


class GoalConditionedBC(BehaviorCloning):
    @staticmethod
    def get_training_spans(seq_len):
        # Differs from the super() method in that the start index cannot be the last state
        possible_span_limits = []
        for start in np.arange(1, seq_len):
            end = seq_len
            possible_span_limits.append((start, end))
        return possible_span_limits

    def generate_training_span_limits(self):
        # Differs from super in that that span limits is randomized rather than tessellated
        training_spans = self.get_training_spans(self.seq_len)
        span_indices = np.random.choice(len(training_spans), size=self.num_seqs)
        span_limits_n = [training_spans[idx] for idx in span_indices]
        return span_limits_n

    def get_input_masks(self):
        input_masks = super().get_input_masks()
        # Add goal to seen
        input_masks["*"]["state"][:, self.seq_len - 1] = 1
        return input_masks


class RCBehaviorCloning(NextActionPred):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.rtg_masking_type in ["RC_fixed", "RC_fixed_first"]


class DynamicsBatch(Batch, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert (
            self.rtg_masking_type == "BC"
        ), "RC has nothing to do with dynamics, should not include this info ever (batch will handle it)"
        self.random_indices = self.get_random_indices()
        self.input_masks = self.get_input_masks()
        self.prediction_masks = self.get_prediction_masks()

    def get_random_indices(self):
        """
        Get the index for which we will have the state, and for which we are either trying to infer
        the next state or the previous one (depending on the task)
        """
        random_indices = np.random.choice(self.seq_len - 1, size=self.num_seqs)
        return random_indices

    @classmethod
    def must_have_size_multiple_of(cls, seq_len):
        """Batch should have a number of sequences multiple of the returned number"""
        return seq_len - 1


class ForwardDynamics(DynamicsBatch):
    def get_input_masks(self):
        act_in_mask, rtg_in_mask, s_in_mask = self.empty_input_masks()
        for traj_idx, rnd_idx in enumerate(self.random_indices):
            # Give the network the state and action at this timestep
            s_in_mask[traj_idx, rnd_idx] = 1
            act_in_mask[traj_idx, rnd_idx] = 1
        return {"*": {"state": s_in_mask, "action": act_in_mask, "rtg": rtg_in_mask}}

    def get_prediction_masks(self):
        a_mask, r_mask, s_mask = self.empty_pred_masks()
        # Only predict the next state
        for traj_idx, rnd_idx in enumerate(self.random_indices):
            # Give the network the state and action at this timestep
            s_mask[traj_idx, rnd_idx + 1] = 1
        return {"*": {"state": s_mask, "action": a_mask, "rtg": r_mask}}


class BackwardsDynamics(DynamicsBatch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Shifting the random indices up by one so that we can't have any random indices = 0
        # (which would not have a previous state to predict)
        self.random_indices += 1

    def get_input_masks(self):
        act_in_mask, rtg_in_mask, s_in_mask = self.empty_input_masks()
        for traj_idx, rnd_idx in enumerate(self.random_indices):
            # Give the network the state at this timestep and the previous action
            s_in_mask[traj_idx, rnd_idx] = 1
            act_in_mask[traj_idx, rnd_idx - 1] = 1
        return {"*": {"state": s_in_mask, "action": act_in_mask, "rtg": rtg_in_mask}}

    def get_prediction_masks(self):
        a_mask, r_mask, s_mask = self.empty_pred_masks()
        # Only predict the previous state
        for traj_idx, rnd_idx in enumerate(self.random_indices):
            # Give the network the state and action at this timestep
            s_mask[traj_idx, rnd_idx - 1] = 1
        return {"*": {"state": s_mask, "action": a_mask, "rtg": r_mask}}


class DTActionPred(Batch):
    """
    Always takes in everything, and predicts all actions.
    It only makes sense to use this with a GPT architecture because otherwise
    the network could just copy stuff over from the input.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_masks = self.get_input_masks()
        self.prediction_masks = self.get_prediction_masks()

    def get_input_masks(self):
        mask_size = (self.num_seqs, self.seq_len)
        s_mask = np.ones(mask_size)
        a_mask = np.ones(mask_size)
        rtg_mask = np.ones(mask_size)
        rtg_mask = self.postprocess_rtg_mask(rtg_mask, self.rtg_masking_type)
        return {"*": {"state": tt(s_mask), "action": tt(a_mask), "rtg": tt(rtg_mask)}}

    def get_prediction_masks(self):
        s_mask = torch.zeros_like(self.input_masks["*"]["state"])
        a_mask = torch.ones_like(self.input_masks["*"]["action"])
        r_mask = torch.zeros_like(self.input_masks["*"]["rtg"])
        return {"*": {"state": s_mask, "action": a_mask, "rtg": r_mask}}


BATCH_TYPES = get_inheritors(Batch)
BATCH_NAMES = [c.__name__ for c in BATCH_TYPES]
BATCH_TYPES_BY_NAME = {name: cls for name, cls in zip(BATCH_NAMES, BATCH_TYPES)}
