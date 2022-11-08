

import math
import os
from abc import ABC

import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import GPT2Config, GPT2Model

import uniMASK.utils
from uniMASK.batches import DTActionPred, FuturePred
from uniMASK.sequences import MASKED_VALUE
from uniMASK.utils import load_from_json, save_as_json

print("LOCAL={}".format(uniMASK.utils.LOCAL))
if uniMASK.utils.LOCAL:
    print("Running locally! (or GPU not recognized)")


class CustomModel(nn.Module):
    MODEL_PARAMS_SAVE_NAME = "model_params"

    def __init__(self, train_data):
        super().__init__()
        if train_data is None:
            self.s_mean = self.a_mean = self.rtg_mean = 0
            self.s_std = self.a_std = self.rtg_std = 1
        else:
            if "state_mean" in train_data.data_keys:
                self.s_mean = train_data.state_mean
                self.s_std = train_data.state_std
            if "action_mean" in train_data.data_keys:
                self.a_mean = train_data.action_mean
                self.a_std = train_data.action_std
            if "rtg_mean" in train_data.data_keys:
                self.rtg_mean = train_data.rtg_mean
                self.rtg_std = train_data.rtg_std

    @staticmethod
    def model_save_name(is_best):
        """Formatting for model saved name"""
        return "model_best" if is_best else "model"

    def save(self, save_dir, is_best=False):
        model_params_path = os.path.join(save_dir, self.MODEL_PARAMS_SAVE_NAME)
        model_path = os.path.join(save_dir, self.model_save_name(is_best))
        save_as_json(self.params, model_params_path)
        torch.save(self.state_dict(), model_path)
        print("Saved model at {}".format(model_path))

    @classmethod
    def load(cls, save_dir, best=False):
        model_params_path = os.path.join(save_dir, cls.MODEL_PARAMS_SAVE_NAME)
        # Create model class with same architecture
        model_params = load_from_json(model_params_path)
        model = cls(**model_params)
        # Load model weights
        model_path = os.path.join(save_dir, cls.model_save_name(best))
        model.load_state_dict(torch.load(model_path, map_location=uniMASK.utils.DEVICE))
        return model.to(uniMASK.utils.DEVICE)

    def forward(self, batch):
        """
        Calls the model on the batch input, and adds the output information to the batch instance

        NOTE: this method is mutable, because copying the data would be unnecessarily expensive
        """
        raise NotImplementedError()

    @staticmethod
    def validate_training_params(tp):
        """Only implemented for DT for now"""

    # NOTE: stub of logic for reward-scaling and normalizations
    # def pre_forward_data_scaling(self, batch):
    #     """
    #     Transforms the data according to relevant axes: rescales state/action/rtg
    #     """
    #     if self.reward_scale is not None:
    #         rtgs = batch.get_factor("rtg")
    #
    #     if self.state_normalization:
    #         # Normalize (at network time, so don't have to worry about doing it at batch and eval time)
    #         # TODO: add flag for state normalization, and add state normalization to FB too
    #         # states = (states - self.s_mean) / self.s_std
    #         # actions = (actions - self.a_mean) / self.a_std
    #         # returns_to_go = (returns_to_go - self.rtg_mean) / self.rtg_std
    #         pass
    #
    #     if self.action_normalization:
    #         pass


class FlexiModel(CustomModel, ABC):
    def __init__(
        self,
        cat_input_dim,
        embed_dim,
        nlayers,
        feedforward_nhid,
        dropout,
        seq_len,
        train_dataset,
        timestep_encoding,
        action_tanh,
        **kwargs,
    ):
        """Adapted from the PyTorch transformers tutorial"""
        super().__init__(train_dataset)

        # The dimensionality of each timestep to be embedded
        # In our case it will likely be the dimension of concatenated state, action, and reward factors
        self.cat_input_dim = cat_input_dim

        # What dimension we embed each token in
        self.embed_dim = embed_dim

        # Number of hidden units for the feedforward components of the network
        self.feedforward_nhid = feedforward_nhid

        # Number of transformer encoder layers (for BERT-based model) or NN layers (for FlexiNN)
        self.nlayers = nlayers

        # The dimensionality of each output token. Usually the same as the input token dimensionality,
        # as the output will contain logits or predictions for each dimension of the input
        self.cat_out_dim = cat_input_dim

        # Dropout probability
        self.dropout = dropout

        # TODO: add assertion checks that this is consistent with data encountered
        self.seq_len = seq_len

        self.timestep_encoding = timestep_encoding

        self.action_tanh = action_tanh

    @property
    def params(self):
        return {
            "cat_input_dim": self.cat_input_dim,
            "embed_dim": self.embed_dim,
            "feedforward_nhid": self.feedforward_nhid,
            "nlayers": self.nlayers,
            "dropout": self.dropout,
            "seq_len": self.seq_len,
            "action_tanh": self.action_tanh,
            "timestep_encoding": self.timestep_encoding,
        }


class uniMASKModel(FlexiModel):
    def __init__(
        self,
        cat_input_dim,
        embed_dim,
        nheads,
        nlayers,
        feedforward_nhid,
        dropout,
        action_tanh,
        max_ep_len=1000,
        seq_len=None,
        train_dataset=None,
        timestep_encoding=False,
        **kwargs,
    ):
        super().__init__(
            cat_input_dim,
            embed_dim,
            nlayers,
            feedforward_nhid,
            dropout,
            seq_len,
            train_dataset,
            timestep_encoding,
            action_tanh,
            **kwargs,
        )
        assert not self.action_tanh, "Setting action tanh to true won't do anything with the main uniMASK model"

        # Number of attention heads in multi-headed attention
        self.nheads = nheads

        ###############
        # MODEL SETUP #
        ###############

        # Set up positional encoder
        self.pos_encoder = PositionalEncoding(embed_dim, dropout).to(uniMASK.utils.DEVICE)

        # Set up linear encoder and decoder layers to move from token to embedding space
        # print(
        #     "Setting up linear encoder which goes from {}-d input space to {}-d embeddings".format(
        #         cat_input_dim, embed_dim
        #     )
        # )
        self.linear_encoder = nn.Linear(cat_input_dim, embed_dim).to(uniMASK.utils.DEVICE)
        relu = nn.ReLU().to(uniMASK.utils.DEVICE)

        # Is used to encode the whole input
        self.input_encoder = lambda x: relu(self.linear_encoder(x))
        if self.timestep_encoding:
            self.timestep_encoder = nn.Embedding(max_ep_len, embed_dim).to(uniMASK.utils.DEVICE)

        # This is actual linear layer Ax + b, with A and b learnable
        self.linear_decoder = nn.Linear(embed_dim, self.cat_out_dim).to(uniMASK.utils.DEVICE)

        # Set up encoder layers
        # The basic encoder building block
        encoder_layers = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nheads,
            dim_feedforward=feedforward_nhid,
            dropout=dropout,
        ).to(uniMASK.utils.DEVICE)
        # Stack it `nlayers` times to form the bulk of the network
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers).to(uniMASK.utils.DEVICE)

        self.init_weights()

    @property
    def params(self):
        params = super().params
        params["nheads"] = self.nheads
        return params

    def init_weights(self):
        initrange = 0.1
        self.linear_decoder.bias.data.zero_()
        self.linear_decoder.weight.data.uniform_(-initrange, initrange)

        self.linear_encoder.bias.data.zero_()
        self.linear_encoder.weight.data.uniform_(-initrange, initrange)

    def pre_forward(self, batch):
        """Encode the data times some normalizing constant"""
        src, timesteps = batch.model_input

        # [one_hot_d, seq_len, num_parallel_seqs]
        src = src.float().to(uniMASK.utils.DEVICE)
        # assert src.shape[1] == self.seq_len # NOTE: can't check this because will fail with stacked=False

        # Transposing so the linear layer can transform the one-hot dimension to
        # the hidden dimension
        # [num_parallel_seqs, seq_len, one_hot_d]
        src = self.input_encoder(src)

        # [seq_len, num_parallel_seqs, hidden_dim]
        src = src * math.sqrt(self.embed_dim)
        src = src.permute(1, 0, 2)

        # [num_parallel_seqs, seq_len, hidden_dim]
        # Encode position info, either with timesteps directly or with positional encodings
        if self.timestep_encoding:
            assert timesteps is not None
            timesteps = timesteps.to(uniMASK.utils.DEVICE)
            timesteps_embed = self.timestep_encoder(timesteps.squeeze(2).int())
            timesteps_embed = timesteps_embed.permute(1, 0, 2)

            # In the case of unstacked inputs, you might have multiple token sequences, so that you will have src.shape[0] != number of timesteps in each window.
            # For example, in the original DT paper, src.shape[0] was always 3T where T was the context window size, because each action, state, and reward was fed into the network as a separate token.
            assert src.shape[0] % timesteps_embed.shape[0] == 0
            num_token_seqs = src.shape[0] // timesteps_embed.shape[0]
            if num_token_seqs > 1:
                # The same thing can happen here, and if we are in that 3T case, we should triple our timestep encodings concatenating them
                timesteps_embed = torch.cat([timesteps_embed] * num_token_seqs, dim=0)

            src = src + timesteps_embed
        else:
            assert timesteps is None
            src = self.pos_encoder(src)
        return src

    def forward(self, batch):
        """
        Calls the model on the batch input, and adds the output information to the batch instance

        This no-return call is required because the batch has to process the model output internally
        """
        src = self.pre_forward(batch)

        # Transformer encoder
        # [seq_len, batch_size, embedding_dimension]
        # TODO: Figure out whether feeding in masks at this point would be useful too
        #  Currently the network is just learning that the inputs are masked implicitly,
        #  instead of zeroing out the attention outputs explicitly
        output = self.transformer_encoder(src)

        output = self.linear_decoder(output)
        # Returns tensor that is also with shape [batch_size, seq_len, embedding_dimension]
        batch.add_model_output(output.permute(1, 0, 2))


class FlexiNNModel(FlexiModel):
    def __init__(
        self,
        cat_input_dim,
        embed_dim,
        nlayers,
        feedforward_nhid,
        dropout,
        seq_len,
        action_tanh,
        train_dataset=None,
        timestep_encoding=False,
        **kwargs,
    ):
        super().__init__(
            cat_input_dim,
            embed_dim,
            nlayers,
            feedforward_nhid,
            dropout,
            seq_len,
            train_dataset,
            timestep_encoding,
            action_tanh,
            **kwargs,
        )
        assert timestep_encoding is False, "currently not supported"

        ###############
        # MODEL SETUP #
        ###############
        self.in_dim = cat_input_dim * seq_len

        # NOTE: no timestep info right now
        nh = feedforward_nhid
        layers = [nn.Linear(self.in_dim, nh)]
        for _ in range(nlayers - 1):
            layer = [nn.ReLU(), nn.Dropout(dropout), nn.Linear(nh, nh)]
            layers.extend(layer)
        layers.extend([nn.ReLU(), nn.Dropout(dropout), nn.Linear(nh, self.in_dim)])

        if self.action_tanh:
            layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)
        self.model.float()
        self.model.to(uniMASK.utils.DEVICE)

    @property
    def params(self):
        params = super().params
        params["action_tanh"] = self.action_tanh
        return params

    def forward(self, batch):
        """
        Calls the model on the batch input, and adds the output information to the batch instance

        This no-return call is required because the batch has to process the model output internally
        """
        src, _ = batch.model_input
        src = src.to(uniMASK.utils.DEVICE)

        # Transformer encoder
        # [seq_len, batch_size, embedding_dimension]
        # TODO: kind of weird that float casting is done here. The type should be set correctly upstream of here.
        #  the same is happening for the FB model
        output = self.model(src.reshape(-1, self.in_dim).float()).reshape(-1, self.seq_len, self.cat_input_dim)

        # Returns tensor that is also with shape [batch_size, seq_len, embedding_dimension]
        batch.add_model_output(output)


class DecisionTransformer(CustomModel):
    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)

    NOTE: Taken from DT codebase, and adapted to our trainer. Verified that very similar losses and evaluations would
     occur in Mujoco when setting trainer and model parameters similarly.
    """

    def __init__(
        self,
        input_dims,
        embed_dim,
        nheads,
        nlayers,
        feedforward_nhid,
        dropout,
        seq_len,
        action_tanh,
        max_ep_len=1000,
        max_length=None,
        timestep_encoding=False,
        activation_function="relu",
        train_dataset=None,
        **kwargs,
    ):
        super().__init__(train_dataset)
        self.seq_len = seq_len
        self.nheads = nheads
        self.nlayers = nlayers
        self.feedforward_nhid = feedforward_nhid
        self.dropout = dropout
        # Individual input factors' dimentions
        self.input_dims = input_dims
        self.state_dims, self.act_dim, self.rtg_dim = self.parse_input_dims(input_dims)

        self.action_tanh = action_tanh
        self.activation_function = activation_function
        self.max_ep_len = max_ep_len
        self.max_length = max_length
        self.hidden_size = feedforward_nhid
        self.embed_dim = embed_dim
        self.timestep_encoding = timestep_encoding
        config = GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=embed_dim,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            summary_first_dropout=dropout,
            n_inner=feedforward_nhid,
            n_layer=nlayers,
            n_head=nheads,
            n_positions=max_ep_len,
            n_ctx=max_ep_len,
            activation_function=activation_function,
        )

        if self.timestep_encoding:
            # is that the positional embeddings are removed (since we'll add those ourselves)
            self.transformer = GPT2Model(config)
            self.embed_timestep = nn.Embedding(max_ep_len, embed_dim)
        else:
            self.transformer = GPT2Model(config)

        self.embed_return = torch.nn.Linear(self.rtg_dim, embed_dim).to(uniMASK.utils.DEVICE)
        self.embed_action = torch.nn.Linear(self.act_dim, embed_dim).to(uniMASK.utils.DEVICE)
        # The state might have multiple components, and we want an embedding fn for each one.
        self.embed_state_k = {k: torch.nn.Linear(v, embed_dim).to(uniMASK.utils.DEVICE) for k, v in self.state_dims.items()}
        for k, v in self.embed_state_k.items():
            # Adding explicitly as submodule because each embedding component was not added automatically without
            # being in a standalone instance property
            self.add_module("embed_state_k_" + k, v)
        self.embed_ln = nn.LayerNorm(embed_dim).to(uniMASK.utils.DEVICE)

        # NOTE: DT does not predict states or returns for their paper.
        #  We could simply not predict states and returns, and fill the output with NaNs.
        #  We just follow the original DT code and predict them and then ignore them.
        self.predict_return = torch.nn.Linear(embed_dim, self.rtg_dim).to(uniMASK.utils.DEVICE)
        network_end = [nn.Tanh()] if action_tanh else []
        self.predict_action = nn.Sequential(*([nn.Linear(embed_dim, self.act_dim)] + network_end)).to(
            uniMASK.utils.DEVICE)
        # The state might have multiple components, and we want an embedding fn for each one.
        self.predict_state_k = {k: torch.nn.Linear(embed_dim, v).to(uniMASK.utils.DEVICE) for k, v in self.state_dims.items()}

    @property
    def params(self):
        return {
            "input_dims": self.input_dims,
            "embed_dim": self.embed_dim,
            "nheads": self.nheads,
            "nlayers": self.nlayers,
            "feedforward_nhid": self.feedforward_nhid,
            "dropout": self.dropout,
            "seq_len": self.seq_len,
            "max_ep_len": self.max_ep_len,
            "max_length": self.max_length,
            "action_tanh": self.action_tanh,
            "timestep_encoding": self.timestep_encoding,
            "activation_function": self.activation_function,
        }

    @staticmethod
    def parse_input_dims(input_dims):
        """
        In most cases, you'll only have one input factor for each of state, action, and rtg.
        In the minigrid environment however, there will be two input factors for the state. We need to be able to
        account for that.
        """
        state_dims = {k: v for k, v in input_dims.items() if "state" in k}
        action_dims = [v for k, v in input_dims.items() if "action" in k]
        rtg_dims = [v for k, v in input_dims.items() if "rtg" in k]
        assert len(action_dims) == 1, "There should only be one action factor"
        assert len(rtg_dims) == 1, "There should only be one rtg factor"
        return state_dims, action_dims[0], rtg_dims[0]

    @staticmethod
    def validate_training_params(tp):
        # Do sanity checks if we're in the DT case
        lw = tp["loss_weights"]
        assert tp["stacked"], (
            "Stacked / non-stacked doesn't mean anything for DT. We always input states, actions, rtgs separately. "
            "Counterintuitively, we currently require stacked to be set to True for everything to work right."
        )
        assert "rtg" in lw
        assert lw["rtg"] == 0, "The original DT does not predict RTG"
        assert all(lw[k] == 0 for k in lw if "state" in k), "Original DT does not predict s"
        assert lw["action"] != 0
        # iterating over strings for more informative assert message.
        for bc_n_str in [
            "train_batch_params_n",
            "val_batch_params_n",
            "rew_batch_params_n",
        ]:
            for bc in tp[bc_n_str]:
                assert (
                    bc["type"] == DTActionPred
                ), f"{bc_n_str} has type incompatible with DT model. Found {bc['type']}, expected only DTActionPred."

    def forward(self, batch):
        """
        Annoyingly with DT, it's not easy to use the batch class model_input directly, as we do with FlexiBiT.
        We want to embed each factor separately. This is because of the way we want to embed things as done in the
        original DT paper.
        """
        # Differences:
        # - get_batch has weird padding
        # - we randomize over masking way more

        assert batch.__class__ in [DTActionPred, FuturePred]
        # TODO: figure out a better way to do this that doens't involve mask_nans. Were necessary because at reward
        #  evaluation time we were passing in nans for timesteps that weren't reached yet
        # TODO: check whether cloning the input here is necessary. Otherwise remove
        states_k = [
            batch.get_masked_input_factor(k, mask_nans=True).float().clone().to(uniMASK.utils.DEVICE) for k in self.state_dims
        ]
        actions = batch.get_masked_input_factor("action", mask_nans=True).float().clone().to(uniMASK.utils.DEVICE)

        # NOTE: have to do this to make RC_fixed masking scheme. That's because it will automatically copy that
        #  rtg masking scheme to the eval. At t=0, all rtgs except for the first will have never been seen before.
        #  So they'll be nans. And that fucks things up. And so we change them to something else, but given that they
        #  come later in the sequence, the model can't even attend to them so it's fine 🤷
        rtgs = batch.get_factor("rtg").input.clone().to(uniMASK.utils.DEVICE)
        rtgs[rtgs.isnan()] = MASKED_VALUE  # Checked that this doesn't affect anything
        mask = batch.get_input_mask_for_factor("rtg")
        masked_input = rtgs.float()
        masked_input[mask == 0] = MASKED_VALUE
        assert not masked_input.isnan().any()
        returns_to_go = masked_input

        batch_size, seq_length = actions.shape[0], actions.shape[1]
        assert seq_length == self.seq_len

        # attention mask for GPT: 1 if can be attended to, 0 if not
        # Whatever was nan in the input, remove it from the attention mask
        actions_att_mask = 1 - actions[:, :, 0].isnan().int()
        returns_att_mask = 1 - returns_to_go[:, :, 0].isnan().int()
        states_att_mask_k = [1 - states[:, :, 0].isnan().int() for states in states_k]

        # Also checked that value here doesn't affect anything
        actions[actions[:, :, 0].isnan(), :] = MASKED_VALUE
        for states in states_k:
            states[states[:, :, 0].isnan(), :] = MASKED_VALUE
        returns_to_go[returns_to_go[:, :, 0].isnan(), :] = MASKED_VALUE

        # embed each modality with a different head
        state_embeds_k = [embed_fn(states) for embed_fn, states in zip(self.embed_state_k.values(), states_k)]
        action_embeds = self.embed_action(actions)
        rtg_embeds = self.embed_return(returns_to_go)
        if self.timestep_encoding:
            timesteps = batch.get_factor("timestep").input
            timesteps = timesteps.squeeze(2).long().to(uniMASK.utils.DEVICE)
            time_embeddings = self.embed_timestep(timesteps)
            # time embeddings are treated similar to positional embeddings
            state_embeds_k = [state_embeds + time_embeddings for state_embeds in state_embeds_k]
            action_embeds = action_embeds + time_embeddings
            rtg_embeds = rtg_embeds + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack((rtg_embeds, *state_embeds_k, action_embeds), dim=1)
        n_inputs_per_t = stacked_inputs.shape[1]
        expected_inputs_per_t = len(self.input_dims) - int(self.timestep_encoding)
        assert n_inputs_per_t == expected_inputs_per_t
        assert stacked_inputs.shape == (
            batch_size,
            n_inputs_per_t,
            self.seq_len,
            self.embed_dim,
        )
        stacked_inputs = stacked_inputs.permute(0, 2, 1, 3).reshape(
            batch_size, n_inputs_per_t * self.seq_len, self.embed_dim
        )

        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((returns_att_mask, *states_att_mask_k, actions_att_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, n_inputs_per_t * seq_length)
        )

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs["last_hidden_state"]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, n_inputs_per_t, self.embed_dim).permute(0, 2, 1, 3).to(
            uniMASK.utils.DEVICE)

        ###################
        # Get predictions #
        ###################
        # NOTE: currently ignoring return and state predictions, as in original DT paper

        # Predict next return given state and action.
        # Will be in last position because that's the first thing we include when we stack the inputs.
        return_preds = self.predict_return(x[:, -1])

        # Predict next state given state and action
        # NOTE: haven't tested that this works, as we never actually use the state and rtg predictions
        state_preds_k = [
            pred_fn(x[:, pred_idx])
            for pred_fn, pred_idx in zip(self.predict_state_k.values(), range(len(self.state_dims)))
        ]

        # Predict next action given state. This will always correspond to the second to last position, as the last
        # position will predict the first factor from the next timestep. Another way of thinking about it is that the
        # last position in the input is the action, so the second to last output will be predicting the last input.
        action_preds = self.predict_action(x[:, -2])

        # De-normalize (this might also help?)
        # state_preds = (state_preds + self.s_mean) * self.s_std
        # action_preds = (action_preds + self.a_mean) * self.a_std
        # return_preds = (return_preds + self.rtg_mean) * self.rtg_std

        input_factors = [f for f in batch.input_data.factor_names if f != "timestep"]
        assert np.all(input_factors[:2] == ["action", "rtg"]), "order matters"
        assert input_factors[2:] == list(self.state_dims.keys())
        output = torch.cat([action_preds, return_preds, *state_preds_k], axis=2)
        batch.add_model_output(output)


class PositionalEncoding(nn.Module):
    """Taken from the PyTorch transformers tutorial"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


STR_TO_MODEL_CLASS = {
    "FB": uniMASKModel,
    "NN": FlexiNNModel,
    "DT": DecisionTransformer,
}
