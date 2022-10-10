

from copy import deepcopy

import numpy as np
from torch import tensor as tt

from uniMASK.batches import Batch, CustomPred
from uniMASK.envs.minigrid.data import r_to_r_idx, zero_out_timesteps
from uniMASK.envs.minigrid.env import CustomDoorKeyEnv6x6, make_env


def get_forward_completion(
    env,
    data,
    masks,
    trainer,
    input_keys,
    stacked=True,
    argmax=True,
    debug_print=True,
):
    """
    Completes trajectory conditioned on arbitrary timesteps forward in time, starting from the first missing action timestep. Handles inconsistencies by overwriting the conditioned value.

    Args:
        data: unmasked data (will be masked with mask)
        masks (dict[str, np.array]): dict with keys "state_masks", "action_masks", "rtg_masks", each value a numpy array
            containing masks for each timestep
        mask_rtg (bool): True to mask RTG at all timesteps, False will unmask the first one
    """
    masks = deepcopy(masks)

    # Zero out the masked timesteps
    data = zero_out_timesteps(
        data,
        {k: np.argwhere(v == 0).flatten() for k, v in masks.items()},
        stacked=stacked,
    )

    first_missing_act_t = min(data.get_factor("action").missing_ts)

    for next_missing_act_t in range(first_missing_act_t, data.seq_len):
        # Predict next action
        samples, sample_probs = sample_first_action_prediction_for_timestep(
            data,
            masks,
            trainer,
            input_keys,
            argmax_pred=argmax,
        )

        data = data.with_added_missing_input("action", next_missing_act_t, samples, env.NUM_ACTIONS)

        if debug_print:
            fwd_debug_print(data, next_missing_act_t, sample_probs)

        # Query env for next state
        next_missing_obs_t = next_missing_act_t + 1
        if next_missing_obs_t < data.seq_len:
            # NOTE: Zero out next timestep to fill in. This is equivalent to fixing inconsistencies by overwriting
            data = zero_out_timesteps(
                data,
                {
                    "state_masks": [next_missing_obs_t],
                    "action_masks": [],
                    "rtg_masks": [next_missing_obs_t],
                    "reward_masks": [next_missing_obs_t],
                },
                stacked=stacked,
            )
            data = get_new_obs_forwards(env, data, next_missing_obs_t)

            if debug_print:
                fwd_debug_print(data, next_missing_act_t, sample_probs)

            # Update mask for next CustomPred forward pass
            masks["action_masks"][next_missing_act_t] = 1
            masks["state_masks"][next_missing_obs_t] = 1
            # Note: with new rtg training, only the first RTG is ever unmasked
            #  no update for rtg mask needed

    assert data.get_factor("action").missing_ts == []
    assert data.get_factor("state").missing_ts == []
    return data


def fwd_debug_print(data, next_missing_act_t, sample_probs=None):
    print(f"Predicted action at time {next_missing_act_t}")
    print("State: ", data.get_factor("state").inputs_hr)
    print("Act: ", data.get_factor("action").inputs_hr)
    if sample_probs is not None:
        print("Act sample probs:", sample_probs)
    print("RTG: ", data.get_factor("rtg").inputs_hr)


def get_new_obs_forwards(env, data, first_missing_obs_t):
    """
    Given the current observations, actions, and rewards which will have missing information,
    get the next action (forwards) by querying the ground truth environment dynamics.
    """
    env_type = "key" if isinstance(env.base_env, CustomDoorKeyEnv6x6) else "empty"

    s_factor = data.get_factor("state")
    a_factor = data.get_factor("action")

    if env_type == "key":
        # Get key position for all batch elems
        s_key_pos_factor = data.get_factor("state_key_pos")
        s_key_factor = data.get_factor("state_key")

    assert min(s_factor.missing_ts) == first_missing_obs_t
    assert min(a_factor.missing_ts) == first_missing_obs_t

    # Last t for which we have info
    curr_t = first_missing_obs_t - 1

    new_s_n = []
    new_rew_n = []
    new_key_state_n = []
    new_key_pos_n = []

    for traj_idx in range(data.num_seqs):
        curr_a_idx = a_factor.inputs_hr[traj_idx, curr_t].int().item()
        curr_s_idx = s_factor.inputs_hr[traj_idx, curr_t].int().item()
        curr_agent_pos = np.array(env.IDX_TO_POS[curr_s_idx])

        if env_type == "key":
            curr_key_pos_idx = s_key_pos_factor.inputs_hr[traj_idx, curr_t].int().item()
            curr_key_pos = env.IDX_TO_POS[curr_key_pos_idx]
            curr_key_picked_up = s_key_factor.inputs_hr[traj_idx, curr_t].int().item()

        # Make dummy env to query dynamics
        env = make_env(env_type)
        env.reset()

        # Set agent position manually and get next state from `step`
        env.set_agent_pos(curr_agent_pos)
        if env_type == "key":
            env.set_key_position(curr_key_pos)
            env.set_key_state(curr_key_picked_up)

        _, new_rew, _, _ = env.step(curr_a_idx)
        new_s_idx = tt([env.POS_TO_IDX[tuple(env.agent_pos)]])
        new_s_n.append(new_s_idx)

        # Note: with new rtg training, only the first RTG is ever unmasked, so we don't need to calculate updated RTG
        new_rew_idx = tt([r_to_r_idx(env, new_rew)])
        new_rew_n.append(new_rew_idx)

        if env_type == "key":
            # Update key state given new agent position
            new_key_pos = env.get_key_position()
            new_key_state = tt([int(new_key_pos is None)])
            new_key_state_n.append(new_key_state)

            if new_key_pos is None:
                # Same as agent pos
                new_key_pos_idx = new_s_idx
            else:
                # Same as previous timestep
                new_key_pos_idx = env.POS_TO_IDX[tuple(new_key_pos)]
            new_key_pos_n.append(new_key_pos_idx)

    new_data = data.with_added_missing_input("state", first_missing_obs_t, tt(new_s_n), env.NUM_STATES)

    # Not used for model typically, just for visualization / debugging output
    new_data = new_data.with_added_missing_input("reward", first_missing_obs_t, tt(new_rew_n), env.NUM_REWARDS)
    if env_type == "key":
        # Update key position
        new_data = new_data.with_added_missing_input(
            "state_key_pos", first_missing_obs_t, tt(new_key_pos_n), env.NUM_STATES
        )
        # Update key state
        new_data = new_data.with_added_missing_input(
            "state_key", first_missing_obs_t, tt(new_key_state_n), env.NUM_KEY_STATES
        )

    return new_data


def sample_first_action_prediction_for_timestep(data, masks, trainer, input_keys, argmax_pred=True):
    # Get only the subset of data we'll feed into the model (although we may fill in some of these later, like reward)
    data = data.get_factor_subset(input_keys)

    # Get network output
    b_params = deepcopy(masks)
    b_params["type"] = CustomPred
    b = Batch.get_dummy_batch_output(data, batch_params=b_params, trainer=trainer)

    # Get specific predictions we care about
    first_missing_act_t = min(np.argwhere(masks["action_masks"] == 0)).item()
    return b.get_factor("action").sample_timestep_predictions(num_samples=1, t=first_missing_act_t, argmax=argmax_pred)


def get_backward_completion(
    env,
    data,
    masks,
    trainer,
    input_keys,
    stacked=True,
    argmax=False,
    debug_print=True,
    consistency_checks=True,
):
    """
    Completes trajectory starting from the max timestep.
    Note `mask` can mask anything, not just all timesteps [0, seq_len) -- e.g. we can do goal-conditioned backwards prediction.
    Currently assumes that at least the last state s_t, t=(seq_len - 1), is provided.

    Returns if inconsistency is encountered (no resampling).
    """
    assert data.num_token_seqs == 1, "untested for batch size > 1."
    env_type = "key" if isinstance(env.base_env, CustomDoorKeyEnv6x6) else "empty"
    seq_len = data.shape[1]
    masks = deepcopy(masks)
    data = zero_out_timesteps(
        data,
        {k: np.argwhere(v == 0).flatten() for k, v in masks.items()},
        stacked=stacked,
    )

    # Next timestep could be an action or a state, depending on what is missing
    #  e.g. given (s_t, a_t) --> predict a_{t-1}
    #       given (a_t) --> predict s_t
    next_t_to_pred = max(
        max(data.get_factor("action").missing_ts),
        max(data.get_factor("state").missing_ts),
    )

    for t in reversed(range(0, next_t_to_pred + 1)):
        print(f"== t={t} ==")
        # NOTE: using the same model for both states and actions
        # If the last action is missing, predict it
        if masks["action_masks"][t] == 0:
            act_samples, act_samples_probs = sample_last_t_missing_for_factor(
                data,
                masks,
                trainer,
                input_keys,
                factor_name="action",
                argmax=argmax,
            )
            data = data.with_added_missing_input("action", t, act_samples, env.NUM_ACTIONS)
            masks["action_masks"][t] = 1
            if debug_print:
                print("Filled in ac at t=", t)
                print("act_samples_probs: ", act_samples_probs)
                print("sampled a=", act_samples.item())
                print("Mask after ac: ", masks)

        # Get new state backwards
        if masks["state_masks"][t] == 0:
            # state_samples: (batch_size, num_samples=1)
            # state_samples_probs: (batch_size, num_states)
            state_samples, state_probs = sample_last_t_missing_for_factor(
                data,
                masks,
                trainer,
                input_keys,
                factor_name="state",
                argmax=argmax,
            )
            data = data.with_added_missing_input("state", t, state_samples, env.NUM_STATES)

            if env_type == "key":
                key_pos_samples, key_pos_probs = sample_last_t_missing_for_factor(
                    data,
                    masks,
                    trainer,
                    input_keys,
                    factor_name="state_key_pos",
                    argmax=argmax,
                )
                data = data.with_added_missing_input("state_key_pos", t, key_pos_samples, env.NUM_STATES)

                # Manually calculate key state
                is_holding_key = key_pos_samples[0] == state_samples[0]

                # Copy key position
                data = data.with_added_missing_input("state_key", t, tt([int(is_holding_key)]), env.NUM_KEY_STATES)

            # If there's a state after this, verify that the transition just defined is consistent
            if t + 1 < seq_len and consistency_checks:
                next_s = data.get_factor("state").inputs_hr[:, t + 1][0]
                if not env.is_transition_valid(
                    state_samples[0].item(),
                    act_samples[0],
                    next_s.item(),
                    is_holding_key=is_holding_key,
                ):
                    print(
                        "INVALID!!!: ",
                        state_samples[0].item(),
                        act_samples[0],
                        next_s.item(),
                        is_holding_key,
                    )
                    return  # state_samples, act_samples, next_s

            masks["state_masks"][t] = 1
            if debug_print:
                print()
                print("Filled in state at t=", t)
                print("state_samples_probs: ", state_probs)
                print("sampled s=", state_samples.item())
                print("Mask after state: ", masks)

    return data


def sample_last_t_missing_for_factor(data, masks, trainer, input_keys, factor_name, argmax=False):
    """
    Given some data which is missing some states and actions, predict the factor passed in, at the last missing index
    """
    assert factor_name in ("state", "state_key_pos", "action")
    data = data.get_factor_subset(input_keys)
    # Minor hack to make the state mask also be applied to state_key_pos
    mask_name = "state_masks" if factor_name[:5] == "state" else f"{factor_name}_masks"

    # Get network output
    b_params = deepcopy(masks)
    b_params["type"] = CustomPred
    b = Batch.get_dummy_batch_output(data, batch_params=b_params, trainer=trainer)

    # Get specific predictions we care about
    last_missing_t = max(np.argwhere(masks[mask_name] == 0)).item()
    return b.get_factor(factor_name).sample_timestep_predictions(num_samples=1, t=last_missing_t, argmax=argmax)
