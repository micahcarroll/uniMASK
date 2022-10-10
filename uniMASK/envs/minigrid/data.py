from collections import defaultdict
from copy import deepcopy

import torch

from uniMASK.sequences import MISSING_VALUE, FullTokenSeq


def r_to_r_idx(env, r):
    return r + env.REW_OFFSET


def r_idx_to_r(env, r):
    return int(r - env.REW_OFFSET)


def rtg_to_rtg_idx(env, rtg):
    return rtg + env.RTG_OFFSET


def rtg_idx_to_rtg(env, rtg_idx):
    return int(rtg_idx - env.RTG_OFFSET)


def r_to_rtg(r):
    """Converts rewards to rewards to go. Expects tensor of size [num_trajs, traj_len]"""
    new_r = torch.zeros_like(r)
    seq_len = r.shape[1]
    for t in range(seq_len):
        new_r[:, t] = r[:, t:].sum(dim=1)
    return new_r


def filter_data_by(env, data, factor_name, query):
    """Interface to the filter_by method of the data itself

    query = {
        0: (1, 1),
        9: (4, 4)
    }
    matching_data, indices = filter_data_by(train_data, "state", query)

    is equivalent to:

    matching_data, indices = train_data.filter_by("state", [POS_TO_IDX[(1, 1)], POS_TO_IDX[(4, 4)]], [0, 9])
    """
    assert isinstance(data, FullTokenSeq)
    timesteps = list(query.keys())
    locations = list(query.values())
    idx_locations = [env.POS_TO_IDX[xy_pos] for xy_pos in locations]
    return data.filter_by(factor_name, idx_locations, timesteps)


def get_s_a_r_matrices_from_data(data):
    """Get the raw environment data from the FullTokenSeq"""
    factors = data.raw_data
    return (
        deepcopy(factors["state"].float()),
        deepcopy(factors["action"].float()),
        deepcopy(factors["reward"].float()),
        deepcopy(factors["rtg"].float()),
    )


def get_s_a_matrices_from_data(data):
    """Get the raw environment data from the FullTokenSeq"""
    factors = data.raw_data
    return factors["state"].float(), factors["action"].float()


def get_sa_transition_counter(token_seqs):
    """Given a token seq, returns a counter which keeps track of how many times each (s, a) pair was encountered"""
    sa_transition_counter = defaultdict(lambda: defaultdict(int))

    states = token_seqs.get_factor("state").inputs_hr
    actions = token_seqs.get_factor("action").inputs_hr

    for traj_states, traj_acts in zip(states, actions):
        for s, a in zip(traj_states, traj_acts):
            sa_transition_counter[s.item()][a.item()] += 1
    return sa_transition_counter


def get_s_s_prime_transition_counter(token_seqs):
    """Currently not used. The (s, s') version of the above."""
    transition_counter = defaultdict(lambda: defaultdict(int))
    for traj_states in token_seqs.get_factor("state").get_targets():
        traj_states = [s.item() for s in traj_states]
        prev_s = None
        for curr_s in traj_states:
            if prev_s is None:
                prev_s = curr_s
                continue
            transition_counter[prev_s][curr_s] += 1
            prev_s = curr_s
    return transition_counter


def zero_out_timesteps(data, timesteps_to_mask, stacked=True):
    """
    Args:
        timesteps_to_mask (dict[str, list]): for each of "state", "action", "rtg", list of indices to mask out
    """
    new_data = {}
    for factor_name, mat in data.raw_data.items():
        # e.g. for state_key_pos
        base_factor_name = factor_name.split("_")[0]
        factor_masks_name = base_factor_name + "_masks"

        if factor_masks_name not in timesteps_to_mask:
            # If the raw_data has some extra factors that we haven't provided masks for, should skip
            continue

        # (batch, num_timesteps, num_one_hot_classes)
        factor_data = deepcopy(mat.float())
        factor_data[:, timesteps_to_mask[factor_masks_name], :] = MISSING_VALUE
        new_data[factor_name] = factor_data
    return FullTokenSeq.raw_data_to_token_seq(new_data, stacked=stacked)


def zero_out_until(data, curr_t_to_pred, stacked=True):
    """Will zero out all indices up _and including_ curr_t_to_pred"""
    new_data = {}
    for factor_name, mat in data.raw_data.items():
        factor_data = deepcopy(mat.float())
        factor_data[:, : curr_t_to_pred + 1, :] = MISSING_VALUE
        new_data[factor_name] = factor_data
    return FullTokenSeq.raw_data_to_token_seq(new_data, stacked=stacked)
