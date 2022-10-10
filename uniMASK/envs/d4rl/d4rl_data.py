import pickle
from random import shuffle

import numpy as np
import torch
from torch import tensor as tt

from uniMASK.envs.base_data import Dataset
from uniMASK.utils import append_dictionaries

MAZE_NAMES = {
    "open": "maze2d-open-v0",
    "umaze": "maze2d-umaze-v1",
    "medium": "maze2d-medium-v1",
    "large": "maze2d-large-v1",
    "open-dense": "maze2d-open-dense-v0",
    "umaze-dense": "maze2d-umaze-dense-v1",
    "medium-dense": "maze2d-medium-dense-v1",
    "large-dense": "maze2d-large-dense-v1",
}

MUJOCO_NAMES = {
    "halfcheetah": "HalfCheetah-v3",
    "hopper": "Hopper-v3",
    "walker2d": "Walker2d-v3",
}

MUJOCO_GYM_ENV_NAMES = dict(MUJOCO_NAMES)
MUJOCO_GYM_ENV_NAMES.update(MAZE_NAMES)


class D4RLDataset(Dataset):
    @classmethod
    def get_datasets(
        cls, env_name, dataset_info, prop, num_train_trajs=None, leave_for_eval=None
    ):
        """
        Gets training and test datasets from the corresponding Dataset classes
        """
        assert prop is None or num_train_trajs is None
        train_data = cls.create_dataset(
            env_name,
            dataset_info,
            prop=prop,
            num_trajs=num_train_trajs,
            leave_for_eval=leave_for_eval,
        )

        # Hack used for testing
        if leave_for_eval == 0:
            return train_data, None

        test_data = cls.create_dataset(
            env_name,
            dataset_info,
            num_trajs=leave_for_eval,
            used_indices=train_data.traj_indices,
        )
        return train_data, test_data

    @classmethod
    def create_dataset(
        cls,
        dataset_params,
        dataset_info,
        prop=None,
        num_trajs=None,
        used_indices=(),
        leave_for_eval=False,
    ):
        """
        Creates a dataset instance with data from the environment specified, with the specified expert type.
        One can also specify the proportion of the data or the number of trajectories that one wants to use for creating
        the dataset.

        `used_indices` removes the those trajectory indices from the sampling of the current dataset. Useful if sampling
        a test set that we don't want to have repeats of what is in the training set.
        """
        trajs, indices, tot_num_trajs = cls.get_trajs_from_d4rl_dataset(
            env_name=dataset_params,
            dataset_info=dataset_info,
            proportion=prop,
            num_trajs=num_trajs,
            used_indices=used_indices,
        )
        cut_num_trajs = len(trajs)
        remaining_trajs = tot_num_trajs - cut_num_trajs
        if leave_for_eval and remaining_trajs < leave_for_eval:
            trajs_too_many = leave_for_eval - remaining_trajs
            trajs = trajs[:-trajs_too_many]
            indices = indices[:-trajs_too_many]
            print(
                f"Reducing number of training trajectories from the requested of {cut_num_trajs} to {len(trajs)} so that there are enough trajs for validation loss"
            )

        traj_dict = append_dictionaries(trajs)
        traj_dict = {k: [tt(v_prime) for v_prime in v] for k, v in traj_dict.items()}

        unsqueezed_timesteps = []
        unsqueezed_rews = []
        rtg_seqs_n = []
        for rew_seq in traj_dict["rewards"]:
            rtg_seqs_n.append(tt(r_to_rtg_not_vec(rew_seq)[:, np.newaxis]))
            unsqueezed_rews.append(rew_seq[:, np.newaxis])
            unsqueezed_timesteps.append(torch.arange(len(rew_seq))[:, np.newaxis])

        data_dict = {
            "state": traj_dict["observations"],
            "action": traj_dict["actions"],
            "terminals": traj_dict["terminals"],
            "reward": unsqueezed_rews,
            "rtg": rtg_seqs_n,
            "timestep": unsqueezed_timesteps,
        }
        if "infos/s_qvel" in traj_dict:
            data_dict["s_qvel"] = traj_dict["infos/s_qvel"]
            data_dict["goal"] = traj_dict["infos/goal"]

        return cls(data_dict, traj_indices=indices)

    @classmethod
    def get_trajs_from_d4rl_dataset_path(
        cls, dataset_path, num_trajs, proportion, used_indices
    ):
        with open(dataset_path, "rb") as f:
            # trajectories is list({'observations':, 'next_observations':, 'actions':, 'terminals':, 'rewards:'})
            trajectories = pickle.load(f)
        tot_num_trajs = len(trajectories)
        if proportion:
            # Using floor so that not going to be rounding up both training and test set size by 1 if using all data
            # Using max with 1 so that we don't create 0-trajectory datasets by accident
            n = max(int(np.floor(tot_num_trajs) * proportion), 1)
        elif num_trajs:
            n = num_trajs
        else:
            raise ValueError("")
        print("Getting {} trajs out of {}".format(n, len(trajectories)))
        # Only sample trajectories that have not been specified in `used_indices`
        traj_indices = [i for i in np.arange(tot_num_trajs) if i not in used_indices]
        assert (
            len(traj_indices) >= n
        ), f"Only {len(traj_indices)} left, wanted {n}, (tot {tot_num_trajs})"
        shuffle(traj_indices)
        chosen_traj_indices = traj_indices[:n]
        return (
            np.take(trajectories, chosen_traj_indices),
            chosen_traj_indices,
            tot_num_trajs,
        )

    def to_trajs(self):
        """Go from the Dataset class back to the normal Mujoco format"""
        s, a, r, d = self.state, self.action, self.reward, self.terminals
        trajs = []
        num_trajs = len(s)
        for traj_idx in range(num_trajs):
            traj_d = {
                "observations": s[traj_idx],
                "actions": a[traj_idx],
                "rewards": r[traj_idx],
                "terminals": d[traj_idx],
            }
            trajs.append(traj_d)
        return trajs

    @staticmethod
    def rtgs_for_same_len_trajs(traj_dict):
        obs = traj_dict["observations"]
        traj_length = len(obs[0])
        assert all(
            traj_length == len(obs[i]) for i in range(len(obs))
        ), "Trajs were not same len"
        # Calculate rewards to go
        upper_tri = tt(np.triu(np.ones((traj_length, traj_length)))).float()
        rtg_seqs_n = traj_dict["rewards"] @ upper_tri.T

        # NOTE: normalizing RTGs
        rtg_seqs_n = rtg_seqs_n - rtg_seqs_n.mean() / rtg_seqs_n.std()
        # rtg_seqs_n = torch.zeros_like(t["rewards"])
        return rtg_seqs_n

    @classmethod
    def get_trajs_from_d4rl_dataset(
        cls, env_name, dataset_info, proportion=None, num_trajs=None, used_indices=()
    ):
        """This method should be created by subclasses"""
        raise NotImplementedError()


def r_to_rtg_not_vec(r):
    """Converts rewards to rewards to go, not vectorized. Expects tensor of size [traj_len]"""
    new_r = np.zeros_like(r)
    for t, _ in enumerate(r):
        new_r[t] = r[t:].sum()
    return new_r
