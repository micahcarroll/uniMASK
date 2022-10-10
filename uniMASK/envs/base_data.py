import numpy as np
import torch

from uniMASK.sequences import FullTokenSeq
from uniMASK.utils import load_pickle, save_pickle


class Dataset:
    def __init__(self, data, traj_indices=None, compute_means_and_stds=False):
        assert type(data) is dict
        self.data = data
        self.data_keys = list(data.keys())
        for k, v in self.data.items():
            self.__dict__[k] = v
            # if k in ["state", "action", "rtg"] and compute_means_and_stds:
            # if type(v) is list:
            #     v = torch.cat(v, dim=0)
            # v_std = v.float().std(dim=0)
            # v_mean = v.float().mean(dim=0)
            # self.__dict__[f"{k}_mean"] = v_mean
            # TODO: evenutally add something here to warn about small std
            # if v.shape[0] > 1:
            #     print(
            #         f"STD for {k} IS VERY LOW. If normalizing, might cause overflow errors",
            #         torch.all(v_std > 1e-5),
            #     )
            # self.__dict__[f"{k}_std"] = v_std + 1e-6

        assert all(
            len(data[k]) == len(data[self.data_keys[0]]) for k in self.data_keys
        ), "The num of trajs should be equal across (states, actions, etc)"

        self.num_trajs = len(self.data[self.data_keys[0]])
        self.traj_lengths = self._get_traj_lengths()
        self.total_timesteps = sum(self.traj_lengths)
        self.traj_indices = traj_indices

    def validate_seq_len(self, seq_len):
        assert np.all(
            self.traj_lengths >= seq_len
        ), "There are some trajectories that are shorter than this seq_len. Can't form sequence from them trivially"

    def get_num_possible_seqs_by_traj(self, seq_len):
        """Returns a list with the number of possible unique sequences of seq_len which can be created from each traj"""
        self.validate_seq_len(seq_len)
        possible_seqs_by_traj = []
        for l in self.traj_lengths:
            possible_seqs_by_traj.append(l - seq_len + 1)
        return np.array(possible_seqs_by_traj)

    def get_tot_num_possible_seqs(self, seq_len):
        """
        Returns the total number of unique sequences of len seq_len which can be formed from the dataset
        NOTE: this doesn't take into account possible maskings, just the starting and ending positions
        """
        return self.get_num_possible_seqs_by_traj(seq_len).sum()

    def __getitem__(self, key):
        data_subset = {k: v[key] for k, v in self.data.items()}
        return self.__class__(data=data_subset)

    def split_data(self, train_prop=None, num_train_trajs=None, num_val_trajs=None, info=True):
        """
        Returns split data.
        NOTE: if you vary the amount of data and use `split_data`
         you will have a varying size validation set, which is likely not wanted.
         Instead of using this method, it's better to subclass the Dataset class and use
         the create_dataset method (similar to the one in D4RLDataset) which offers more flexibility.
        """
        assert train_prop is None or num_train_trajs is None
        if num_train_trajs is not None:
            train_prop = num_train_trajs / float(self.num_trajs)
        train_size = max(int(self.total_timesteps * train_prop), 1)

        # Naive way of doing this
        count, i = 0, 0
        while count < train_size:
            count += self.traj_lengths[i]
            i += 1
            remaining_trajs = self.num_trajs - i
            if num_val_trajs and remaining_trajs <= num_val_trajs:
                assert (
                    remaining_trajs == num_val_trajs
                ), "You requested a validation set size that's incompatible with the training prop selected"
                break
        assert i < len(self.traj_lengths) - 1, "You probably want to have something in the validation set"

        if num_val_trajs is not None:
            val_start_idx = -num_val_trajs
        else:
            val_start_idx = i

        train_data, test_data = self[:i], self[val_start_idx:]

        if info:
            print(f"Train trajs {train_data.num_trajs}\tVal trajs {test_data.num_trajs}")
        return train_data, test_data

    @staticmethod
    def print_split_info(train_data, test_data):
        print(f"Train trajs {train_data.num_trajs}\tVal trajs {test_data.num_trajs}")
        if "reward" in train_data.data_keys:
            avg_train_rew = np.mean([r.sum().numpy() for r in train_data.reward])
            avg_test_rew = np.mean([r.sum().numpy() for r in test_data.reward])
            max_train_rew = np.max([r.sum().numpy() for r in train_data.reward])
            max_test_rew = np.max([r.sum().numpy() for r in test_data.reward])
            print(
                f"Average traj reward:"
                f"\n\ttrain\t {avg_train_rew} \ttest {avg_test_rew}"
                f"\n\tmax\t\t {max_train_rew} \t\t {max_test_rew}"
            )

    def _get_traj_lengths(self):
        traj_lengths = []
        for i in range(self.num_trajs):
            single_traj_data = [self.data[k][i] for k in self.data_keys]
            assert all(
                len(traj_feature) == len(single_traj_data[0]) for traj_feature in single_traj_data
            ), "The length of states, actions, etc. should match for a single traj"
            traj_lengths.append(len(single_traj_data[0]))
        return np.array(traj_lengths)

    def save(self, path):
        save_pickle(self.data, path)
        print("Saved dataset at", path)

    @classmethod
    def load(cls, path):
        print("Loading dataset from", path)
        return cls(load_pickle(path))

    def to_token_seq(self, input_keys=None, loss_types=None, stacked=True):
        """
        Generate a dataset of n trajs directly in a FullTokenSeq format.

        Stacked is a flag for what formatting the tokens should be in.
        """
        # Now that all sequences will be same length so we can convert to tensor
        raw_data = {k: torch.cat([v_prime.unsqueeze(0) for v_prime in v]) for k, v in self.data.items()}
        return FullTokenSeq.raw_data_to_token_seq(raw_data, stacked, input_keys, loss_types)

    def cat_t_and_rtg(self, horizon):
        """
        Returns a new dataset instance in which the rtg and t information have been concatenated, so that
        now rtg is 2-dimensional. NOTE: this overwrites rtg.
        NOTE: horizon should be the same as the max_ep_len at eval time.
        """
        # We could either compute this for every dataset at dataset creation time, or compute it on the fly here.
        # Given that we have this replace the "rtg" in the dataset, no point computing it ahead of time.
        rtg_cat_t = [torch.cat([r, t / horizon], dim=-1) for r, t in zip(self.rtg, self.timestep)]
        data = {k: v for k, v in self.data.items()}
        data["rtg"] = rtg_cat_t
        return self.__class__(data, traj_indices=self.traj_indices)

    def get_rnd_batch(self, batch_size, seq_len, input_keys, loss_types, stacked=True, rew_scale=1):
        """
        Takes the dataset and selects a subset of it to form a batch.
        """
        # Sample trajectory indices weighted by the traj length
        sampling_p = self.traj_lengths / self.total_timesteps
        traj_indices_for_batch = np.random.choice(
            np.arange(self.num_trajs), size=batch_size, replace=True, p=sampling_p
        )

        raw_data = {k: [] for k in self.data_keys}

        # Given that doing np.random.choice batch_size times is a major source of slowness, but the
        # sequence lengths can change, sample uniforms here and then discretize later on the right range.
        window_starts_prop = np.random.uniform(0, 1, size=batch_size)

        # For each of the sampled traj indices, select a random window within it
        for i in range(batch_size):
            curr_traj_idx = traj_indices_for_batch[i]
            curr_traj_len = self.traj_lengths[curr_traj_idx]
            assert curr_traj_len >= seq_len

            curr_traj_data = {k: self.data[k][curr_traj_idx] for k in self.data_keys}

            possible_window_starts = list(range(0, curr_traj_len - seq_len + 1))

            window_start_idx = np.floor(window_starts_prop[i] * len(possible_window_starts))
            window_start = possible_window_starts[int(window_start_idx)]
            window_end = window_start + seq_len

            for k in self.data_keys:
                raw_data[k].append(curr_traj_data[k][window_start:window_end])

        for k in self.data_keys:
            raw_data[k] = torch.stack(raw_data[k])

        # TODO: move this to the transfomer class
        if "rtg" in raw_data:
            raw_data["rtg"] = raw_data["rtg"].float()
            raw_data["rtg"] /= rew_scale

        return self.__class__(raw_data).to_token_seq(input_keys, loss_types, stacked)

    def get_full_data_batch(self, seq_len, input_keys=None, loss_types=None, stacked=True, rew_scale=1):
        """
        Takes the dataset and returns all possible sequences as a single batch.
        """
        self.validate_seq_len(seq_len)

        raw_data = {k: [] for k in self.data_keys}
        for traj_idx in range(self.num_trajs):
            traj_data = {k: self.data[k][traj_idx] for k in self.data_keys}
            traj_len = self.traj_lengths[traj_idx]
            last_seq_start_idx = traj_len - seq_len

            for window_start in range(last_seq_start_idx + 1):
                window_end = window_start + seq_len
                for k in self.data_keys:
                    raw_data[k].append(traj_data[k][window_start:window_end])

        assert len(raw_data[k]) == self.get_tot_num_possible_seqs(seq_len)

        # TODO: move this to the transfomer class
        if "rtg" in raw_data:
            raw_data["rtg"] = torch.stack(raw_data["rtg"]).float()
            raw_data["rtg"] /= rew_scale

        return self.__class__(raw_data).to_token_seq(input_keys, loss_types, stacked)
