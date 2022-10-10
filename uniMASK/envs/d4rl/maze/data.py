import os.path

from uniMASK.data import DATASET_DIR
from uniMASK.envs.d4rl.mujoco.data import MujocoDataset


class MazeDataset(MujocoDataset):
    @classmethod
    def get_trajs_from_d4rl_dataset(
        cls, env_name, dataset_info, proportion=None, num_trajs=None, used_indices=()
    ):
        assert not (proportion and num_trajs), "Either select a prop or a num of trajs"
        horizon = dataset_info["horizon"]
        dataset_name = f"{env_name}-h{horizon}.pkl"
        dataset_path = os.path.join(DATASET_DIR, dataset_name)
        return cls.get_trajs_from_d4rl_dataset_path(
            dataset_path, num_trajs, proportion, used_indices
        )
