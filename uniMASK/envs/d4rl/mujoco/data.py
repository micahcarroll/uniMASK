import os.path

from uniMASK.data import DATASET_DIR
from uniMASK.envs.d4rl.d4rl_data import D4RLDataset


class MujocoDataset(D4RLDataset):
    @classmethod
    def get_trajs_from_d4rl_dataset(
        cls, env_name, dataset_info, proportion=None, num_trajs=None, used_indices=()
    ):
        assert not (proportion and num_trajs), "Either select a prop or a num of trajs"
        expert_type = dataset_info["expert_type"]
        dataset_path = os.path.join(DATASET_DIR, f"{env_name}-{expert_type}-v2.pkl")
        return cls.get_trajs_from_d4rl_dataset_path(
            dataset_path, num_trajs, proportion, used_indices
        )
