import argparse
import collections
import pickle

import gym
import numpy as np


def pkl_dataset(dataset, next_obs, output_name):

    N = len(dataset["rewards"])
    data_ = collections.defaultdict(list)

    use_timeouts = "timeouts" in dataset

    episode_step = 0
    paths = []
    for i in range(N):
        done_bool = bool(dataset["terminals"][i])
        final_timestep = dataset["timeouts"][i] if use_timeouts else episode_step == N - 1

        if next_obs:
            keys = [
                "observations",
                "next_observations",
                "actions",
                "rewards",
                "terminals",
            ]
        else:
            keys = [
                "observations",
                "actions",
                "rewards",
                "terminals",
            ]

        if "infos/s_qvel" in data_:
            keys.append("infos/s_qvel")
            keys.append("infos/goal")

        for k in keys:
            data_[k].append(dataset[k][i])
        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            paths.append(episode_data)
            data_ = collections.defaultdict(list)
        episode_step += 1

    print(f"{output_name}.pkl")
    with open(f"{output_name}.pkl", "wb") as f:
        pickle.dump(paths, f)
    return paths


def download_dataset(name):
    env = gym.make(name)
    return env.get_dataset()


MUJOCO = [
    f"{env_name}-{dataset_type}-v2"
    for env_name in ["halfcheetah", "hopper", "walker2d"]
    for dataset_type in ["medium", "medium-replay", "expert"]
]

MAZE = [
    f"maze2d-{env_name}-{density}v{version}"
    for env_name, version in [("open", 0), ("umaze", 1), ("medium", 1), ("large", 1)]
    for density in ["", "dense-"]
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--download", type=str, choices=["mujoco", "maze", "debug_maze"])
    group.add_argument("--hdf5", type=str, help="Path to hdf5 file.")
    args = vars(parser.parse_args())
    if args["download"]:
        if args["download"] == "mujoco":
            envnames, next_obs = MUJOCO, True
        elif args["download"] == "maze":
            envnames, next_obs = MAZE, False
        else:
            raise NotImplementedError
        for name in envnames:
            pkl_dataset(download_dataset(name), next_obs=next_obs, output_name=name)
    else:
        raise NotImplementedError
