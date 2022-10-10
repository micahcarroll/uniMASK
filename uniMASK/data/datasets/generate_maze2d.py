import argparse
import random

import gym
import numpy as np
from d4rl.pointmaze.waypoint_controller import WaypointController
from matplotlib import pyplot as plt
from tqdm import tqdm

from uniMASK.data.datasets.parse_d4rl_datasets import pkl_dataset
from uniMASK.envs.mujoco.maze_env import GymMazeEnv


class CustomWaypointController(WaypointController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.new_traj = True

    def get_action(self, location, velocity, target):
        # Fixes issue with reset if the goal is in the same location as before
        # in which waypoints don't get updated
        if self.new_traj:
            self._new_target(location, target)
            self.new_traj = False
        return super().get_action(location, velocity, target)

    def reset(self):
        self.new_traj = True


def reset_data():
    return {
        "observations": [],
        "actions": [],
        "terminals": [],
        "rewards": [],
        "infos/goal": [],
        "infos/qpos": [],
        "infos/qvel": [],
        "infos/s_qvel": [],
    }


def append_data(data, s, a, r, tgt, done, env_data, s_qvel):
    data["observations"].append(s)
    data["actions"].append(a)
    data["rewards"].append(r)
    data["terminals"].append(done)
    data["infos/goal"].append(tgt)
    data["infos/qpos"].append(env_data.qpos.ravel().copy())
    data["infos/qvel"].append(env_data.qvel.ravel().copy())
    data["infos/s_qvel"].append(s_qvel)


def npify(data):
    for k in data:
        if k == "terminals":
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)


def main():
    # --num_traj 10 --traj_len 50 --env_name maze2d-umaze-v1 with no_randomness=True should give 97% reward.
    # The dataset for final the experiments is generated as defaults

    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Render trajectories")
    parser.add_argument(
        "--env_name",
        type=str,
        default="maze2d-umaze-v1",
        help="Maze type",
        choices=[
            "maze2d-open-v0",
            "maze2d-umaze-v1",
            "maze2d-medium-v1",
            "maze2d-large-v1",
            "maze2d-open-dense-v0",
            "maze2d-umaze-dense-v1",
            "maze2d-medium-dense-v1",
            "maze2d-large-dense-v1",
        ],
    )
    parser.add_argument("--num_traj", type=int, default=int(10), help="Num trajectories to collect")
    parser.add_argument("--traj_len", type=int, default=10, help="Num timesteps per traj (fixed).")
    parser.add_argument(
        "--no_randomness",
        action="store_true",
        help="Use fixed start and goal positions in eval rollouts (maze2d only).",
    )
    parser.add_argument(
        "--bimodal_rew",
        action="store_true",
        help="Use fixed start and goal positions in eval rollouts (maze2d only).",
    )
    args = parser.parse_args()

    if args.bimodal_rew:
        assert args.no_randomness, "To have bimodal reward you need all other randomness to be controlled for"

    # Seeding
    random.seed(0)
    np.random.seed(0)

    horizon = args.traj_len
    n = args.num_traj

    env, maze = make_maze_and_str(args.env_name, horizon, no_rnd=args.no_randomness)
    env.seed(0)

    data = reset_data()
    for traj_idx in tqdm(range(n)):
        s = env.reset()
        controller = CustomWaypointController(maze_str=maze, solve_thresh=0.1)

        # Random action
        # act = env.action_space.sample()

        done = False
        while not done:
            position = s[0:2]
            velocity = env.sim.data.qvel
            act, _ = controller.get_action(position, velocity, env._target)
            # NOTE: Adding noise to actions if randomness enabled. Note that this means that the actions being used
            #  by the system to train have some noise in them. This can be interpreted as having a "suboptimal agent".
            #  In terms of how bad the suboptimality is:
            #  np.concatenate([(np.clip( np.abs(np.random.randn(2))*0.5 , -1, 1) **2) for _ in range(100000)]).mean()
            #  0.2299229309828065 -> any validation loss lower than this is impossible
            if not args.no_randomness:
                act = act + np.random.randn(*act.shape) * 0.5

            act = np.clip(act, -1.0, 1.0)

            if args.bimodal_rew and traj_idx % 2 == 0:
                # Artificially inject noise in the agent performance by slowing down every second trajectory.
                # Dataset collected should be perfectly bimodal
                act /= 5

            ns, rew, done, _ = env.step(act)

            init_qvel = env.reset_info["start_vel"]
            append_data(data, s, act, rew, env._target, done, env.sim.data, init_qvel)
            s = ns

            if args.render:
                env.render()

    assert len(data["actions"]) == horizon * n
    output_name = f"{args.env_name}-h{horizon}"
    paths = pkl_dataset(data, next_obs=False, output_name=output_name)

    returns = np.array([np.sum(p["rewards"]) for p in paths])
    num_samples = np.sum([len(p["rewards"]) for p in paths])
    print(f"Number of samples collected: {num_samples}")
    print(
        f"Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}"
    )
    plt.hist(returns)
    plt.show()


def make_maze_and_str(maze_type, horizon, no_rnd):
    env = gym.make(maze_type)
    maze = env.str_maze_spec
    env = GymMazeEnv(horizon=horizon, maze_spec=maze, reset_target=True, no_randomness=no_rnd)
    return env, maze


def make_maze(maze_type, horizon, no_rnd):
    return make_maze_and_str(maze_type, horizon, no_rnd)[0]


if __name__ == "__main__":
    main()
