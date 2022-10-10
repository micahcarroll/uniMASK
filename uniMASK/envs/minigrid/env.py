

import itertools
import math
from enum import IntEnum

import gym
import numpy as np
import torch
from gym import register
from gym_minigrid.envs import DoorKeyEnv, DoorKeyEnv6x6, EmptyEnv
from gym_minigrid.minigrid import TILE_PIXELS, Door, Goal, Grid, Key, MiniGridEnv, Wall
from gym_minigrid.rendering import downsample, fill_coords, highlight_img, point_in_rect, point_in_triangle, rotate_fn
from gym_minigrid.wrappers import FullyObsWrapper
from torch import tensor as tt
from torch.nn import functional as F
from tqdm import tqdm

from uniMASK.envs.base_data import Dataset
from uniMASK.envs.minigrid.data import r_to_r_idx, r_to_rtg, rtg_to_rtg_idx
from uniMASK.utils import get_class_attributes, manhattan_dist


class EmptyRandomEnv(EmptyEnv):
    """
    An open room environment with:
    - A size which can be set
    - A fixed goal location in the bottom right corner
    - A random starting position for the agent (which I believe is never the goal itself)
    """

    # Hardcoded params for discrete single agent minigrid
    GRID_WIDTH = GRID_HEIGHT = 4
    GOAL_LOC = [GRID_WIDTH, GRID_HEIGHT]
    NUM_STATES = GRID_WIDTH * GRID_HEIGHT
    NUM_ACTIONS = 4  # left, right, south, north
    NUM_REWARDS = 3  # -1, 0 or 1
    NUM_RTGS = 20

    REW_OFFSET = 1
    RTG_OFFSET = 5  # e.g. 2 left + 3 up. Can't have a rtg that is lower than -5

    VALID_POSITIONS = list(itertools.product(np.arange(1, 1 + GRID_WIDTH), np.arange(1, 1 + GRID_HEIGHT)))
    POS_TO_IDX = {(x, y): i for i, (x, y) in enumerate(VALID_POSITIONS)}
    IDX_TO_POS = {v: k for k, v in POS_TO_IDX.items()}

    def __init__(self, size=GRID_WIDTH + 2):
        super().__init__(size=size, agent_start_pos=None)

    def _gen_grid(self, width, height, rnd_goal=False):
        # Create an empty grid
        self.grid = CustomColorGrid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        if rnd_goal:
            self.place_obj(Goal())
        else:
            self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

    @staticmethod
    def sample_dataset(env, n, horizon, agent, show=False):
        """
        Get trajs from the base abstraction of the environment and return them in a Dataset format,
        which can later be converted to the uniMASK format
        """
        obs_n, acts_n, rews_n = [], [], []

        for _ in tqdm(range(n)):
            agent.reset()
            env.reset()
            obs = env.state_idx
            obs_t, acts_t, rews_t = [obs], [], []

            for t in range(horizon):
                action = agent.step(obs)
                _, rew, done, info = env.step(action)
                obs = env.state_idx

                if show:
                    from IPython.core.display import display
                    from PIL import Image

                    display(Image.fromarray(env.render(mode="rgb_array", highlight=False), "RGB"))

                acts_t.append(action)
                rews_t.append(rew)

                if t != horizon - 1:
                    obs_t.append(obs)

            # Everything is discretized, so can reshape here
            obs_n.append(np.array(obs_t).reshape(-1))
            acts_n.append(np.array(acts_t).reshape(-1))
            rews_n.append(np.array(rews_t).reshape(-1))

        rews_n = tt(np.array(rews_n))
        rtg_n = rtg_to_rtg_idx(env, r_to_rtg(rews_n))
        rtg_n = F.one_hot(rtg_n, num_classes=env.NUM_RTGS)
        rews_n = F.one_hot(r_to_r_idx(env, rews_n), num_classes=env.NUM_REWARDS)

        obs_n = F.one_hot(tt(np.array(obs_n)), num_classes=env.NUM_STATES)
        acts_n = F.one_hot(tt(np.array(acts_n)), num_classes=env.NUM_ACTIONS)

        return Dataset({"state": obs_n, "action": acts_n, "reward": rews_n, "rtg": rtg_n})

    @property
    def state_idx(self):
        """The current observation"""
        return self.POS_TO_IDX[tuple(self.agent_pos)]

    @classmethod
    def get_next_state(cls, s_idx, a_idx):
        """
        NOTE: Extremely inefficient. Reimplement better if want to use extensively
        """
        env = make_env()
        for next_s_idx in range(cls.NUM_STATES):
            assert False, "wont work with stochastic dynamics"
            if env.is_transition_valid(s_idx, a_idx, next_s_idx):
                return next_s_idx
        raise ValueError("No state appears to be the next state?")


# Register the environment with gym so that it can be created easily
register(
    id="MiniGrid-Empty-v0",
    entry_point="uniMASK.envs.minigrid.env:EmptyRandomEnv",
)


class CustomDoorKeyEnv6x6(DoorKeyEnv6x6):
    # Hardcoded params for discrete single agent minigrid
    GRID_WIDTH = GRID_HEIGHT = 4
    GOAL_LOC = [GRID_WIDTH, GRID_HEIGHT]
    NUM_STATES = GRID_WIDTH * GRID_HEIGHT
    NUM_ACTIONS = 4  # left, right, south, north
    NUM_REWARDS = 3  # -1, 0 or 1
    NUM_RTGS = 17
    NUM_KEY_STATES = 2  # 1 (picked up) or 0

    REW_OFFSET = 1
    RTG_OFFSET = 6

    VALID_POSITIONS = list(itertools.product(np.arange(1, 1 + GRID_WIDTH), np.arange(1, 1 + GRID_HEIGHT)))
    POS_TO_IDX = {(x, y): i for i, (x, y) in enumerate(VALID_POSITIONS)}
    IDX_TO_POS = {v: k for k, v in POS_TO_IDX.items()}

    def __init__(self):
        super(CustomDoorKeyEnv6x6, self).__init__()

    @property
    def DISTS_TO_GOAL(self):
        grid_hw = self.grid.height
        grid = np.array(self.grid.grid).reshape((grid_hw, grid_hw))
        splitIdx = self.splitIdx
        dist_d = {}
        for x in reversed(range(0, grid_hw)):
            for y in reversed(range(0, grid_hw)):
                item = grid[y][x]
                curr_loc = np.array([x, y])

                if isinstance(item, Wall):
                    continue
                if x > splitIdx:
                    goal = np.array(self.GOAL_LOC)
                    dist_d[tuple(curr_loc)] = np.abs(goal - curr_loc).sum()
                elif x == splitIdx:
                    assert isinstance(item, Door), "entire column should be wall or door"
                    assert tuple(curr_loc) == (splitIdx, 1)
                    dist_d[tuple(curr_loc)] = 1 + dist_d[(splitIdx + 1, 1)]
                elif x < splitIdx:
                    pre_door_loc = np.array([splitIdx - 1, 1])
                    dist_to_door = np.abs(pre_door_loc - curr_loc).sum() + 1
                    dist_d[tuple(curr_loc)] = dist_to_door + dist_d[(splitIdx, 1)]
        return dist_d

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = CustomColorGrid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        self.splitIdx = 3  # self._rand_int(2, width-2)
        self.grid.vert_wall(self.splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(self.splitIdx, height))

        # Place a door in the wall
        doorIdx = 1  # self._rand_int(1, width - 2)
        self.put_obj(Door("red", is_locked=True), self.splitIdx, doorIdx)

        # Place a yellow key on the left side
        self.place_obj(obj=Key("red"), top=(0, 0), size=(self.splitIdx, height))

        # NOTE: Removing randomness in initialization can be useful for debugging
        if False:
            self.agent_pos = (1, 1)
            self.set_key_position((2, 1))

        self.mission = "use the key to open the door and then get to the goal"

    def get_door(self):
        doors = [item for item in self.grid.grid if isinstance(item, Door)]
        assert len(doors) == 1
        return doors[0]

    def get_door_status(self):
        return self.get_door().is_open

    def get_door_pos(self):
        doors = [item for item in self.grid.grid if isinstance(item, Door)]
        assert len(doors) == 1
        door = doors[0]
        return np.array(door.cur_pos)

    def get_key(self):
        keys = [item for item in self.grid.grid if isinstance(item, Key)]
        assert len(keys) in [0, 1]
        if len(keys) == 1:
            key = keys[0]
            assert tuple(key.cur_pos) == tuple(key.init_pos)
            return key
        return None

    def get_key_position(self):
        key = self.get_key()
        return key.cur_pos if key else None

    def _remove_key(self):
        key = self.get_key()
        if key is not None:
            x, y = key.cur_pos
            self.grid.set(x, y, None)

    def set_key_position(self, new_pos):
        """Removes key from current position (if any) and creates a new key at the desired position.

        Args:
            new_idx (int)
        """
        self._remove_key()

        key = Key("red")
        key.cur_pos = np.array(new_pos)
        key.init_pos = new_pos
        self.grid.set(new_pos[0], new_pos[1], key)

    def set_key_state(self, picked_up):
        """Sets env.carrying if key is picked up."""
        if picked_up:
            self.carrying = self.get_key()
            self._remove_key()
        else:
            self.carrying = None

    def get_grid_matrix(self):
        full_width = self.GRID_WIDTH + 2
        grid = self.grid.grid[full_width:-full_width]
        grid = np.array([grid[i::full_width] for i in range(1, full_width - 1)])
        return grid

    def get_internal_wall_coords(self):
        grid = self.get_grid_matrix()
        grid_coords = itertools.product(np.arange(self.GRID_WIDTH), repeat=2)
        return [(x + 1, y + 1) for x, y, in grid_coords if isinstance(grid[x][y], Wall)]

    @staticmethod
    def sample_dataset(env, n, horizon, agent, show=False, varying_agent=False):
        """
        Get trajs from the base abstraction of the environment and return them in a Dataset format,
        which can later be converted to the uniMASK format
        """
        agent_pos_n, door_n, key_pos_n, key_status_n, walls_n, acts_n, rews_n = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for _ in tqdm(range(n)):

            if varying_agent:
                from uniMASK.envs.minigrid.agents import StochGoalAgent

                agent = StochGoalAgent(env, np.random.choice([0.5, 1, 3], p=[0.3, 0.6, 0.1]))

            agent.reset()
            env.reset()

            agent_pos_idx = env.state_idx
            wall_coords = env.get_internal_wall_coords()
            key_pos = env.POS_TO_IDX[tuple(env.get_key_position())]
            key_status = env.get_key_position() is None
            door_state = env.get_door_status()

            agent_pos_t, door_t, key_status_t, key_pos_t, acts_t, rews_t = (
                [agent_pos_idx],
                [door_state],
                [key_status],
                [key_pos],
                [],
                [],
            )

            for t in range(horizon):
                action = agent.step(agent_pos_idx, env)
                _, rew, done, info = env.step(action)

                agent_pos_idx = env.state_idx
                key_status = env.get_key_position() is None
                door_state = env.get_door_status()

                if show:
                    from IPython.core.display import display
                    from PIL import Image

                    display(Image.fromarray(env.render(mode="rgb_array", highlight=False), "RGB"))

                acts_t.append(action)
                rews_t.append(rew)

                if t != horizon - 1:
                    agent_pos_t.append(agent_pos_idx)
                    door_t.append(door_state)
                    key_status_t.append(key_status)
                    key_pos_t.append(agent_pos_idx if key_status else key_pos)

            # Everything is discretized, so can reshape here
            agent_pos_n.append(np.array(agent_pos_t).reshape(-1))
            door_n.append(np.array(door_t).reshape(-1))
            key_status_n.append(np.array(key_status_t).reshape(-1))
            key_pos_n.append(np.array(key_pos_t).reshape(-1))
            walls_n.append(np.array(wall_coords).reshape(-1))

            acts_n.append(np.array(acts_t).reshape(-1))
            rews_n.append(np.array(rews_t).reshape(-1))

        # Dealing with rewards
        rews_n = tt(np.array(rews_n))
        rtg_n = rtg_to_rtg_idx(env, r_to_rtg(rews_n))
        print("rews: ", rews_n)
        print("rtgs:", r_to_rtg(rews_n))
        rtg_n = F.one_hot(rtg_n, num_classes=env.NUM_RTGS)
        rews_n = F.one_hot(r_to_r_idx(env, rews_n), num_classes=env.NUM_REWARDS)

        # Dealing with state
        agent_pos_n = F.one_hot(tt(np.array(agent_pos_n)), num_classes=env.NUM_STATES)
        door_n = F.one_hot(tt(np.array(door_n)).to(int), num_classes=2)
        key_status_n = F.one_hot(tt(np.array(key_status_n)).to(int), num_classes=2)

        # Dealing with non-timestep-specific info
        # walls_n = F.one_hot(tt(np.array(walls_n)), num_classes=env.NUM_STATES)
        key_pos_n = F.one_hot(tt(np.array(key_pos_n)), num_classes=env.NUM_STATES)

        # Dealing with actions
        acts_n = F.one_hot(tt(np.array(acts_n)), num_classes=env.NUM_ACTIONS)

        # Timesteps
        timesteps_n = torch.arange(horizon).expand(n, -1).unsqueeze(-1)

        return Dataset(
            {
                "state": agent_pos_n,
                "action": acts_n,
                "reward": rews_n,
                "rtg": rtg_n,
                "state_door": door_n,
                "state_key": key_status_n,
                "state_key_pos": key_pos_n,
                "timestep": timesteps_n,
            }
        )

    @property
    def state_idx(self):
        """The current observation"""
        return self.POS_TO_IDX[tuple(self.agent_pos)]


# Register the environment with gym so that it can be created easily
register(
    id="MiniGrid-CustomKey-v0",
    entry_point="uniMASK.envs.minigrid.env:CustomDoorKeyEnv6x6",
)


class DoorKeyEnv16x16(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=16)


class CustomDoorKeyEnv16x16(DoorKeyEnv16x16):
    # TODO: eventually do multi-inheritance with other class above (a KeyEnv class and a DoorEnvNxN class)

    # Hardcoded params for discrete single agent minigrid
    GRID_WIDTH = GRID_HEIGHT = 14
    GOAL_LOC = [GRID_WIDTH, GRID_HEIGHT]
    NUM_STATES = GRID_WIDTH * GRID_HEIGHT
    NUM_ACTIONS = 4  # left, right, south, north
    NUM_REWARDS = 3  # -1, 0 or 1
    NUM_RTGS = 21  # Depends on sequence length (assuming max of seq len 10, you can only go -10 to 10)
    NUM_KEY_STATES = 2  # 1 (picked up) or 0

    REW_OFFSET = 1
    RTG_OFFSET = 10

    VALID_POSITIONS = list(itertools.product(np.arange(1, 1 + GRID_WIDTH), np.arange(1, 1 + GRID_HEIGHT)))
    POS_TO_IDX = {(x, y): i for i, (x, y) in enumerate(VALID_POSITIONS)}
    IDX_TO_POS = {v: k for k, v in POS_TO_IDX.items()}

    def __init__(self):
        super(CustomDoorKeyEnv16x16, self).__init__()

    @property
    def DISTS_TO_GOAL(self):
        grid_hw = self.grid.height
        grid = np.array(self.grid.grid).reshape((grid_hw, grid_hw))
        splitIdx = self.splitIdx
        dist_d = {}
        for x in reversed(range(0, grid_hw)):
            for y in reversed(range(0, grid_hw)):
                item = grid[y][x]
                curr_loc = np.array([x, y])

                if isinstance(item, Wall):
                    continue
                if x > splitIdx:
                    goal = np.array(self.GOAL_LOC)
                    dist_d[tuple(curr_loc)] = np.abs(goal - curr_loc).sum()
                elif x == splitIdx:
                    assert isinstance(item, Door), "entire column should be wall or door"
                    assert tuple(curr_loc) == (splitIdx, 1)
                    dist_d[tuple(curr_loc)] = 1 + dist_d[(splitIdx + 1, 1)]
                elif x < splitIdx:
                    pre_door_loc = np.array([splitIdx - 1, 1])
                    dist_to_door = np.abs(pre_door_loc - curr_loc).sum() + 1
                    dist_d[tuple(curr_loc)] = dist_to_door + dist_d[(splitIdx, 1)]
        return dist_d

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = CustomColorGrid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        self.splitIdx = 5  # self._rand_int(2, width-2)
        self.grid.vert_wall(self.splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(self.splitIdx, height))

        # Place a door in the wall
        doorIdx = 1  # self._rand_int(1, width - 2)
        self.put_obj(Door("red", is_locked=True), self.splitIdx, doorIdx)

        # Place a yellow key on the left side
        self.place_obj(obj=Key("red"), top=(0, 0), size=(self.splitIdx, height))

        # NOTE: Removing randomness in initialization can be useful for debugging
        if False:
            self.agent_pos = (1, 1)
            self.set_key_position((2, 1))

        self.mission = "use the key to open the door and then get to the goal"

    def get_door(self):
        doors = [item for item in self.grid.grid if isinstance(item, Door)]
        assert len(doors) == 1
        return doors[0]

    def get_door_status(self):
        return self.get_door().is_open

    def get_door_pos(self):
        doors = [item for item in self.grid.grid if isinstance(item, Door)]
        assert len(doors) == 1
        door = doors[0]
        return np.array(door.cur_pos)

    def get_key(self):
        keys = [item for item in self.grid.grid if isinstance(item, Key)]
        assert len(keys) in [0, 1]
        if len(keys) == 1:
            key = keys[0]
            assert tuple(key.cur_pos) == tuple(key.init_pos)
            return key
        return None

    def get_key_position(self):
        key = self.get_key()
        return key.cur_pos if key else None

    def _remove_key(self):
        key = self.get_key()
        if key is not None:
            x, y = key.cur_pos
            self.grid.set(x, y, None)

    def set_key_position(self, new_pos):
        """Removes key from current position (if any) and creates a new key at the desired position.

        Args:
            new_idx (int)
        """
        self._remove_key()

        key = Key("red")
        key.cur_pos = np.array(new_pos)
        key.init_pos = new_pos
        self.grid.set(new_pos[0], new_pos[1], key)

    def set_key_state(self, picked_up):
        """Sets env.carrying if key is picked up."""
        if picked_up:
            self.carrying = self.get_key()
            self._remove_key()
        else:
            self.carrying = None

    def get_grid_matrix(self):
        full_width = self.GRID_WIDTH + 2
        grid = self.grid.grid[full_width:-full_width]
        grid = np.array([grid[i::full_width] for i in range(1, full_width - 1)])
        return grid

    def get_internal_wall_coords(self):
        grid = self.get_grid_matrix()
        grid_coords = itertools.product(np.arange(self.GRID_WIDTH), repeat=2)
        return [(x + 1, y + 1) for x, y, in grid_coords if isinstance(grid[x][y], Wall)]

    @staticmethod
    def sample_dataset(env, n, horizon, agent, show=False, varying_agent=False):
        """
        Get trajs from the base abstraction of the environment and return them in a Dataset format,
        which can later be converted to the uniMASK format
        """
        agent_pos_n, door_n, key_pos_n, key_status_n, walls_n, acts_n, rews_n = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for _ in tqdm(range(n)):

            if varying_agent:
                from uniMASK.envs.minigrid.agents import StochGoalAgent

                agent = StochGoalAgent(env, np.random.choice([0.5, 1, 3], p=[0.3, 0.6, 0.1]))

            agent.reset()
            env.reset()

            agent_pos_idx = env.state_idx
            wall_coords = env.get_internal_wall_coords()
            key_pos = env.POS_TO_IDX[tuple(env.get_key_position())]
            key_status = env.get_key_position() is None
            door_state = env.get_door_status()

            agent_pos_t, door_t, key_status_t, key_pos_t, acts_t, rews_t = (
                [agent_pos_idx],
                [door_state],
                [key_status],
                [key_pos],
                [],
                [],
            )

            for t in range(horizon):
                action = agent.step(agent_pos_idx, env)
                _, rew, done, info = env.step(action)

                agent_pos_idx = env.state_idx
                key_status = env.get_key_position() is None
                door_state = env.get_door_status()

                if show:
                    from IPython.core.display import display
                    from PIL import Image

                    display(Image.fromarray(env.render(mode="rgb_array", highlight=False), "RGB"))

                acts_t.append(action)
                rews_t.append(rew)

                if t != horizon - 1:
                    agent_pos_t.append(agent_pos_idx)
                    door_t.append(door_state)
                    key_status_t.append(key_status)
                    key_pos_t.append(agent_pos_idx if key_status else key_pos)

            # Everything is discretized, so can reshape here
            agent_pos_n.append(np.array(agent_pos_t).reshape(-1))
            door_n.append(np.array(door_t).reshape(-1))
            key_status_n.append(np.array(key_status_t).reshape(-1))
            key_pos_n.append(np.array(key_pos_t).reshape(-1))
            walls_n.append(np.array(wall_coords).reshape(-1))

            acts_n.append(np.array(acts_t).reshape(-1))
            rews_n.append(np.array(rews_t).reshape(-1))

        # Dealing with rewards
        rews_n = tt(np.array(rews_n))
        rtg_n = rtg_to_rtg_idx(env, r_to_rtg(rews_n))
        print("rews: ", rews_n)
        print("rtgs:", r_to_rtg(rews_n))
        rtg_n = F.one_hot(rtg_n, num_classes=env.NUM_RTGS)
        rews_n = F.one_hot(r_to_r_idx(env, rews_n), num_classes=env.NUM_REWARDS)

        # Dealing with state
        agent_pos_n = F.one_hot(tt(np.array(agent_pos_n)), num_classes=env.NUM_STATES)
        door_n = F.one_hot(tt(np.array(door_n)).to(int), num_classes=2)
        key_status_n = F.one_hot(tt(np.array(key_status_n)).to(int), num_classes=2)

        # Dealing with non-timestep-specific info
        # walls_n = F.one_hot(tt(np.array(walls_n)), num_classes=env.NUM_STATES)
        key_pos_n = F.one_hot(tt(np.array(key_pos_n)), num_classes=env.NUM_STATES)

        # Dealing with actions
        acts_n = F.one_hot(tt(np.array(acts_n)), num_classes=env.NUM_ACTIONS)

        # Timesteps
        timesteps_n = torch.arange(horizon).expand(n, -1).unsqueeze(-1)

        return Dataset(
            {
                "state": agent_pos_n,
                "action": acts_n,
                "reward": rews_n,
                "rtg": rtg_n,
                "state_door": door_n,
                "state_key": key_status_n,
                "state_key_pos": key_pos_n,
                "timestep": timesteps_n,
            }
        )

    @property
    def state_idx(self):
        """The current observation"""
        return self.POS_TO_IDX[tuple(self.agent_pos)]


# Register the environment with gym so that it can be created easily
register(
    id="MiniGrid-CustomKey16x16-v0",
    entry_point="uniMASK.envs.minigrid.env:CustomDoorKeyEnv16x16",
)


class CustomActions(IntEnum):
    """The no-dir version of the environment only has 4 actions, rather than the 6 (?) of the original environment"""

    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

    @classmethod
    def apply_noise(cls, a, error_p=0.3):
        assert a in [_a.value for _a in CustomActions]
        a0, a1 = cls.get_adj_actions(a)
        correct_p = 1 - error_p
        return np.random.choice([a, a0, a1], p=[correct_p, error_p / 2, error_p / 2])

    @classmethod
    def get_adj_actions(cls, a):
        if a == cls.RIGHT:
            return [cls.UP, cls.DOWN]
        elif a == cls.UP:
            return [cls.RIGHT, cls.LEFT]
        elif a == cls.LEFT:
            return [cls.UP, cls.DOWN]
        elif a == cls.DOWN:
            return [cls.RIGHT, cls.LEFT]
        else:
            raise ValueError()


def make_env(env_type="empty"):
    """
    To make the environment, we first make the base empty room env, and then we add the wrapper that makes the
    env directionless
    """
    if env_type == "empty":
        env = gym.make("MiniGrid-Empty-v0")
    elif env_type == "key":
        env = gym.make("MiniGrid-CustomKey-v0")
    elif env_type == "key16x16":
        env = gym.make("MiniGrid-CustomKey16x16-v0")
    else:
        raise ValueError()
    env = NoDirEnvWrapper(FullyObsWrapper(env))
    return env


class NoDirEnvWrapper(gym.core.ObservationWrapper):
    """
    A wrapper that returns a MiniGrid env without agent direction. This enables for easier visualizations of
    predicted trajs and actions.

    What this means for agent control is that the agent can be thought of as a dot which just decides which cardinal
    direction to move in at every timestep.
    """

    def __init__(self, env):
        assert isinstance(env, FullyObsWrapper)
        super().__init__(env)
        self.base_env = env.unwrapped

        for (attr, val) in get_class_attributes(self.base_env.__class__):
            # HACK: If env attribute is capitalized, copy it over to the wrapper (trying to copy over ACT/OBS space attributes)
            if attr.upper() == attr:
                self.__dict__[attr] = val

    def step(self, action, action_noise=0.0):
        assert action in [a.value for a in CustomActions]

        if action_noise:
            action = CustomActions.apply_noise(action, error_p=action_noise)

        prev_pos = self.agent_pos

        # To go in a direction which doesn't correspond to the agent's underlying direction in the original env,
        # you need to first rotate a bunch of times until you're facing the right way, and then you can take the
        # "forward" action
        while self.agent_dir != action:
            _, _rew, done, _ = self.env.step(MiniGridEnv.Actions.left)
            assert _rew == 0
            assert done is False

        for a in [MiniGridEnv.Actions.pickup]:
            obs, _, done, info = self.env.step(a)
            assert done is False

        obs, _, done, info = self.env.step(MiniGridEnv.Actions.forward)

        if tuple(self.agent_pos) == tuple(prev_pos):
            # If going forward has no effect, try to open a door
            obs, _, done, info = self.env.step(MiniGridEnv.Actions.toggle)
            obs, _, done, info = self.env.step(MiniGridEnv.Actions.forward)

        new_pos = self.agent_pos

        if isinstance(self.base_env, EmptyRandomEnv):
            # Reward by how much the agent advances, all shifted by 1 so as to have the reward be non-negative for
            # easier one-hot encoding
            prev_d = manhattan_dist(EmptyRandomEnv.GOAL_LOC, prev_pos)
            new_d = manhattan_dist(EmptyRandomEnv.GOAL_LOC, new_pos)
            rew = int(prev_d - new_d)
        elif isinstance(self.base_env, CustomDoorKeyEnv6x6) or isinstance(self.base_env, CustomDoorKeyEnv16x16):
            prev_d = self.base_env.DISTS_TO_GOAL[tuple(prev_pos)]
            new_d = self.base_env.DISTS_TO_GOAL[tuple(new_pos)]
            rew = int(prev_d - new_d)
            # rew = int(tuple(new_pos) == (4, 4))
        else:
            raise ValueError()

        return obs, rew, done, info

    def set_agent_pos(self, pos):
        env = self.env
        while not isinstance(env, MiniGridEnv):
            # Go up all wrapper layers
            env = env.env
        env.agent_pos = pos

    def observation(self, observation):
        # Defer to actual wrapper
        return self.env.observation(observation)

    @property
    def state_idx(self):
        """The current observation"""
        return self.base_env.state_idx

    def is_transition_valid(self, curr_s, curr_a, next_s, is_holding_key=None):
        """
        NOTE: Requires deterministic dynamics.
        TODO: have attribute for determinism so that can make assertions

        NOTE: Modifies the current environment instance to simulate a specific transition
        """
        self.reset()
        curr_pos, next_pos = self.IDX_TO_POS[curr_s], self.IDX_TO_POS[next_s]
        self.set_agent_pos(curr_pos)
        if is_holding_key is not None:
            self.env.set_key_state(is_holding_key)
        self.step(curr_a)
        new_pos = tuple(self.agent_pos)
        return new_pos == next_pos


class CustomColorGrid(Grid):
    """Same as Grid excpet for one line which sets the background color of the rendering to white rather than black"""

    def render(self, tile_size, agent_pos=None, agent_dir=None, highlight_mask=None):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=np.bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                agent_here = np.array_equal(agent_pos, (i, j))
                tile_img = self.render_tile(
                    cell,
                    agent_dir=agent_dir if agent_here else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    @classmethod
    def render_tile(cls, obj, agent_dir=None, highlight=False, tile_size=TILE_PIXELS, subdivs=3):
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key = (agent_dir, highlight, tile_size)
        key = obj.encode() + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.ones(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8) * 255

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj != None:
            obj.render(img)

        # Overlay the agent on top
        if agent_dir is not None:
            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir)
            fill_coords(img, tri_fn, (255, 0, 0))

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img
        return img
