

import numpy as np
from gym_minigrid.minigrid import DIR_TO_VEC
from scipy import special

from uniMASK.envs.minigrid.env import CustomActions, CustomDoorKeyEnv6x6, CustomDoorKeyEnv16x16, EmptyRandomEnv
from uniMASK.utils import manhattan_dist


class GoalAgent:
    """
    An agent that goes towards the goal along the shortest path, without making any mistakes, but breaking ties
    randomly.
    """

    def __init__(self, env):
        self.env_cls = env.base_env.__class__

    def get_actions_and_costs(self, obs, goal, obstacles=()):
        """
        Get costs for all possible actions (i.e. -1 if the action leads to get closer to the goal, 0 if leaves
        distance the same, and 1 if the action increases the distance by 1), and information about which actions have
        the lowest cost.
        """
        position = np.array(self.env_cls.IDX_TO_POS[obs])

        curr_dist = manhattan_dist(goal, position)
        actions_and_costs = []
        max_dist_gain = -2
        best_actions = []
        for d in CustomActions:
            direction = DIR_TO_VEC[d.value]

            assert self.env_cls.GRID_WIDTH == self.env_cls.GRID_HEIGHT
            # NOTE: at some point we might want to simulate movement to determine this.
            new_pos = np.minimum(np.maximum(position + direction, 0), self.env_cls.GRID_WIDTH)
            if tuple(new_pos) in obstacles:
                new_pos = position

            new_dist = manhattan_dist(goal, new_pos)
            dist_gain = curr_dist - new_dist
            actions_and_costs.append((d, dist_gain))

            if dist_gain > max_dist_gain:
                max_dist_gain = dist_gain
                best_actions = [d]
            elif dist_gain == max_dist_gain:
                best_actions.append(d)

        return actions_and_costs, best_actions

    def get_curr_goal(self, env=None):
        if self.env_cls is EmptyRandomEnv:
            goal = self.env_cls.GOAL_LOC
            obstacles = ()

        elif self.env_cls in [CustomDoorKeyEnv6x6, CustomDoorKeyEnv16x16]:
            obstacles = env.get_internal_wall_coords()
            door_pos = env.get_door_pos()
            door_open = env.get_door_status()
            key_pos = env.get_key_position()
            key_picked_up = key_pos is None

            if door_open:
                assert key_picked_up
                goal = self.env_cls.GOAL_LOC
            elif not door_open and key_picked_up:
                goal = door_pos
            elif not key_picked_up:
                assert not door_open
                goal = key_pos
            else:
                raise ValueError()
        else:
            raise ValueError()
        return goal, obstacles

    def step(self, obs, env=None):
        goal, obstacles = self.get_curr_goal(env)
        _, best_actions = self.get_actions_and_costs(obs, goal, obstacles=obstacles)
        return np.random.choice(best_actions)

    def reset(self):
        pass


class StochGoalAgent(GoalAgent):
    """
    An agent that goes towards the goal but has a probability of making mistakes, controlled by beta
    """

    def __init__(self, env, beta):
        super(StochGoalAgent, self).__init__(env)
        self.beta = beta

    def get_action_probs(self, obs, goal, obstacles=()):
        actions_and_costs, _ = self.get_actions_and_costs(obs, goal, obstacles)
        costs = np.array([a_and_c[1] for a_and_c in actions_and_costs])
        sampling_probs = special.softmax(self.beta * costs)
        return sampling_probs

    def step(self, obs, env=None):
        goal, obstacles = self.get_curr_goal(env)
        sampling_probs = self.get_action_probs(obs, goal, obstacles)
        sampled_action = np.random.choice(np.arange(4), p=sampling_probs)
        return sampled_action

    def reset(self):
        pass
