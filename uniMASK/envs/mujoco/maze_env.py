import numpy as np
from d4rl.pointmaze import maze_model


class GymMazeEnv(maze_model.MazeEnv):
    """
    Actually make the env conform to the gym API.

    Expert should get almost 100% on umaze with no_randomness.
    This can be used for testing memorization capabilities of models.
    """

    def __init__(self, horizon, noise_bounds=0.3, no_randomness=False, **kwargs):
        # All these are used in the super __init__ so should set it before super()
        self.no_randomness = no_randomness
        self.noise_bounds = noise_bounds
        self.t = 0
        self.horizon = horizon
        self.initial_distance = 0
        self.prev_distance = 0
        super().__init__(**kwargs)
        self.reset()

    def reset(self):
        """
        If return_idx, returns (obs, idx) where idx is of the start and goal locations. Else, returns obs.
        """
        self.t = 0
        # This is in mujoco code, but it calls our reset_model(). So unpacking will work.
        ob = super().reset()

        self.initial_distance = np.linalg.norm(self._target - ob[:2])
        self.prev_distance = self.initial_distance
        return ob

    def step(self, action):
        ob, _, done, info = super().step(action)
        self.t += 1
        if self.t == self.horizon:
            done = True
        curr_dist = np.linalg.norm(self._target - ob[:2])
        distance_delta = self.prev_distance - curr_dist

        # Percentage progress increase relative to previous timestep
        # reward = distance_delta / self.initial_distance

        # How much did you get closer to the goal?
        reward = distance_delta

        self.prev_distance = curr_dist
        return ob, reward, done, info

    def _get_obs(self):
        # NOTE: velocity has been removed so that the problem is not Markov anymore – this motivates the usage of
        #  sequence models!
        return np.concatenate([self.sim.data.qpos, self._target]).ravel()

    def set_target(self, target_location=None, agent_pos_idx=None):
        if target_location is None:
            if self.no_randomness:
                idx = 1
            else:
                idx = self.np_random.choice(len(self.empty_and_goal_locations))

                # Kind of hacky way to not have agent spawned on target
                while idx == agent_pos_idx:
                    idx = self.np_random.choice(len(self.empty_and_goal_locations))

            reset_location = np.array(self.empty_and_goal_locations[idx]).astype(self.observation_space.dtype)
            # Useless to have target noise if PID is not able to get there because of discretization
            target_location = reset_location
        self._target = target_location

    def reset_model(self):
        if self.no_randomness:
            idx = 2
        else:
            idx = self.np_random.choice(len(self.empty_and_goal_locations))

        reset_location = np.array(self.empty_and_goal_locations[idx]).astype(self.observation_space.dtype)

        qpos = reset_location
        qvel = self.init_qvel
        if not self.no_randomness:
            qpos += self.np_random.uniform(low=-self.noise_bounds, high=self.noise_bounds, size=self.model.nq)
            qvel += self.np_random.randn(self.model.nv) * 0.1

        self.set_state(qpos, qvel)

        if self.reset_target:
            self.set_target(agent_pos_idx=idx)

        self.reset_info = {
            "start_pos": qpos.ravel().copy(),
            "start_vel": qvel.ravel().copy(),
            "goal_pos": self._target,
        }
        return self._get_obs()
