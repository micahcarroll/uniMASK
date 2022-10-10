
from collections import defaultdict

import glfw
import numpy as np
import torch
from torch import tensor as tt

from uniMASK.batches import Batch, DTActionPred, SpanPred
from uniMASK.sequences import FullTokenSeq
from uniMASK.trainer import BCEvalIndicator, RCAutoEvalIndicator
from uniMASK.utils import to_numpy


class Evaluator:

    _env_idx_to_render = 0

    def __init__(
        self,
        make_env,
        rew_eval_num,
        eval_types,
        max_ep_len,
        rew_scale=None,
        render=False,
        env_seeds=None,
        target_factor=None,
    ):
        assert rew_eval_num > 0
        assert len(eval_types) > 0
        self.max_ep_len = max_ep_len
        self.rew_scale = rew_scale
        self.eval_types = eval_types
        self.render = render
        self.rew_eval_num = rew_eval_num
        self.env_seeds = np.random.randint(0, 10e6, size=rew_eval_num) if env_seeds is None else env_seeds
        self.env_seeds = [int(seed) for seed in self.env_seeds]
        self.target_factor = target_factor

        # Creating envs takes a while, so create them lazily
        self._envs = None
        self.make_env = make_env

        # to be initialized in set_trainer()
        self.trainer = None
        self.seq_len = None
        self.trained_with_timestep = None
        self.trained_with_rtg = None
        self.names_and_shapes = None

    @property
    def envs(self):
        if self._envs is not None:
            return self._envs

        self._envs = [self.make_env() for _ in range(self.rew_eval_num)]
        return self._envs

    def set_trainer(self, trainer):
        # This exists because right now trainer needs pointer to evaluator, and vice versa. But it should not be this way.
        # I think eventually there will be a "runner" class that runs trainer and evaluator.
        self.trainer = trainer
        self.seq_len = trainer.seq_len
        self.trained_with_timestep = "timestep" in trainer.loss_weights
        # this is independent of whether this is a BC eval; it is determined by the architecture.
        self.trained_with_rtg = "rtg" in trainer.loss_weights
        # set up names and shapes: All possible factors (superset of curr_data.factor_names, later)
        self.names_and_shapes = {
            "state": (1, self.seq_len) + self.envs[0].observation_space.shape,
            "action": (1, self.seq_len) + self.envs[0].action_space.shape,
            "rtg": (1, self.seq_len, 1),
        }
        if self.trained_with_timestep:
            self.names_and_shapes["timestep"] = (1, self.seq_len, 1)
        if self.trainer.rtg_cat_t:
            assert not self.trained_with_timestep
            self.names_and_shapes["rtg"] = (1, self.seq_len, 2)

    def evaluate(self, dataset=None):
        assert self.trainer is not None
        # eventually, {rew_batch_params: {tr: {'ep_rews': np.array, 'ep_lens': List}}}
        eval_metrics = defaultdict(dict)
        for batch_params in self.trainer.rew_batch_params_n:
            # Evaluation metrics for this set of reward batch params
            eval_m_for_bp = eval_metrics[batch_params["logging_name"]]

            # Ignore specified target rewards if doing BC eval. This is necessary because sometimes you
            # might have both BC and RC rew_batch_params for a RND model, and you want to ignore the extra stuff when
            # doing the BC evals.
            if batch_params["rtg_masking_type"] == "BC":
                curr_trs = [BCEvalIndicator]
            else:
                # assert (
                #     BCEvalIndicator not in self.eval_types
                # ), "rtg_masking_type should be BC if you have this in your eval_types. Maybe you forgot to specify the RC rew eval type (-et flag)?"
                curr_trs = self.eval_types

            for target_rew in curr_trs:
                self._evaluate_single_tr(target_rew, batch_params, dataset)
                eval_m_for_bp[target_rew] = {
                    "ep_rews": np.array([sum(rews) for rews in self.rews_n]),
                    "ep_lens": np.array(self.ep_lens_n),
                }
        return eval_metrics

    def _evaluate_single_tr(self, tr, batch_params, dataset):
        # Reset all the envs
        self._reset()

        self.rtg_n = []
        if tr is BCEvalIndicator:
            self.rtg_n.extend([tr for _ in range(self.rew_eval_num)])
        elif tr is RCAutoEvalIndicator:
            self._auto_rew_target_setting(dataset)
        else:
            self.rtg_n.extend([tr / self.rew_scale for _ in range(self.rew_eval_num)])

        for timestep in range(self.max_ep_len):
            self.seq_idxes_across_envs = []  # Only used for sanity check below

            self.seq_idx = None
            self._prepare_data(timestep)
            assert len(set(self.seq_idxes_across_envs)) == 1, (
                "At every iteration of the evaluation loop, the point of the sequence which "
                "we should care about should be the same in each env"
            )
            self.seq_idx = self.seq_idxes_across_envs[0]
            self.actions_n = self._get_actions(batch_params)
            self._take_actions(timestep)
            if all(self.done_n):
                break

        if self.render:
            rendered_env = self.envs[self._env_idx_to_render]
            glfw.destroy_window(rendered_env._get_viewer(mode="human").window)

        assert all(self.done_n)
        [env.close() for env in self.envs]

    def _auto_rew_target_setting(self, dataset):
        from uniMASK.data.datasets.generate_maze2d import GymMazeEnv

        assert isinstance(self.envs[0], GymMazeEnv), "Only supported in Maze rn"
        # For each env, get the info on the current reset.
        envs_reset_info = [env.reset_info for env in self.envs]
        for env_reset_info in envs_reset_info:
            env_initial_condition = np.concatenate(
                [
                    env_reset_info["start_pos"] * 3,
                    env_reset_info["goal_pos"] * 3,
                    env_reset_info["start_vel"],
                ]
            )
            data_initial_conditions = torch.cat(
                [
                    torch.stack(dataset.state)[:, 0] * 3,
                    torch.stack(dataset.s_qvel)[:, 0],
                ],
                dim=-1,
            )
            deltas = torch.norm(data_initial_conditions - tt(env_initial_condition), dim=-1)
            # Finding the closest initial position (to the one that was just randomly generated)
            # in the training data
            closest_init_idx = torch.argmin(deltas)
            init_rtg = torch.stack(dataset.rtg)[closest_init_idx][0, 0]

            # Rescale the target RTG by a constant (which still leads the reward to be reasonable)
            target_mult = 1.2 if self.target_factor is None else self.target_factor
            init_rtg = init_rtg * target_mult

            # Each env will have a reasonable rtg
            self.rtg_n.append(init_rtg)

    def _reset(self):
        self.rews_n = [[] for _ in range(self.rew_eval_num)]
        self.done_n = [False] * self.rew_eval_num
        self.ep_lens_n = [None] * self.rew_eval_num
        self.raw_obs_n = [None] * self.rew_eval_num
        self.curr_data_n = [None] * self.rew_eval_num

        # initialize envs
        for idx, (env, seed) in enumerate(zip(self.envs, self.env_seeds)):
            # Create an empty token sequence based on the loss weights dictionary
            curr_data = FullTokenSeq.empty_token_seq(
                self.names_and_shapes,
                stacked=self.trainer.stacked,
                input_keys=set(self.trainer.loss_weights),
                loss_types=self.trainer.loss_types,
            )

            if self.trained_with_timestep:
                for i in range(self.seq_len):
                    curr_data = curr_data.with_added_missing_input("timestep", seq_idx=i, x_n=tt(i).unsqueeze(0))
            self.curr_data_n[idx] = curr_data
            env.seed(seed)
            self.raw_obs_n[idx] = env.reset()

    def _prepare_data(self, timestep):
        """
        Based on what trajectory timestep we are at, shift and populate the current input tensor or just populate it further.
        Updates the index at which we want to sample the action in the sequence passed into the transformer, plus
        the data formatted with the right information in it.
        """
        for idx, (curr_data, observation, rtg) in enumerate(zip(self.curr_data_n, self.raw_obs_n, self.rtg_n)):
            if self.done_n[idx]:
                continue
            if timestep < self.seq_len:
                # If we're in the first couple of timesteps, the index of the sequence that we're interested in
                # corresponds directly to the timestep
                seq_idx = timestep
            else:
                # If we're past the window, the index of the sequence that we're interested in will be the last one
                # and the previous entries will be full of context
                curr_data = curr_data.shift_by_one()
                seq_idx = self.seq_len - 1

                if self.trained_with_timestep:
                    curr_data = curr_data.with_added_missing_input(
                        "timestep", seq_idx=seq_idx, x_n=tt(timestep).unsqueeze(0)
                    )

            curr_data = curr_data.with_added_missing_input("state", seq_idx=seq_idx, x_n=tt(observation).unsqueeze(0))
            if self.trained_with_rtg:
                # The rtg should is set to something that errors out if not masked appropriately
                rtg = np.nan if rtg == BCEvalIndicator else rtg
                # If rtg present in the training input, add it to the eval input too. The masking will be dealt with later
                if not self.trainer.rtg_cat_t:
                    curr_data = curr_data.with_added_missing_input("rtg", seq_idx=seq_idx, x_n=tt(rtg).unsqueeze(0))
                else:
                    assert not self.trained_with_timestep
                    # If BC eval, timestep slot should also be nan, otherwise add with normalized timestep
                    t = np.nan if rtg is np.nan else timestep / self.max_ep_len
                    # We want to concatenate the timestep info with the rtg. We normalize the timestep with the max ep len.
                    curr_data = curr_data.with_added_missing_input(
                        "rtg",
                        seq_idx=seq_idx,
                        x_n=tt([rtg, t]),
                    )
            self.seq_idxes_across_envs.append(seq_idx)
            self.curr_data_n[idx] = curr_data

    def _get_actions(self, batch_params):
        # Current set of input datas for all envs
        curr_data = FullTokenSeq.concatenate(self.curr_data_n)
        bp = dict(batch_params)
        batch_type = bp["type"]
        assert (
            issubclass(batch_type, SpanPred) or batch_type is DTActionPred
        ), "span_limits only works with SpanPred subclasses"
        bp["span_limits"] = (self.seq_idx + 1, self.seq_len)
        b = Batch.get_dummy_batch_output(curr_data, bp, self.trainer)
        return to_numpy(b.get_factor("action").output[:, self.seq_idx])

    def _take_actions(self, timestep):
        for idx, (action, curr_data, env, rtg, done) in enumerate(
            zip(self.actions_n, self.curr_data_n, self.envs, self.rtg_n, self.done_n)
        ):
            if done:
                continue
            new_obs, reward, done, info = env.step(action)
            if self.render and idx == self._env_idx_to_render:
                env.render()
            if rtg != BCEvalIndicator:
                # TODO: move rew scaling to the transfomer class
                rtg -= reward / self.rew_scale
            self.rews_n[idx].append(reward)
            self.curr_data_n[idx] = curr_data.with_added_missing_input(
                "action", seq_idx=self.seq_idx, x_n=tt(action).unsqueeze(0)
            )
            self.raw_obs_n[idx] = new_obs
            self.rtg_n[idx] = rtg
            self.done_n[idx] = done
            if done and self.ep_lens_n[idx] is None:
                self.ep_lens_n[idx] = timestep + 1

    # TODO do we want saving and loading functionality? Trainer might want this from us.
    def save(self):
        pass

    def load(self):
        pass
