import abc
import argparse
import os
import random
import typing
from logging import warning

import gym

from uniMASK.batches import *
from uniMASK.data import TEST_DATA_DIR
from uniMASK.envs.base_data import Dataset
from uniMASK.envs.evaluator import Evaluator
from uniMASK.scripts.argparsing import parse_common_args
from uniMASK.trainer import BCEvalIndicator, RCAutoEvalIndicator
from uniMASK.utils import get_inheritors, imdict


def parse(_input):
    # Make parser and subparsers
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="environment", dest="environment", help="environment", required=True)
    # To get optionals to appear after positionals, must iterate twice: first add_subparser(), then parse_common_args().
    subparsers_list = []
    for config_class in get_inheritors(Config):
        subparser = config_class.add_subparser(subparsers)
        subparser.set_defaults(**config_class.default_args)
        subparsers_list.append(subparser)
    for subparser in subparsers_list:
        parse_common_args(subparser, Config.possible_batch_codes)

    args = parser.parse_args(args=_input)
    args = vars(args)

    # Now we extract all defaults args: common ones, and subconfig-specific ones.
    # NOTE: Here we must work off of the subparser rather than the parser.
    subparser = subparsers.choices[args["environment"]]
    all_default_args = {key: subparser.get_default(key) for key in args}

    config = CONFIG_NAME_TO_CLASS[args["environment"]](args, all_default_args)
    return config


def batch_code_to_params_n_dict(args):
    if args is None:  # Dummy run (for cl parsing). Put dummy values.
        seq_len = 0
    else:
        seq_len = args["seq_len"]
    bc_to_params = {
        "rnd": {"type": RandomPred, "rtg_masking_type": "BCRC_uniform_first"},
        # Random but with no reward-conditioning
        "rnd_BC": {"type": RandomPred, "rtg_masking_type": "BC"},
        "rnd_RC": {"type": RandomPred, "rtg_masking_type": "RC_fixed_first"},
        "future": {"type": FuturePred, "rtg_masking_type": "BC"},
        "past": {"type": PastPred, "rtg_masking_type": "BC"},
        "BC": {"type": BehaviorCloning, "rtg_masking_type": "BC"},
        # BC last
        "BC_last": {
            "type": BehaviorCloning,
            "rtg_masking_type": "BC",
            "span_limits": (seq_len, seq_len),
        },
        # BC second to last
        "BC_stlast": {
            "type": BehaviorCloning,
            "rtg_masking_type": "BC",
            "span_limits": (seq_len - 1, seq_len),
        },
        "RC": {"type": RCBehaviorCloning, "rtg_masking_type": "RC_fixed_first"},
        # RC last
        "RC_last": {
            "type": RCBehaviorCloning,
            "rtg_masking_type": "RC_fixed_first",
            "span_limits": (seq_len, seq_len),
        },
        # RC second to last
        "RC_stlast": {
            "type": RCBehaviorCloning,
            "rtg_masking_type": "RC_fixed_first",
            "span_limits": (seq_len - 1, seq_len),
        },
        "DT_BC": {"type": DTActionPred, "rtg_masking_type": "BC"},
        "DT_RC": {"type": DTActionPred, "rtg_masking_type": "RC_fixed_first"},
        "forwards": {"type": ForwardDynamics, "rtg_masking_type": "BC"},
        "backwards": {"type": BackwardsDynamics, "rtg_masking_type": "BC"},
        "goal_conditioned": {"type": GoalConditionedBC, "rtg_masking_type": "BC"},
        # GC second to last
        "gc_stlast": {
            "type": GoalConditionedBC,
            "rtg_masking_type": "BC",
            "span_limits": (seq_len - 1, seq_len),
        },
        "waypoint": {
            "type": WaypointConditionedBC,
            "waypoints": (0, seq_len // 3, (seq_len * 2) // 3, seq_len - 1),
            "rtg_masking_type": "BC",
        },
    }
    for k in bc_to_params.keys():
        # logging name will be the same as the batch code, with the only exception being for DT runs, e.g. it will
        # need a different batch code even for the same exact type of training (because it requires a different batch)
        if k[:2] == "DT":
            bc_to_params[k]["logging_name"] = k[3:]
        else:
            bc_to_params[k]["logging_name"] = k
    bc_to_params = {k: imdict(v) for k, v in bc_to_params.items()}

    bc_to_params_n = {k: [v] for k, v in bc_to_params.items()}
    bc_to_params_n["all"] = [bc_to_params[k] for k in ["past", "future", "BC", "RC", "goal_conditioned", "waypoint"]]
    bc_to_params_n["all_w_dyna"] = bc_to_params_n["all"] + bc_to_params_n["forwards"] + bc_to_params_n["backwards"]
    bc_to_params_n["BCs"] = bc_to_params_n["BC"] + bc_to_params_n["BC_last"]
    bc_to_params_n["RCs"] = bc_to_params_n["RC"] + bc_to_params_n["RC_last"]
    bc_to_params_n["DT_all"] = [
        bc_to_params["DT_BC"],
        bc_to_params["DT_RC"],
    ]
    bc_to_params_n["BC_RC"] = [bc_to_params["BC"], bc_to_params["RC"]]
    for k, v in bc_to_params_n.items():
        bc_to_params_n[k] = tuple(v)
    return bc_to_params_n


FB_TO_DT_BATCH_CODE = {"BC": "DT_BC", "RC": "DT_RC", "all": "DT_all"}


class Config(abc.ABC):
    """This class bridges between raw command-line arguments ("args") and data used by the trainer.
    In particular, it parses arguments (typically ints or strings) into uniMASK classes, functions, etc.
    Since arguments differ between envs (either the arguments themselves, or their meaning), Config should be extended
    and implemented for each env to be added.
    """

    loss_type: str  # E.g., "l2" or "sce"
    log_interval: int  # How often logs should be generated (in iterations).
    possible_batch_codes = batch_code_to_params_n_dict(args=None).keys()

    default_args: typing.Dict[str, typing.Any]  # Maps argument name to default value.

    max_ep_len: int

    SPECIAL_EVAL_CODES = {"RC_auto": RCAutoEvalIndicator, "BC": BCEvalIndicator}

    _train_evaluator: typing.Optional[Evaluator]
    _final_evaluator: typing.Optional[Evaluator]

    def __init__(self, args, all_default_args):
        self._args = args
        self._all_default_args = all_default_args
        self.validate_args()
        self.bc_to_params_dict = batch_code_to_params_n_dict(self._args)

        # Sets self.train_data, self.test_data
        self._set_train_test_data()
        # Must happen after setting self.train_data
        if self._args["timesteps"]:
            assert self._args["epochs"] is None
            self._args["epochs"] = max(self._args["timesteps"] // self.train_data.total_timesteps, 1)
            print(f"Running for {self._args['epochs']} epochs.")

        self._train_evaluator = None
        self._final_evaluator = None

    def _set_train_test_data(self):
        # Fix all seeds to 0 so that we get the same dataset every run.
        np.random.seed(0)
        random.seed(0)
        train_batch_code = self._args["train_batch_code"]
        self.train_data, self.test_data = self.get_data(train_batch_code, self.loss_weights)
        # Now reset the seed
        np.random.seed(self._args["seed"])
        random.seed(self._args["seed"])
        torch.manual_seed(self._args["seed"])

    @staticmethod
    def parse_wandb_sweep_params(args):
        if args["wandb_sweep_params"] is None:
            return args
        print("Note: Overwriting some of (model, batch_code, val_batch_code, rew_batch_code, epoch) for sweep.")
        params = args["wandb_sweep_params"].split(",")
        if len(params) != 5:
            raise ValueError(
                "wandb_sweep_params should have four commas. Empty string leaves an arg unspecified."
                "E.g., FB,rnd,all,,, leaves rew_batch_code and epochs unspecified."
            )
        new_args = {}
        (
            new_args["model_class"],
            new_args["train_batch_code"],
            new_args["val_batch_code"],
            new_args["rew_batch_code"],
            new_args["epochs"],
        ) = params
        if new_args["epochs"].isnumeric():
            new_args["epochs"] = int(new_args["epochs"])
        args.update((k, v) for k, v in new_args.items() if v is not "")
        return args

    def validate_args(self):
        """Validates and updates variant parameters."""
        args = self._args
        args = self.parse_wandb_sweep_params(args)

        if args["sweep_timestep_encoding"] is not None:
            print("Note: Overwriting t_enc for sweep.")
            args["timestep_encoding"] = bool(args["sweep_timestep_encoding"])

        if args["wandb_notes"] is not None and args["wandb_project"] is None:
            warning(
                "wandb_notes specified but will not have any effect since there will be no logging to wandb. "
                "To enable logging, specify --wandb_project (-wp) PROJECT_NAME"
            )

        if args["reward_scale"] == 0.0:
            raise ValueError("Reward scaling factor must be nonzero.")

        if args["reward_scale"] < 1.0:
            warning(
                "reward_scale is the factor by which to *shrink* rewards. "
                "Are you sure you want it to be less than 1?"
            )

        if args["model_class"] == "DT":
            if args["state_loss"] != 0 or args["rtg_loss"] != 0:
                warning("Having state or rtg loss !=0 for model_class DT has no effect")
                args["state_loss"] = 0
                args["rtg_loss"] = 0
            for batch_code in ["train_batch_code", "val_batch_code", "rew_batch_code"]:
                # If necessary, attempt to convert batch_code to DT equivalent.
                if args[batch_code] is None or args[batch_code][:2] == "DT":
                    continue
                if args[batch_code] in FB_TO_DT_BATCH_CODE:
                    warning(f"Converting {batch_code}={args[batch_code]} to {FB_TO_DT_BATCH_CODE[args[batch_code]]}.")
                    args[batch_code] = FB_TO_DT_BATCH_CODE[args[batch_code]]
                    continue
                raise ValueError(f"{batch_code}={args[batch_code]} has no DT equivalent, and you are using a DT model.")
            if args["rtg_loss"] != 0 or args["state_loss"] != 0:
                warning("rtg_loss and state_loss must be 0 when using DT. Overwriting")
                args["rtg_loss"], args["state_loss"] = 0, 0
        else:
            for batch_code in ["train_batch_code", "val_batch_code", "rew_batch_code"]:
                if args[batch_code] is not None and args[batch_code][:2] == "DT":
                    # Could convert to non-DT, but a user that chose DT code for non-DT is really confused...
                    raise ValueError("Using a DT batchcode on a non-DT model")

        if args["final_rew_eval_num"] > 0:
            final_evaluator = CONFIG_NAME_TO_CLASS[args["environment"]].final_evaluator
            if final_evaluator is None:
                raise ValueError(
                    f"final_evaluator is not implemented for env {args['environment']}. Cannot do final reward evals."
                )
            if not args["save_best"] == "rew":
                raise ValueError(
                    "Save the model with best reward if you want to do a final reward eval. (--save_best rew)"
                )

        # If save_best=rew, need to be doing evals
        if args["save_best"] == "rew" and not args["rew_batch_code"]:
            raise ValueError(
                f"rew_batch_code={args['rew_batch_code']} does not have BC evals, which is necessary for save_best=rew."
            )

        if args["model_class"] == "NN" and args["nheads"] != self._all_default_args["nheads"]:
            warning("Setting nheads when model_class=NN has no effect. Setting nheads back to default.")
            args["nheads"] = self._all_default_args["nheads"]

        if args["torch_cpus"] is not None and args["torch_cpus"] < 1:
            raise ValueError("Torch CPU limit must be at least one.")

        if args["eval_types"] is not None:
            eval_types = []
            for et in args["eval_types"]:
                if et not in self.SPECIAL_EVAL_CODES:
                    # Assuming that if it's not a special eval token, then it's a target reward for RC
                    eval_types.append(float(et))
                else:
                    eval_types.append(self.SPECIAL_EVAL_CODES[et])
            args["eval_types"] = eval_types

        self._args = args

    def batch_params(self, batch_code):
        return self.bc_to_params_dict[batch_code]

    @property
    @abc.abstractmethod
    def loss_weights(self):
        """
        Returns:
            dict[str, float]: loss_weights, e.g. {"state": 0.0, "action:" 1.0}
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_data(self, train_batch_code, loss_weights):
        """
        Returns:
            (Dataset, Dataset): test_data, train_data
        """
        raise NotImplementedError()

    def make_env(self):
        """Returns one instance of the environment.
        NOTE: make sure self._args is set before invoking.

        Returns:
            gym.env: The environment.
        """
        raise NotImplementedError()

    def _evaluator(self, rew_eval_num):
        if rew_eval_num == 0:
            return None
        return Evaluator(
            self.make_env,
            rew_eval_num,
            self._args["eval_types"],
            self.max_ep_len,
            self._args["reward_scale"],
            0 if self._args["render"] else None,
        )

    @property
    def train_evaluator(self):
        if self._train_evaluator is None:
            self._train_evaluator = self._evaluator(self._args["train_rew_eval_num"])
        return self._train_evaluator

    @property
    def final_evaluator(self):
        if self._final_evaluator is None:
            self._final_evaluator = self._evaluator(self._args["final_rew_eval_num"])
        return self._final_evaluator

    @staticmethod
    @abc.abstractmethod
    def add_subparser(subparsers):
        """Adds a subparser to the given subparsers object, as well as all env-specific command line arguments.

        Args:
            subparsers (argparse.Subparsers?): argparser subparsers object. See https://docs.python.org/3/library/argparse.html#sub-commands

        Returns:
            (argparse.ArgumentParser): subparser. See https://docs.python.org/3/library/argparse.html#sub-commands
        """
        raise NotImplementedError()


class MinigridConfig(Config):
    # TODO: make sure not using action tanh here
    loss_type = "sce"
    log_interval = 3

    train_evaluator = None
    final_evaluator = None

    default_args = {
        "batch_size": 250,
        "reward_scale": 1,
        "final_rew_eval_num": 0,
        "rew_batch_code": None,  # This is already what's happening, but saying it explicitly to match MujocoConfig.
        "rtg_cat_t": False,
        "save_best": "",
        "torch_cpus": 1,  # otherwise a single run will max out CPU.
    }

    def __init__(self, args, all_default_args):
        super().__init__(args, all_default_args)
        # Useful. But is this the clearest place to set this?
        # NOTE: super().__init__() is nontrivial (sets train/test data) so make sure env_spec is not used for minigrid.
        self._args["env_spec"] = "minigrid"

    @property
    def loss_weights(self):
        loss_weights = {
            "state": self._args["state_loss"],
            "state_key_pos": self._args["state_loss"],
            # "state_key": self._args["state_loss"],
            "action": self._args["action_loss"],
        }
        if self._args["rtg_loss"] is not None:
            loss_weights["rtg"] = self._args["rtg_loss"]
        if self._args["timestep_encoding"]:
            loss_weights["timestep"] = np.nan
        return loss_weights

    def get_data(self, train_batch_code, loss_weights):
        data_name = f"2000_keyenv_{self._args['seq_len']}len"
        test_data_path = os.path.join(TEST_DATA_DIR, data_name)
        dataset = Dataset.load(test_data_path)
        train_data, test_data = dataset.split_data(
            train_prop=self._args["data_prop"],
            num_train_trajs=self._args["num_trajs"],
            num_val_trajs=1000,
        )
        # HACK to make DT work for minigrid key
        if train_batch_code[:2] == "DT":
            # For DT, things would break if there was anything other than a "state" factor.
            # Additionally, for DT, we never try to reconstruct states in the output, so we don't have
            # to keep track of the various components of the state. We simply concatenate the state factors here
            # to avoid the bug, and the input will be identical to before.
            for data in [train_data, test_data]:
                data.state = torch.cat([data.state, data.state_key_pos], dim=2)
                data.state_key_pos = None
            del loss_weights["state_key_pos"]

        return train_data, test_data

    @staticmethod
    def add_subparser(subparsers):
        subparser = subparsers.add_parser("minigrid")
        return subparser

    def make_env(self):
        raise NotImplementedError()

    @property
    def train_evaluator(self):
        return None

    @property
    def final_evaluator(self):
        return None


class MinigridBigConfig(Config):
    # TODO: make sure not using action tanh here
    loss_type = "sce"
    log_interval = 3

    train_evaluator = None
    final_evaluator = None

    default_args = {
        "batch_size": 250,
        "reward_scale": 1,
        "final_rew_eval_num": 0,
        "rew_batch_code": None,  # This is already what's happening, but saying it explicitly to match MujocoConfig.
        "rtg_cat_t": False,
        "save_best": "",
        "torch_cpus": 1,  # otherwise a single run will max out CPU.
    }

    def __init__(self, args, all_default_args):
        super().__init__(args, all_default_args)
        # Useful. But is this the clearest place to set this?
        # NOTE: super().__init__() is nontrivial (sets train/test data) so make sure env_spec is not used for minigrid.
        self._args["env_spec"] = "minigrid"

    @property
    def loss_weights(self):
        loss_weights = {
            "state": self._args["state_loss"],
            "state_key_pos": self._args["state_loss"],
            # "state_key": self._args["state_loss"],
            "action": self._args["action_loss"],
        }
        if self._args["rtg_loss"] is not None:
            loss_weights["rtg"] = self._args["rtg_loss"]
        if self._args["timestep_encoding"]:
            loss_weights["timestep"] = np.nan
        return loss_weights

    def get_data(self, train_batch_code, loss_weights):
        data_name = f"2000_keyenv16x16_{self._args['seq_len']}len"
        test_data_path = os.path.join(TEST_DATA_DIR, data_name)
        dataset = Dataset.load(test_data_path)
        train_data, test_data = dataset.split_data(
            train_prop=self._args["data_prop"],
            num_train_trajs=self._args["num_trajs"],
            num_val_trajs=1000,
        )
        # HACK to make DT work for minigrid key
        if train_batch_code[:2] == "DT":
            # For DT, things would break if there was anything other than a "state" factor.
            # Additionally, for DT, we never try to reconstruct states in the output, so we don't have
            # to keep track of the various components of the state. We simply concatenate the state factors here
            # to avoid the bug, and the input will be identical to before.
            for data in [train_data, test_data]:
                data.state = torch.cat([data.state, data.state_key_pos], dim=2)
                data.state_key_pos = None
            del loss_weights["state_key_pos"]

        return train_data, test_data

    @staticmethod
    def add_subparser(subparsers):
        subparser = subparsers.add_parser("minigridbig")
        return subparser

    def make_env(self):
        raise NotImplementedError()

    @property
    def train_evaluator(self):
        return None

    @property
    def final_evaluator(self):
        return None


class MujocoConfig(Config):
    # TODO: make the config information also loadable and saveable (or make the subtrain only depend on training params! this is nicer),
    #  otherwise impossible to evaluate models post-training on exactly the same subtrain params.
    # TODO: make a class, and have that be loadable and saveable, instead of having training params and model params. Not sure where we can look at examples of this having been done before. Maybe PPO classes in RLLib or something?
    loss_type = "l2"
    log_interval = 10
    leave_for_eval = 30
    max_ep_len = 1000

    default_args = {
        "batch_size": 100,
        "reward_scale": 1000.0,
        "final_rew_eval_num": 1,
        "rew_batch_code": "BC",  # Need BC to have reward evaluations for save_best.
        "save_best": "rew",
    }

    def __init__(self, args, all_default_args):
        super().__init__(args, all_default_args)

    def make_env(self):
        """Returns one instance of the environment.
        NOTE: make sure self._args is set before invoking.

        Returns:
            gym.env: The environment.
        """
        from uniMASK.envs.d4rl.d4rl_data import MUJOCO_GYM_ENV_NAMES

        assert hasattr(self, "_args")
        return gym.make(MUJOCO_GYM_ENV_NAMES[self._args["env_spec"]])

    @property
    def loss_weights(self):
        loss_weights = {
            "state": self._args["state_loss"],
            "action": self._args["action_loss"],
        }
        if self._args["rtg_loss"] is not None:
            loss_weights["rtg"] = self._args["rtg_loss"]
        if self._args["timestep_encoding"]:
            loss_weights["timestep"] = np.nan
        return loss_weights

    def get_data(self, train_batch_code, loss_weights):
        from uniMASK.envs.d4rl.mujoco.data import MujocoDataset

        return MujocoDataset.get_datasets(
            env_name=self._args["env_spec"],
            dataset_info={"expert_type": self._args["expert_type"]},
            prop=self._args["data_prop"],
            num_train_trajs=self._args["num_trajs"],
            leave_for_eval=self.leave_for_eval,
        )

    @staticmethod
    def add_subparser(subparsers):
        from uniMASK.envs.d4rl.d4rl_data import MUJOCO_NAMES

        subparser = subparsers.add_parser("mujoco")
        subparser.add_argument("--env_spec", type=str, default="hopper", choices=MUJOCO_NAMES.keys())
        subparser.add_argument(
            "--expert_type",
            "-exp",
            type=str,
            default="medium",
            help="Expert type: medium, expert, etc.",
        )
        # TODO this should be supported by minigrid too, down the line.
        subparser.add_argument(
            "--rtg_cat_t",
            "-rct",
            default=False,
            action="store_true",
            help="Concatenate timesteps with rtg tokens. Better version of positional encoding.",
        )
        return subparser

    def _set_train_test_data(self):
        super()._set_train_test_data()
        if self._args["rtg_cat_t"]:
            assert "timestep" not in self.loss_weights
            self.train_data = self.train_data.cat_t_and_rtg(self.max_ep_len)
            self.test_data = self.test_data.cat_t_and_rtg(self.max_ep_len)


class MazeConfig(MujocoConfig):
    def make_env(self):
        from uniMASK.data.datasets.generate_maze2d import make_maze
        from uniMASK.envs.d4rl.d4rl_data import MUJOCO_GYM_ENV_NAMES

        assert hasattr(self, "_args")
        return make_maze(
            MUJOCO_GYM_ENV_NAMES[self._args["env_spec"]],
            self._args["horizon"],
            self._args["no_randomness"],
        )

    def get_data(self, train_batch_code, loss_weights):
        from uniMASK.envs.d4rl.d4rl_data import MUJOCO_GYM_ENV_NAMES
        from uniMASK.envs.d4rl.maze.data import MazeDataset

        # Could potentially have less code reuse if we make MazeDataSet and
        # dataset_info an attribute (the rest is the same), but I think this
        # is clearer.
        return MazeDataset.get_datasets(
            env_name=MUJOCO_GYM_ENV_NAMES[self._args["env_spec"]],
            dataset_info={"horizon": self._args["horizon"]},
            prop=self._args["data_prop"],
            num_train_trajs=self._args["num_trajs"],
            leave_for_eval=30,
        )

    @staticmethod
    def add_subparser(subparsers):
        from uniMASK.envs.d4rl.d4rl_data import MAZE_NAMES

        subparser = subparsers.add_parser("maze")
        subparser.add_argument("--env_spec", type=str, default="medium", choices=MAZE_NAMES)
        subparser.add_argument(
            "--horizon",
            dest="horizon",
            type=int,
            default=200,
            help="Horizon of training data and evaluations (note: you should first generate the data with datasets/generate_maze2d.py).",
        )
        subparser.add_argument(
            "--no_randomness",
            action="store_true",
            help="Use fixed start and goal positions in eval rollouts (maze2d only).",
        )
        subparser.add_argument(
            "--rtg_cat_t",
            "-rct",
            default=False,
            action="store_true",
            help="Concatenate timesteps with rtg tokens. Better version of positional encoding.",
        )
        return subparser


CONFIG_NAME_TO_CLASS = {
    "mujoco": MujocoConfig,
    "maze": MazeConfig,
    "minigrid": MinigridConfig,
    "minigridbig": MinigridBigConfig,
}
