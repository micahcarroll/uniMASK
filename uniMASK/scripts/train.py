"""
Evaluate different masking methods (rnd, simple action pred, all action pred) on the DT/BC-like first action pred loss.
"""
#!/usr/bin/env python
# coding: utf-8

import random

import numpy as np
import torch
import wandb

from uniMASK.data import DATA_DIR
from uniMASK.scripts.configs import Config, parse
from uniMASK.trainer import Trainer
from uniMASK.transformer_train import base_model_params, base_training_params, train_from_params


# TODO this should be moved to configs.py (and this might resolve the "num_trajs specified separately" issue, too).
def args_to_name(args, num_trajs):
    # num_trajs specified separately for compatability with non-minigrid environements.
    name = f"{num_trajs}N_{args['seq_len']}len_{args['train_batch_code']}"
    if args["rtg_loss"] is not None:
        rtg_loss_string = "" if args["rtg_loss"] == 0 else str(args["rtg_loss"])
        name += "_rl" + rtg_loss_string
    if not np.allclose([args["state_loss"]], [1]):
        name += f"_sl{args['state_loss']}"
    if not np.allclose([args["action_loss"]], [1]):
        name += f"_al{args['action_loss']}"
    if args["timestep_encoding"]:
        name += "_t_enc"
    if args["model_class"] == "NN":
        name += "_NN"
    elif args["model_class"] == "DT":
        name += "_DT"
    if args["finetune"]:
        name = f"{args['finetune'][:-6]}_finetune_{args['train_batch_code']}"
    if args["suffix"]:
        name += f"_{args['suffix']}"
    if args["rnd_suffix"]:
        # Get 5 character random code (~ 1 in 10^6 chance of conflict)
        # TODO make independent of seeding, right now it will be fully determined by _args["seed"].
        name += "_" + ("%032x" % random.getrandbits(128))[-5:]
    return name


def experiment(config):
    # Common variables
    # TODO ultimately there should be no accesses to config._args; Config should support any access in its API...
    train_batch_code = config._args["train_batch_code"]
    val_batch_code = config._args["val_batch_code"]
    rew_batch_code = config._args["rew_batch_code"]
    seed = config._args["seed"]
    seq_len = config._args["seq_len"]
    wandb_logging = config._args["wandb_project"] is not None
    stacked = not config._args["unstacked"]
    rtg_cat_t = config._args["rtg_cat_t"]

    if config._args["torch_cpus"] is not None:
        torch.set_num_threads(config._args["torch_cpus"])

    # Get loss weights and types
    loss_weights = config.loss_weights
    loss_types = config.loss_type

    train_data, test_data = config.train_data, config.test_data

    # Should be validated at the parser level!
    assert train_batch_code in Config.possible_batch_codes
    assert val_batch_code in Config.possible_batch_codes
    train_batch_params_n = config.batch_params(train_batch_code)
    val_batch_params_n = config.batch_params(val_batch_code)

    if rew_batch_code is not None:
        assert config.train_evaluator is not None, "Using rew batch code but no evaluator for env"
        assert rew_batch_code in Config.possible_batch_codes
        rew_batch_params_n = config.batch_params(rew_batch_code)
    else:
        rew_batch_params_n = None

    name = args_to_name(config._args, train_data.num_trajs)  # TODO see todo near args_to_name() def.

    tp = base_training_params(
        loss_weights=loss_weights,
        loss_types=loss_types,
        seq_len=seq_len,
        epochs=config._args["epochs"],
        batch_size=config._args["batch_size"],
        log_interval=config.log_interval,
        eval_interval=min(10, config._args["epochs"]),
        rew_eval_interval=config._args["rew_eval_freq"],
        save_best=config._args["save_best"],
        lr=config._args["lr"],
        stacked=stacked,
        run_name=name,
        train_batch_params_n=train_batch_params_n,
        val_batch_params_n=val_batch_params_n,
        rew_batch_params_n=rew_batch_params_n,
        wandb_logging=wandb_logging,
        model_class=config._args["model_class"],
        rtg_cat_t=rtg_cat_t,
        seed=seed,
    )

    wandb_run = None
    if wandb_logging:
        wandb_run = wandb.init(
            project=f"{config._args['wandb_project']}_{config._args['env_spec']}",
            group=name,
            reinit=True,  # To be able to have multiple .init calls in the same script
            name=f"{name}_seed{seed}",
            dir=DATA_DIR,
            config=tp,
            tags=config._args["wandb_tags"],
        )

    print(f"Starting run {name}")
    if config._args["finetune"]:
        trainer = Trainer.load_from_name_and_seed(f"{config._args['finetune']}", **tp, best=True, finetune=True)
        trainer.train(train_data, test_data)
    else:
        mp = base_model_params(
            nlayers=config._args["nlayers"],
            nheads=config._args["nheads"],
            embed_dim=config._args["embed_dim"],
            feedforward_nhid=config._args["feedforward_nhid"],
            seq_len=seq_len,
            dropout=config._args["dropout"],
            timestep_encoding=config._args["timestep_encoding"],
            action_tanh=config._args["action_tanh"],
        )
        trainer = train_from_params(
            train_data,
            test_data,
            mp,
            tp,
            train_evaluator=config.train_evaluator,
            final_evaluator=config.final_evaluator,
        )

    if config._args["final_rew_eval_num"] > 0:
        trainer.end_of_experiment_rew_eval(train_data)

    if wandb_run is not None:
        wandb_run.finish()


def run(_input=None):
    """
    Args:
        _input (str, optional): Command line args. Default (None) reads from sys.argv.
    """
    config = parse(_input)
    experiment(config)


if __name__ == "__main__":
    run()
