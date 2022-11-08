

import argparse
import time

import numpy as np
import torch

import uniMASK.utils
from uniMASK.trainer import Trainer
from uniMASK.transformer import STR_TO_MODEL_CLASS
from uniMASK.utils import imdict


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--setting",
        default=None,
        type=str,
        help="Qualitative setting",
    )

    parser.add_argument(
        "--task",
        default=None,
        type=str,
        help="Which task are we training on",
    )

    parser.add_argument(
        "--n",
        default=10000,
        type=int,
        help="Dataset Size",
    )
    return parser


# from uniMASK.utils import profile
# @profile
def train_from_params(
    train_trajs,
    val_trajs,
    model_params,
    training_params,
    dataset_name=None,
    train_evaluator=None,
    final_evaluator=None,
):
    """
    Trains a model on the data provided, with the provided model and training parameters

    TODO: clean up all these different training functions. Why do we have so much complexity? What are best practices?
    """
    mp, tp = model_params, training_params
    # NOTE: Currently we have deterministic seeding to ensure reproducibility
    np.random.seed(tp["seed"])
    torch.manual_seed(tp["seed"])

    # TODO: find a better way to determine max factor size
    dummy_data = train_trajs.get_rnd_batch(
        batch_size=1,
        seq_len=tp["seq_len"],
        input_keys=set(tp["loss_weights"]),
        loss_types=tp["loss_types"],
        stacked=tp["stacked"],
        rew_scale=tp["rew_scale"],
    )
    max_fact_size = dummy_data.max_factor_size

    # TODO: add the model class to the model params dict
    model_class = STR_TO_MODEL_CLASS[tp["model_class"]]
    model_class.validate_training_params(tp)
    model = model_class(
        train_dataset=train_trajs,
        cat_input_dim=max_fact_size,
        input_dims=dummy_data.factor_dims,
        **mp,
    )
    trainer = Trainer(
        model=model.to(uniMASK.utils.DEVICE),
        train_evaluator=train_evaluator,
        final_evaluator=final_evaluator,
        extra_config={"dataset_name": dataset_name},
        **tp,
    )
    trainer.train(train_trajs, val_trajs)
    return trainer


def base_training_params(
    loss_weights,
    loss_types,
    seq_len=10,
    epochs=20,
    batch_size=2,
    log_interval=1,
    eval_interval=1,
    rew_eval_interval=5,
    save_best="",
    lr=0.0008,
    stacked=True,
    run_name=None,
    train_batch_params_n=None,
    val_batch_params_n=None,
    rew_batch_params_n=None,
    seed=0,
    model_class="FB",
    trajs_prop=None,
    wandb_logging=False,
    rew_scale=1,
    val_loss_weights=None,
    rtg_cat_t=False,
):
    """
    Returns a set of default training params unless overwritten
    """
    if val_batch_params_n is None:
        # Will be made immutable so no need to deepcopy
        val_batch_params_n = tuple(train_batch_params_n)
    if rew_batch_params_n is None:
        rew_batch_params_n = []

    if val_loss_weights is None:
        # By default, consider everything equally. This allows to play around with the
        # training loss weights while still having consistent validation plots across those runs
        val_loss_weights = {k: 1 for k in loss_weights}
        # TODO: clean this hack
        if "rtg" in val_loss_weights:
            # RTG prediction is something that we currently don't want to do or account for
            val_loss_weights["rtg"] = 0

    # Turn all batch params objects to being immutable (prevents downstream bugs)
    for bp_n in [train_batch_params_n, val_batch_params_n, rew_batch_params_n]:
        for bp in bp_n:
            assert isinstance(bp, imdict), "All batch parameters should be immutable dicts `imdict(•)`"

    hyperparam_string = "lr{}".format(lr)
    if run_name is None:
        run_name = time.strftime("%l_%M%p%b%d_{}".format(hyperparam_string))

    training_params = {
        "stacked": stacked,
        "loss_weights": loss_weights,
        "loss_types": loss_types,
        "val_loss_weights": val_loss_weights,  # The loss weights used for validation (so that logging can be consistent)
        "seq_len": seq_len,
        "epochs": epochs,  # The number of epochs
        "batch_size": batch_size,
        "log_interval": log_interval,
        "eval_interval": eval_interval,
        "rew_eval_interval": rew_eval_interval,
        "save_best": save_best,  # Save best model (wrt val loss or rew) during training
        "train_batch_params_n": train_batch_params_n,
        "val_batch_params_n": val_batch_params_n,
        "rew_batch_params_n": rew_batch_params_n,
        "run_name": run_name,
        "lr": lr,
        "wandb_logging": wandb_logging,
        "seed": seed,
        "model_class": model_class,
        "trajs_prop": trajs_prop,
        "rew_scale": rew_scale,
        "rtg_cat_t": rtg_cat_t,
    }
    return training_params


def base_model_params(
    feedforward_nhid=128,
    nlayers=2,
    nheads=2,
    embed_dim=128,
    dropout=0,
    seq_len=None,
    timestep_encoding=False,
    action_tanh=False,
):
    """Returns a set of default model params unless overwritten"""
    model_params = {
        "embed_dim": embed_dim,  # embedding dimension
        "feedforward_nhid": feedforward_nhid,  # the dimension of the feedforward network model in nn.TransformerEncoder
        "nlayers": nlayers,  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        "nheads": nheads,  # the number of heads in the multiheadattention models
        "dropout": dropout,  # the dropout value
        "seq_len": seq_len,
        "timestep_encoding": timestep_encoding,
        "action_tanh": action_tanh,
    }
    return model_params
