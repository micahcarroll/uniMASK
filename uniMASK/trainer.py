import os
import time
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import wandb

import uniMASK.base as bs
from uniMASK.batches import BATCH_TYPES_BY_NAME, Batch

# Maximum batch size the GPU can handle at once
from uniMASK.data import TRANSFORMER_DIR
from uniMASK.transformer import STR_TO_MODEL_CLASS, CustomModel, DecisionTransformer
from uniMASK.utils import (
    average_dictionaries,
    create_dir_if_not_exists,
    delete_dir_if_exists,
    load_from_json,
    mean_and_std_err,
    save_as_json,
    to_numpy,
)

MAX_BATCH_SIZE = 8000
# Minimum batch size for validation loss computation when not working locally
MIN_VALID_BS = 5000


class BCEvalIndicator:
    # A magic token that tells Evaluator to do BC evaluation.
    pass


class RCAutoEvalIndicator:
    # A magic token that tells Evaluator to do automtic-reward-target RC evaluation.
    pass


class Trainer:
    """
    TODO: update this documentation depending on how we've decided to resolve this issue through loading and saving for
     fine-tuning
    One design decision for the trainer was whether it should support multiple training runs, or whether each
    Trainer instance should only be associated with one training run. In that sense, the Trainer class is maybe
    better thought of as a TrainingRun class. Here are some pros and cons:

    If one-time:
    + Have everything in init and as attribute, easy to get info about the run even after loading.
    + Conceptually simple. One trainer, one training session.
    - Can't easily display finetuning as a continuation of the run.

    If multi-usage:
    + Can keep track of global timestep, keep a pointer to the wandb instance for continuous logging, etc.
    - When save the info, can't save LR because it could have varied across the various trainings. Not really saving info about multiple training

    For simplicity I've currently decided to have the trainer only be used for one training run. For fine-tuning,
    we create a new trainer, that defaults to the parameters of the loaded trainer, but for which we can override
    parameters at loading time.
    """

    TRAINER_PARAMS_SAVENAME = "trainer_params"

    def __init__(self, model, train_evaluator, final_evaluator, last_eval_metrics=None, **kwargs):
        assert isinstance(model, CustomModel) or isinstance(model, DecisionTransformer)

        for k, v in kwargs.items():
            self.__dict__[k] = v

        self.input_keys = set(self.loss_weights)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        # Can also define a scheduler here. Currently not using it
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=gamma)

        self.model = model
        self.last_eval_metrics = last_eval_metrics
        self.train_evaluator = train_evaluator
        if self.train_evaluator is not None:
            self.train_evaluator.set_trainer(self)
        self.final_evaluator = final_evaluator
        if self.final_evaluator is not None:
            self.final_evaluator.set_trainer(self)

    @property
    def params(self):
        # Things that we'll deal with separately (models and optimizer) or ignore (writer and best_model_state_dict)
        params_to_ignore = [
            "optimizer",
            "model",
            "writer",
            "best_model_state_dict",
            "input_keys",
        ]
        return {k: v for k, v in self.__dict__.items() if k not in params_to_ignore}

    def get_save_dir(self, create_if_not_exist=False):
        save_dir = os.path.join(TRANSFORMER_DIR, self._name_and_seed(self.run_name, self.seed))
        if create_if_not_exist:
            create_dir_if_not_exists(save_dir)
        return save_dir

    def delete_old_trainer_files(self):
        """
        Delete previously stored Trainer at the location if they exist. This is so that there is no mix of files that
        occur between different runs with the same name---for example, if runs differ only in save interval or save_best
        """
        save_dir = self.get_save_dir()
        print(f"Deleting directory {save_dir}, and creating new one.")
        delete_dir_if_exists(save_dir)
        create_dir_if_not_exists(save_dir)

    @staticmethod
    def _name_and_seed(save_name, seed):
        """Formatting for name and seed"""
        return f"{save_name}_seed{seed}"

    def save(self, global_t=None, is_best=False):
        """Save the current Trainer for the current epoch at the specified location"""
        save_dir = self.get_save_dir(create_if_not_exist=True)

        if global_t is not None:
            save_dir = os.path.join(save_dir, f"timestep{global_t}")
            create_dir_if_not_exists(save_dir)

        self.save_at_location(save_dir, is_best=is_best)

    def save_at_location(self, save_dir, is_best=False):
        # Save trainer info
        trainer_params_path = os.path.join(save_dir, self.TRAINER_PARAMS_SAVENAME)

        immutable_ks = [
            "train_batch_params_n",
            "val_batch_params_n",
            "rew_batch_params_n",
        ]
        # TODO do something more graceful for "evaluator" key
        to_ignore = ["train_evaluator", "final_evaluator"]
        params_to_save = {k: deepcopy(v) for k, v in self.params.items() if k not in immutable_ks + to_ignore}
        for k in immutable_ks:
            params_to_save[k] = tuple(dict(bp) for bp in self.params[k])
            # Deepcopy further to avoid bug if different params are actually the same mutable object
            # Transform batch classes into strings to allow for saving
            for bp in params_to_save[k]:
                bp["type"] = bp["type"].__name__

        save_as_json(params_to_save, trainer_params_path)

        # Save model
        self.model.save(save_dir, is_best=is_best)

    @classmethod
    def load(
        cls,
        save_name,
        train_evaluator=None,
        final_evaluator=None,
        seed=0,
        best=False,
        **trainer_params_to_update,
    ):
        """Takes in model name and seed separately, and parses it before loading"""
        return cls.load_from_name_and_seed(
            cls._name_and_seed(save_name, seed),
            train_evaluator=train_evaluator,
            final_evaluator=final_evaluator,
            best=best,
            **trainer_params_to_update,
        )

    @classmethod
    def load_from_name_and_seed(
        cls,
        save_name,
        train_evaluator=None,
        final_evaluator=None,
        best=False,
        finetune=False,
        **trainer_params_to_update,
    ):
        """Takes in save name (with seed information)"""
        save_dir = os.path.join(TRANSFORMER_DIR, save_name)

        # Load trainer info
        trainer_params_path = os.path.join(save_dir, cls.TRAINER_PARAMS_SAVENAME)
        trainer_params = load_from_json(trainer_params_path)

        # Transform batch class strings back into classes to allow for loading
        for bp_n_type in ["train", "val", "rew"]:
            bp_n_name = f"{bp_n_type}_batch_params_n"
            for bp in trainer_params[bp_n_name]:
                bp["type"] = BATCH_TYPES_BY_NAME[bp["type"]]

        # Load model
        model_class = STR_TO_MODEL_CLASS[trainer_params["model_class"]]
        model = model_class.load(save_dir, best)

        # Change parameters (for obtaining a Trainer which can be used for fine-tuning)
        for k, v in trainer_params_to_update.items():
            assert k in trainer_params, f"Trying to overwrite a parameter {k} that didn't exist in the saved Trainer"
            if v != trainer_params[k]:
                print(f"Overwriting Trainer param {k} from {trainer_params[k]} to {v}\n")
                trainer_params[k] = v

        # Remove wandb logging if loading but not for finetuning purposes
        if not finetune:
            trainer_params["wandb_logging"] = False
        return cls(model, train_evaluator, final_evaluator, **trainer_params)

    @staticmethod
    def print_eval_info(
        val_loss_metrics,
        rew_metrics,
        training_start_time,
        global_t,
    ):
        elapsed = time.time() - training_start_time
        print(f"{'-' * 89}\n" f"| timestep {global_t:3d} | time: {elapsed:5.2f}s")

        for valid_loss_name, valid_loss in val_loss_metrics.items():
            if "total" not in valid_loss_name:
                continue
            eval_b_type = valid_loss_name[:-6]
            print(f"Valid loss {eval_b_type}: {valid_loss:5.4f}")

        for rbp_logging_name, rew_metrics_per_tr in rew_metrics.items():
            if rbp_logging_name == "global_t":
                continue
            for target_rew, m in rew_metrics_per_tr.items():
                print(
                    f"| {rbp_logging_name}_{Trainer.tr_to_str(target_rew)} "
                    f"rew: {m['eval_avg_rew']:5.4f}, {m['eval_se_rew']:5.2f} "
                    f"len: {m['eval_avg_len']:5.1f}, {m['eval_se_len']:5.2f} "
                    f"time {m['rew_eval_time']:5.2f}"
                )
        print(f"{'-' * 89}")

    def train(self, train_trajs, val_trajs):
        """
        Trains the Trainer object according to the training_params
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        print("Training on {}".format(bs.DEVICE))

        assert self.eval_interval <= self.epochs, "eval_interval should be < epochs"

        best_val_loss = float("inf")
        best_rew = -float("inf")
        rew_evals = {}
        train_start_time = time.time()

        print("Starting training...")
        global_t = 0

        print("Initial evaluation run...")
        validation_losses = self.val_loss_evaluation(val_trajs, global_t)
        rew_metrics = self.rew_evaluation_and_logging(train_trajs, global_t)
        self.print_eval_info(validation_losses, rew_metrics, train_start_time, global_t)

        self.delete_old_trainer_files()

        # Start from 1 because we already do an eval before this loop
        for epoch in range(1, self.epochs + 1):
            global_t = self.training_itereration(train_trajs, global_t, epoch)

            # Reward evaluation
            if epoch % self.rew_eval_interval == 0:
                rew_metrics = self.rew_evaluation_and_logging(train_trajs, global_t)
                rew_evals[global_t] = rew_metrics

                if self.save_best == "rew":
                    # If doing both BC and RC evaluation, consider both when saving the model
                    curr_rew = []
                    if "BC" in rew_metrics:
                        curr_rew.append(rew_metrics["BC"][BCEvalIndicator]["eval_avg_rew"])
                    if "RC" in rew_metrics:
                        rc_eval_types = list(rew_metrics["RC"].keys())
                        assert (
                            len(rc_eval_types) == 1
                        ), "If more than one RC eval type, unclear what we should use to save best"
                        curr_rew.append(rew_metrics["RC"][rc_eval_types[0]]["eval_avg_rew"])

                    if sum(curr_rew) > best_rew:
                        best_rew = sum(curr_rew)
                        self.save(is_best=True)
                        print(f"Saving best model (ep {epoch}) with total rew {curr_rew}")
                    if self.wandb_logging:
                        wandb.log({"best_rew": best_rew}, step=global_t)

            # Validation loss computation
            if epoch % self.eval_interval == 0:
                validation_losses = self.val_loss_evaluation(val_trajs, global_t)

                # NOTE: we print eval_info every validation loss computation because we assume that that's going to
                #  be more often than every reward evaluation.
                self.print_eval_info(validation_losses, rew_metrics, train_start_time, global_t)

                if self.save_best == "loss":
                    # TODO Orr++: maybe have save_batch_params which specify which types to pay attention to. This whole section
                    #  is kind of hacky
                    # We want to save the best model according to some metric. Here we have some logic to figure out
                    # the relevant metric. To do so, we consider the training batch types.
                    tr_b_types = [bp["type"] for bp in self.train_batch_params_n]

                    # What is the overlap between training and validation batch types?
                    matching_eval_type = [
                        vbp["logging_name"] for vbp in self.val_batch_params_n if vbp["type"] in tr_b_types
                    ]
                    # Somwhat of a hack: don't consider the validation losses for forwards and backwards as they
                    # generally continue decreasing indefinitely even as other tasks overfit (so considering the
                    # full-total-loss would actually not be the "best" model by intuitive standards)
                    val_ls_to_skip = ["forwards", "backwards"]

                    if len(matching_eval_type) == 0:
                        # If there's no overlap between the training and validation (e.g. RandomPred training),
                        # consider the total loss across all validation types.
                        # TODO Orr++, HACK: filters out forwards and backwards losses because they would mess things up
                        losses_to_consider = [
                            (name, l)
                            for name, l in validation_losses.items()
                            if "total" in name and not any(name.startswith(ign) for ign in val_ls_to_skip)
                        ]
                        matching_losses = [x[1] for x in losses_to_consider]
                    elif len(matching_eval_type) == 1:
                        # If there is overlap of exactly 1, then we're not in the rnd or all training regime (TODO assert this)
                        # and the only loss we care about is the single-task loss
                        matching_losses = [validation_losses[f"{matching_eval_type[0]}_total"]]
                    else:
                        # If there is overlap (of more than 1), then we're in the ALL training regime (TODO assert this).
                        # So we should consider the total loss except for forwards and backwards, again.

                        # To prevent overfitting, one option is just to consider the loss for BC and RC, which
                        # are the tasks that overfit the most. This leads the rnd model to be underfit in the worst
                        # case, which can be fixed by fine-tuning
                        # matching_eval_type = ["BC", "RC"]

                        matching_losses = [
                            validation_losses[f"{matching_eval}_total"]
                            for matching_eval in matching_eval_type
                            if matching_eval not in val_ls_to_skip
                        ]

                    total_eval_loss = sum(matching_losses)

                    if total_eval_loss < best_val_loss:
                        best_val_loss = total_eval_loss
                        self.save(is_best=True)
                        print(f"Saving best model (ep {epoch}) with total loss {total_eval_loss}")
                    if self.wandb_logging:
                        wandb.log({"best_val_loss": best_val_loss}, step=global_t)

            # This is where scheduler would be called one were to re-enable it
            # scheduler.step()
        self.save_last_eval_metrics(validation_losses, rew_metrics)

    # NOTE: This stores rew_eval_metrics by references, so be sure the pointer changes next iter.
    def save_last_eval_metrics(self, eval_metrics, rew_eval_metrics, global_t=None):
        for metric in rew_eval_metrics.values():
            if isinstance(metric, dict) and BCEvalIndicator in metric:
                # For saving json
                metric[self.tr_to_str(BCEvalIndicator)] = metric.pop(BCEvalIndicator)
            if isinstance(metric, dict) and RCAutoEvalIndicator in metric:
                # For saving json
                metric[self.tr_to_str(RCAutoEvalIndicator)] = metric.pop(RCAutoEvalIndicator)
        self.last_eval_metrics = {k: float(to_numpy(v)) for k, v in eval_metrics.items()}
        self.last_eval_metrics.update(rew_eval_metrics)
        self.save(global_t=global_t)

    def training_itereration(self, data_source, global_t, epoch):
        """Performs one epoch of training"""
        # Following command can be useful for debugging nans
        # torch.autograd.set_detect_anomaly(True)

        # Total number of training datapoints
        num_datapoints = data_source.total_timesteps

        self.model.train()  # Turn on the train mode
        batch_metrics_n = []

        start_time = time.time()

        # How many batches we have to go through before we have gone through N
        # datapoints, where N is the size of the training data.
        num_batches_per_epoch = max(num_datapoints // (self.batch_size * self.seq_len), 1)

        log_interval = min(self.log_interval, num_batches_per_epoch)

        # There will at a minimum be one batch per epoch (basically, we are strictly ciel-ing the number of batches)
        for batch_count in range(num_batches_per_epoch):
            self.optimizer.zero_grad()

            # Get a batch of size 'batch_size' from the data_source
            curr_batch_data = data_source.get_rnd_batch(
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                input_keys=self.input_keys,
                loss_types=self.loss_types,
                stacked=self.stacked,
                rew_scale=self.rew_scale,
            )

            # Iterating through batch types at every batch
            batch_type_idx = (global_t // self.batch_size) % len(self.train_batch_params_n)
            batch_params = self.train_batch_params_n[batch_type_idx]
            curr_batch = Batch.from_params(curr_batch_data, batch_params)

            self.model(curr_batch)
            batch_metrics_n.append(self.get_loggable_metrics_dict(curr_batch.compute_loss_and_acc(self.loss_weights)))

            # Can visualize the network with the following command
            # make_dot(loss, self.model.state_dict()).view()

            curr_batch.loss.backward()

            # Useful nan debugging util
            # print(
            #     "max gradient",
            #     [
            #         (name, float(to_numpy(torch.max(p.grad.ravel()))))
            #         for name, p in self.model.named_parameters()
            #     ],
            # )

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            global_t += self.batch_size

            if (batch_count + 1) % log_interval == 0:
                interval_avg_metrics_d = average_dictionaries(batch_metrics_n)
                cur_loss = interval_avg_metrics_d["total"]

                elapsed = time.time() - start_time
                ms_per_batch = elapsed * 1000 / log_interval
                print(
                    f"epoch {epoch:3d} | {batch_count + 1:3d}/{num_batches_per_epoch:3d} batches | "
                    f"lr {self.lr:.0e} | ms/batch {ms_per_batch:3.0f} | loss {cur_loss:5.4f} |"
                )
                # scheduler.get_last_lr()[0]

                # Logging losses
                interval_metrics = {f"{k}_train": v for k, v in interval_avg_metrics_d.items()}
                interval_metrics.update({"timestep": global_t})

                if self.wandb_logging:
                    wandb.log(interval_metrics, step=global_t)

                batch_metrics_n = []
                start_time = time.time()

        return global_t

    def rew_evaluation_and_logging(self, dataset=None, global_t=None):
        """
        Perform evaluation rollouts with the model to see how well it performs in sequential settings.

        dataset: dataset is required by the evaluator if doing RC_auto evals in the Maze env. Can ignore for others.
        global_t: when performing eval during training, this the timestep is also logged
        """
        metrics_to_log, metrics_to_print = self.perform_rew_eval_and_get_metrics(self.train_evaluator, dataset)

        if self.wandb_logging:
            wandb.log(metrics_to_log, step=global_t)
        metrics_to_print["global_t"] = global_t
        return metrics_to_print

    def perform_rew_eval_and_get_metrics(self, evaluator, dataset):
        if self.rew_eval_interval < 0 or evaluator is None:
            return {}, {}

        with torch.no_grad():
            self.model.eval()  # Turn on the evaluation mode

            eval_start_time = time.time()
            rew_metrics = evaluator.evaluate(dataset=dataset)
            rew_eval_time = time.time() - eval_start_time

        metrics_to_print = defaultdict(dict)
        metrics_to_log = {}  # wandb_backwards_compatible_metrics
        for rbp_log_name in rew_metrics:  # rbp = reward_batch_params
            for target_reward, curr_metrics in rew_metrics[rbp_log_name].items():
                eval_rews = curr_metrics["ep_rews"]
                eval_lens = curr_metrics["ep_lens"]
                eval_avg_rew, eval_std_rew = mean_and_std_err(eval_rews)
                eval_avg_len, eval_std_len = mean_and_std_err(eval_lens)

                rew_eval_metrics = {
                    "eval_avg_rew": eval_avg_rew,
                    "eval_se_rew": eval_std_rew,
                    "eval_avg_len": eval_avg_len,
                    "eval_se_len": eval_std_len,
                    "rew_eval_time": rew_eval_time,
                    "rew_eval_steps_per_sec_avg": eval_avg_len / rew_eval_time,
                    "rew_eval_steps_per_sec_max": max(eval_lens) / rew_eval_time,
                }
                metrics_to_print[rbp_log_name][target_reward] = rew_eval_metrics
                target_rew = self.tr_to_str(target_reward)
                metrics_to_log.update(
                    {f"{rew_metric}_{rbp_log_name}{target_rew}": v for rew_metric, v in rew_eval_metrics.items()}
                )
        return metrics_to_log, metrics_to_print

    # This is its own method for consistency across logs.
    @staticmethod
    def tr_to_str(target_rew):
        """Convert a target_reward to a string."""
        if target_rew is BCEvalIndicator:
            return "BC"
        if target_rew is RCAutoEvalIndicator:
            return "RC"
        if type(target_rew) is float:
            return f"{float(target_rew):5.2f}"
        if target_rew == "BC":  # TODO Orr this is a bug and should be fixed upstream!
            return "BC"
        raise NotImplementedError(f"Unexpected target_reward: {target_rew} of type {type(target_rew)}")

    def val_loss_evaluation(self, data_source, global_t):
        """Do an evaluation pass on the entire validation set. Computes _validation losses_, not _validation rewards_"""
        with torch.no_grad():
            self.model.eval()  # Turn on the evaluation mode

            metrics_dicts = []

            num_batches = 1
            # Getting number of possible sequences from the data
            batch_size = data_source.get_tot_num_possible_seqs(self.seq_len)

            if not bs.LOCAL:
                # if the validation set is small, it's probably good to just resample it multipple times in order
                # to get multiple different maskings
                batch_size = max(batch_size, MIN_VALID_BS)

            # Generally, we can save time by evaluating on the validation set all at once.
            # Only problem is if validation set won't fit in CUDA memory all at the same time.
            # We have a heuristic MAX_BATCH_SIZE which is what we expect can fit in memory at once.

            # If there are more seqs than we expect can fit in memory, just sample various batches
            # from the eval set at random, and average the losses.
            if batch_size > MAX_BATCH_SIZE:
                num_batches = int(batch_size // MAX_BATCH_SIZE)
                batch_size = MAX_BATCH_SIZE
                print(
                    "Eval dataset too large to perform evaluation in one pass. Will use {} batches".format(num_batches)
                )

            batch_count = 0
            while batch_count < num_batches:
                token_seq_batch = data_source.get_rnd_batch(
                    batch_size=batch_size,
                    seq_len=self.seq_len,
                    input_keys=self.input_keys,
                    loss_types=self.loss_types,
                    stacked=self.stacked,
                    rew_scale=self.rew_scale,
                )
                b_metrics = self.get_eval_metrics_from_seqs(token_seq_batch, self.val_loss_weights)
                metrics_dicts.append(b_metrics)
                batch_count += 1

            evaluation_avg_metrics_d = average_dictionaries(metrics_dicts)
            evaluation_avg_metrics_d["batch_count"] = num_batches
            eval_metrics = {f"{k}_valid": v for k, v in evaluation_avg_metrics_d.items()}

            if self.wandb_logging:
                wandb.log(eval_metrics, step=global_t)

            return evaluation_avg_metrics_d

    def get_eval_metrics_from_seqs(self, token_seq_batch, loss_weights):
        """
        From a sampled set of token sequences, create a Batch object, and get metrics
        """
        batch_size = token_seq_batch.num_seqs

        metrics_dicts = {}
        for val_batch_params in self.val_batch_params_n:
            # Given that the forward pass mutates the input, make a copy before it
            eval_data_seqs = deepcopy(token_seq_batch)

            b_class = val_batch_params["type"]
            # Batch size has to be a multiple of this number or everything breaks
            # TODO Orr+: add this logic to fix the batch size in the batch __init__ itself with a try/catch.
            #  Currently it fixes the size even if not necessary because it can't take into account the batch params
            batch_size_multiple = b_class.must_have_size_multiple_of(eval_data_seqs.seq_len)

            assert batch_size >= batch_size_multiple, "Batch size has to be a multiple of {}. Was {}".format(
                batch_size_multiple, batch_size
            )
            if batch_size % batch_size_multiple != 0:
                new_batch_size = batch_size - (batch_size % batch_size_multiple)
                print("Fixing eval batch size from", batch_size, "to", new_batch_size)
                eval_data_seqs = eval_data_seqs[:new_batch_size]

            batch = Batch.from_params(eval_data_seqs, val_batch_params)

            self.model(batch)
            loggable_metrics = self.get_loggable_metrics_dict(batch.compute_loss_and_acc(loss_weights))
            for metric_name, metric_val in loggable_metrics.items():
                prefix = val_batch_params.get("logging_name", b_class.__name__)
                metrics_dicts[f"{prefix}_{metric_name}"] = metric_val
        return metrics_dicts

    def get_loggable_metrics_dict(self, loss_dict):
        metrics_to_log = {"total": loss_dict["total"].item()}
        if self.stacked:
            for k, v in loss_dict["all"].items():
                metrics_to_log[k] = v.item()
        else:
            for k in loss_dict:
                if k == "total":
                    continue
                assert len(list(loss_dict[k].keys())) == 1
                metrics_to_log[k] = loss_dict[k][k].item()
        return metrics_to_log

    def end_of_experiment_rew_eval(self, dataset=None):
        """
        Runs a final reward eval with both the final and best model (and potentially larger N,
        as specified in config.final_evaluator)
        """
        trainers_to_eval = {
            "final": self,
            "best": Trainer.load(self.run_name, seed=self.seed, best=True),
        }
        end_of_training_metrics = {}
        for tr_name, tr in trainers_to_eval.items():
            metrics, _ = tr.perform_rew_eval_and_get_metrics(self.final_evaluator, dataset)
            end_of_training_metrics.update({f"{tr_name}_{metric}": v for metric, v in metrics.items()})
        if self.wandb_logging:
            # Save at some large step so that all experiments will have same point on x-axis
            wandb.log(end_of_training_metrics, step=int(1e10))
        # Save final reward evaluation results to JSON
        final_rew_evals_path = os.path.join(self.get_save_dir(), "final_rew_evals")
        save_as_json(end_of_training_metrics, final_rew_evals_path)
        print("End of training metrics", end_of_training_metrics)
