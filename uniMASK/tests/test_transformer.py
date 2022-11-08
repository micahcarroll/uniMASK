import os
import random

import pytest

from uniMASK.batches import Batch, BehaviorCloning, CustomPred, DTActionPred, FuturePred, SpanPred, np, torch, tt
from uniMASK.envs.base_data import Dataset
from uniMASK.envs.d4rl.d4rl_data import MUJOCO_GYM_ENV_NAMES
from uniMASK.envs.d4rl.maze.data import MazeDataset
from uniMASK.envs.evaluator import Evaluator
from uniMASK.envs.minigrid.agents import StochGoalAgent
from uniMASK.envs.minigrid.data import filter_data_by, get_sa_transition_counter
from uniMASK.envs.minigrid.env import make_env as make_minigrid_env
from uniMASK.envs.minigrid.inference import get_backward_completion, get_forward_completion
from uniMASK.scripts.configs import batch_code_to_params_n_dict
from uniMASK.scripts.train import run
from uniMASK.sequences import FullTokenSeq
from uniMASK.trainer import BCEvalIndicator, Trainer
from uniMASK.transformer_train import base_model_params, base_training_params, train_from_params
from uniMASK.utils import imdict

# TODO: can speed up a lot of the overfitting tasks by reducing the embed_dim and num_layers etc.


def test_basic_overfitting():
    acts_n = tt(
        np.array(
            [
                [2, 2, 1, 2, 2, 3, 3, 4, 5, 0],
                [2, 2, 1, 2, 2, 3, 3, 4, 5, 0],
                [2, 2, 1, 2, 2, 3, 3, 4, 5, 0],
                [2, 2, 1, 2, 2, 3, 3, 4, 5, 0],
                [2, 2, 1, 2, 2, 3, 3, 4, 5, 0],
                [2, 2, 1, 2, 2, 3, 3, 4, 5, 2],
                [2, 2, 1, 2, 2, 3, 3, 4, 5, 2],
                [2, 2, 1, 2, 2, 3, 3, 4, 5, 2],
                [2, 2, 1, 2, 2, 3, 3, 4, 5, 2],
                [2, 2, 1, 2, 2, 3, 3, 4, 5, 2],
            ]
        )
    )

    seq_len = 10

    acts_n = tt(np.array(acts_n)).unsqueeze(2)
    # acts_n = acts_n.expand((10, 10, 28000))

    dataset = Dataset({"action": acts_n})
    loss_weights = {"action": 1}
    # As rtg is not present in the input, we can just leave the rtg masking type unchanged.
    b_params = imdict(
        {
            "type": FuturePred,
            "span_limits": (10, 10),
            "rtg_masking_type": "Unchanged",
        }
    )
    stacked = True

    tp = base_training_params(
        loss_weights=loss_weights,
        loss_types="l2",
        seq_len=seq_len,
        epochs=20,
        batch_size=100,
        log_interval=3,
        eval_interval=10,
        lr=0.00003,
        stacked=stacked,
        run_name="test_basic_overfitting",
        train_batch_params_n=[b_params],
        seed=0,
    )
    trainer = train_from_params(dataset, dataset, base_model_params(seq_len=seq_len), tp)

    # Evaluation
    full_token_seq = dataset.to_token_seq(input_keys=set(loss_weights), loss_types="l2")
    b = Batch.from_params(full_token_seq, b_params)
    trainer.model(b)
    loss = b.compute_loss_and_acc(loss_weights)["total"]

    # Optimal loss is 1, check that it's close to 1
    assert torch.abs(loss - 1) < 0.05, torch.abs(loss - 1)

    act = b.input_data.get_factor("action")

    # Optimal network ouput should be 1, check that it's close to 1
    delta = torch.abs(act.output[:, -1, :] - 1)
    assert torch.all(delta < 0.15), delta


def test_basic_overfitting_DT():
    states_n = torch.ones((10, 2, 4)) / 8

    acts_n_val = torch.ones((10, 2, 4)) / 2
    acts_n_val[:5, 1] = 4
    acts_n_val[5:, 1] = 2

    rew_n = torch.ones((10, 2, 4)) * 0.1

    seq_len = 2
    dataset_val = Dataset({"action": acts_n_val, "state": states_n, "rtg": rew_n})
    loss_weights = {"action": 1, "state": 0, "rtg": 0}
    # As rtg is not present in the input, we can just leave the rtg masking type unchanged.
    b_params = imdict(
        {
            "type": DTActionPred,
            "rtg_masking_type": "Unchanged",
        }
    )
    stacked = True

    tp = base_training_params(
        loss_weights=loss_weights,
        loss_types="l2",
        seq_len=seq_len,
        epochs=100,
        batch_size=100,
        log_interval=3,
        eval_interval=10,
        lr=0.005,
        stacked=stacked,
        run_name="test",
        train_batch_params_n=[b_params],
        model_class="DT",
        seed=0,
    )
    trainer = train_from_params(
        dataset_val,
        dataset_val,
        base_model_params(seq_len=seq_len, embed_dim=32),
        tp,
    )

    # Evaluation
    full_token_seq = dataset_val.to_token_seq(input_keys=set(loss_weights), loss_types="l2")
    b = Batch.from_params(full_token_seq, b_params)
    trainer.model(b)
    loss = b.compute_loss_and_acc(loss_weights)["total"]

    # Optimal loss is 0.5, check that it's close to 0.5 (because averaging over the 2 timesteps)
    assert torch.abs(loss - 0.5) < 0.05, torch.abs(loss - 0.5)

    act = b.input_data.get_factor("action")

    # Optimal network ouput should be 3, check that it's close to 3
    delta = torch.abs(act.output[:, -1, :] - 3)
    assert torch.all(delta < 0.15), delta


def test_RC_overfitting():
    """Checks that RTG info is seen and acted on"""
    for model_class in ["FB", "DT", "NN"]:
        dataset_size = 100
        half_data_size = dataset_size // 2  # Increase valdiation set size
        horizon = 3

        states_n = torch.ones((dataset_size, horizon, 2)) / 8

        acts_n = torch.ones((dataset_size, horizon, 1))
        acts_n[:half_data_size, 0] = 4
        acts_n[half_data_size:, 0] = 2

        # These the network can't distinguish, because there'll be no clue in the input
        acts_n[:half_data_size, 2] = 3
        acts_n[half_data_size:, 2] = -1
        # There will be 2 possible windows because the horizon is 3
        # Each window will have 2 predictions, for a total of 4.
        # 1/4 of these action predictions will not be possible to make, because there's not enough context
        # The best one can do for that 1/4 case is to predict 1 (between 3 and -1), leading to a squared error of 4
        # The average loss should be 1

        rew_n = torch.ones((dataset_size, horizon, 1)) * 0.5
        rew_n[:half_data_size, 0] = 0.3
        rew_n[half_data_size:, 0] = 0.2

        seq_len = 2
        dataset = Dataset(
            {
                "action": acts_n,
                "state": states_n,
                "rtg": rew_n,
                "timestep": [torch.arange(horizon).unsqueeze(1) for _ in range(dataset_size)],
            }
        )
        dataset = dataset.cat_t_and_rtg(horizon)
        loss_weights = {"action": 1, "state": 0, "rtg": 0}

        loss_types = "l2"

        if model_class == "DT":
            batch_code = "DT_RC"
        else:
            batch_code = "RC"

        batch_params_n = batch_code_to_params_n_dict(None)[batch_code]

        tp = base_training_params(
            loss_weights=loss_weights,
            loss_types=loss_types,
            seq_len=seq_len,
            epochs=200,
            batch_size=200,
            log_interval=200,
            eval_interval=200,
            lr=0.0015,
            run_name="test_RC_overfitting",
            train_batch_params_n=batch_params_n,
            seed=2,
            rtg_cat_t=True,
            model_class=model_class,
        )
        trainer = train_from_params(
            dataset,
            dataset,
            model_params=base_model_params(seq_len=seq_len, nlayers=3, embed_dim=32, nheads=2, feedforward_nhid=32),
            training_params=tp,
        )
        last_loss = trainer.last_eval_metrics[f"RC_total"]
        assert abs(last_loss - 1) < 0.15, last_loss

        # full_token_seq = dataset.get_full_data_batch(
        #     seq_len=seq_len, input_keys=set(loss_weights), loss_types="l2"
        # )
        # Get sequences that correspond to the last possible window
        # last_time_normalized = (horizon - 2) / horizon
        # full_token_seq = full_token_seq[
        #     full_token_seq.get_factor("rtg").input[:, 0, 1] == last_time_normalized, :, :
        # ]
        #
        # b = Batch.from_params(full_token_seq, batch_params_n[0])
        # trainer.model(b)
        # loss = b.compute_loss_and_acc(loss_weights)["total"]
        #
        # act = b.input_data.get_factor("action")

        # delta = torch.abs(act.output[:, -1, :] - 1)
        # assert torch.all(delta < 0.15), delta


def setup_minigrid_data_and_env(seed=0, seq_len=10, num_trajs=10, key_env=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = make_minigrid_env("key" if key_env else "empty")
    dataset = env.sample_dataset(n=num_trajs, horizon=seq_len, env=env, agent=StochGoalAgent(env, 1))
    return dataset, env, seq_len


def test_no_immediate_errors():
    dataset, env, seq_len = setup_minigrid_data_and_env()
    loss_weights = {"state": 1, "action": 1, "rtg": 0}
    loss_types = "sce"
    stacked = True

    for rtg_masking_type in Batch.RTG_MASKING_TYPES:
        batch_params = imdict(
            {
                "type": FuturePred,
                "rtg_masking_type": rtg_masking_type,
            }
        )
        tp = base_training_params(
            loss_weights=loss_weights,
            loss_types=loss_types,
            seq_len=seq_len,
            epochs=10,
            batch_size=100,
            log_interval=10,
            eval_interval=10,
            lr=0.002,
            stacked=stacked,
            run_name="test",
            train_batch_params_n=[batch_params],
        )
        train_from_params(
            dataset,
            dataset,
            model_params=base_model_params(seq_len=seq_len),
            training_params=tp,
        )


def test_overfitting():
    """Test that a transformer is able to overfit to 10 deterministic trajectories"""
    from uniMASK.envs.minigrid.viz import visualize_predictions

    dataset, env, seq_len = setup_minigrid_data_and_env()
    loss_weights = {"state": 1, "action": 1, "rtg": 0}
    loss_types = "sce"
    batch_code = "future"
    batch_params_n = batch_code_to_params_n_dict(None)[batch_code]

    for stacked in [True, False]:
        tp = base_training_params(
            loss_weights=loss_weights,
            loss_types=loss_types,
            seq_len=seq_len,
            epochs=200,
            batch_size=100,
            log_interval=10,
            eval_interval=10,
            lr=0.0015,
            stacked=stacked,
            run_name="test_overfitting" if stacked else "test_overfitting_nonstacked",
            train_batch_params_n=batch_params_n,
            seed=2,
        )
        trainer = train_from_params(
            dataset,
            dataset,
            model_params=base_model_params(seq_len=seq_len),
            training_params=tp,
        )
        last_loss = trainer.last_eval_metrics[f"{batch_code}_total"]
        assert last_loss < 0.30, last_loss

        b = Batch.get_dummy_batch_output(
            dataset.get_full_data_batch(
                seq_len,
                input_keys=set(loss_weights),
                loss_types=loss_types,
                stacked=stacked,
            ),
            batch_params_n[0],
            trainer,
        )

        if stacked:
            post_train_eval_loss = b.compute_loss_and_acc(loss_weights)["all"]["action"].item()
        else:
            post_train_eval_loss = b.compute_loss_and_acc(loss_weights)["action"]["action"].item()

        assert post_train_eval_loss < 0.30, post_train_eval_loss

        # Checking that viz code doesn't explode
        visualize_predictions(env, b, traj_idx=0)


def test_overfitting_key():
    """Test that a transformer is able to overfit to 10 deterministic trajectories"""
    dataset, env, seq_len = setup_minigrid_data_and_env(key_env=True)

    loss_weights = {"state": 1, "state_key_pos": 1, "action": 1, "rtg": 0}
    # Switching up order to make sure this doesn't break things
    eval_loss_weights = {"action": 1, "state": 1, "state_key_pos": 1, "rtg": 0}
    loss_types = "sce"

    batch_code = "future"
    for model_class in ["FB", "NN"]:
        print(model_class)
        training_in_key_env(
            batch_code,
            dataset,
            env,
            eval_loss_weights,
            loss_types,
            loss_weights,
            model_class,
            seq_len,
            run_name="test_overfitting_key",
        )


def test_overfitting_key_dt():
    """Test that a DT model is able to overfit to 10 deterministic trajectories"""
    dataset, env, seq_len = setup_minigrid_data_and_env(key_env=True)

    loss_weights = {"state": 0, "state_key_pos": 0, "action": 1, "rtg": 0}
    eval_loss_weights = {"action": 1, "state": 0, "state_key_pos": 0, "rtg": 0}
    loss_types = "sce"

    batch_code = "DT_BC"
    model_class = "DT"

    training_in_key_env(
        batch_code,
        dataset,
        env,
        eval_loss_weights,
        loss_types,
        loss_weights,
        model_class,
        seq_len,
        run_name="test_DT_overfitting_key",
        eval_metric="BC",
    )


def training_in_key_env(
    batch_code,
    dataset,
    env,
    eval_loss_weights,
    loss_types,
    loss_weights,
    model_class,
    seq_len,
    run_name,
    eval_metric=None,
):
    from uniMASK.envs.minigrid.viz import visualize_predictions

    batch_params_n = batch_code_to_params_n_dict(None)[batch_code]
    stacked = True
    tp = base_training_params(
        loss_weights=loss_weights,
        loss_types=loss_types,
        seq_len=seq_len,
        epochs=200,
        batch_size=100,
        log_interval=10,
        eval_interval=10,
        lr=0.002,
        stacked=stacked,
        run_name=run_name,
        train_batch_params_n=batch_params_n,
        model_class=model_class,
    )
    trainer = train_from_params(dataset, dataset, base_model_params(seq_len=seq_len), tp)
    last_loss = trainer.last_eval_metrics[f"{batch_code if eval_metric is None else eval_metric}_total"]
    assert last_loss < 0.25, last_loss
    b = Batch.get_dummy_batch_output(
        dataset.get_full_data_batch(
            seq_len,
            input_keys=set(eval_loss_weights),
            loss_types=loss_types,
            stacked=stacked,
        ),
        batch_params_n[0],
        trainer,
    )
    post_train_eval_loss = b.compute_loss_and_acc(eval_loss_weights)["all"]["action"].item()
    assert post_train_eval_loss < 0.25, post_train_eval_loss
    # Checking that viz code doesn't explode
    visualize_predictions(env, b, traj_idx=0)
    # Testing that loading can happen successfully
    Trainer.load(run_name)


@pytest.mark.skipif(os.environ.get("CI") is not None, reason="Test dependencies are hard in CI")
def test_finetuning():
    # Create a new, different dataset, and fine tune previous model (overfitting to the new data)
    dataset, env, seq_len = setup_minigrid_data_and_env(seed=1, key_env=True)

    loss_weights = {"state": 1, "state_key_pos": 1, "action": 1, "rtg": 0}
    loss_types = "sce"

    # Finetuning previous model "future" to new data and only on RC this time (with only first RC)
    batch_code = "RC"
    tp = base_training_params(
        loss_weights=loss_weights,
        loss_types=loss_types,
        seq_len=seq_len,
        epochs=200,
        batch_size=100,
        eval_interval=10,
        lr=0.001,
        stacked=True,
        run_name="test_finetuning",
        train_batch_params_n=batch_code_to_params_n_dict(None)[batch_code],
        val_batch_params_n=batch_code_to_params_n_dict(None)[batch_code],
    )

    finetuning_trainer = Trainer.load("test_overfitting_key", **tp)
    finetuning_trainer.train(dataset, dataset)
    last_loss = finetuning_trainer.last_eval_metrics[f"{batch_code}_total"]
    assert last_loss < 0.25, last_loss


def test_custom_batch():
    dataset, env, seq_len = setup_minigrid_data_and_env(key_env=True)
    trainer = Trainer.load("test_overfitting_key")

    b_params_custom = {
        "type": CustomPred,
        "state_masks": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "action_masks": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        "rtg_masks": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    }
    b_params_span = {
        "type": SpanPred,
        "span_limits": (seq_len, seq_len),
        "rtg_masking_type": "RC_fixed",
    }
    compare_outputs(b_params_custom, b_params_span, dataset, seq_len, trainer)

    b_params_custom = {
        "type": CustomPred,
        "state_masks": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "action_masks": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        "rtg_masks": np.zeros(seq_len),
    }
    b_params_span = {
        "type": BehaviorCloning,
        "span_limits": (seq_len, seq_len),
        "rtg_masking_type": "BC",
    }
    compare_outputs(b_params_custom, b_params_span, dataset, seq_len, trainer)


def compare_outputs(b_params0, b_params1, dataset, seq_len, trainer):
    loss_weights = {"state": 1, "action": 1, "state_key_pos": 1, "rtg": 0}
    input_keys = set(loss_weights)
    loss_types = "sce"
    stacked = True

    b = Batch.get_dummy_batch_output(
        dataset.get_full_data_batch(seq_len, input_keys=input_keys, loss_types=loss_types, stacked=stacked),
        batch_params=b_params0,
        trainer=trainer,
    )
    b2 = Batch.get_dummy_batch_output(
        dataset.get_full_data_batch(seq_len, input_keys=input_keys, loss_types=loss_types, stacked=stacked),
        batch_params=b_params1,
        trainer=trainer,
    )
    for k1, v1 in b.input_masks.items():
        for k2, v2 in v1.items():
            a = b2.input_masks[k1][k2]
            assert torch.all(a == v2)


def test_forward_sampling():
    """Testing that forward sampling has not been changed"""
    from uniMASK.envs.minigrid.viz import visualize_completions

    # Making the dataset exactly in the same way as in test_overfitting
    # We want to use the overfit model, and make sure we recover the ground truth data perfectly
    # when forward sampling
    dataset, env, seq_len = setup_minigrid_data_and_env(key_env=True)
    full_data = dataset.to_token_seq()

    start_pos = (1, 1)
    matching_data, _ = filter_data_by(env, full_data, "state", query={0: start_pos})
    assert matching_data.shape[0] == 1, "This is the only way that this overfitting test can work"

    gt_actions = matching_data.get_factor("action").inputs_hr[0]
    gt_states = matching_data.get_factor("state").inputs_hr[0]

    loss_weights = {"state": 0, "state_key_pos": 0, "action": 0, "rtg": 0}
    input_keys = set(loss_weights)
    tr = Trainer.load("test_overfitting_key")

    masks = {
        "state_masks": np.zeros(seq_len),
        "action_masks": np.zeros(seq_len),
        "rtg_masks": np.zeros(seq_len),
        # Reward only used for debugging output (not used by model)
        "reward_masks": np.zeros(seq_len),
    }
    masks["state_masks"][:1] = 1
    sampled_data = get_forward_completion(env, matching_data, masks, tr, input_keys, argmax=True)
    sampled_actions = sampled_data.get_factor("action").inputs_hr[0]
    sampled_states = sampled_data.get_factor("state").inputs_hr[0]
    assert torch.all(gt_actions == sampled_actions), f"GT {gt_actions} vs sampled {sampled_actions}"
    assert torch.all(gt_states == sampled_states), sampled_states

    # Just checking that the viz code doesn't throw an error
    trajs_merged = FullTokenSeq.concatenate([sampled_data])
    sa_transition_counter = get_sa_transition_counter(trajs_merged)
    key_pos = matching_data.get_factor("state_key_pos").inputs_hr[0, 0].item()
    visualize_completions(env, sa_transition_counter, start_pos=start_pos, key_pos=env.IDX_TO_POS[key_pos])


@pytest.mark.skip
def test_backwards_sampling():
    """
    Testing that backwards sampling has not been changed

    NOTE: if this doesn't pass, it could even just be due to a different numpy version etc. so don't worry about
     this test passing too much
    """
    # Making the dataset exactly in the same way as in test_overfitting
    # We want to use the overfit model, and make sure we recover the ground truth data perfectly
    # when forward sampling
    dataset, env, seq_len = setup_minigrid_data_and_env(key_env=True)
    full_data = dataset.to_token_seq()

    end_pos = (4, 4)
    matching_data, _ = filter_data_by(env, full_data, "state", query={seq_len - 1: end_pos})

    loss_weights = {"state": 0, "state_key_pos": 0, "action": 0, "rtg": 0}
    input_keys = set(loss_weights)

    trainer_act = Trainer.load("test_overfitting_key")

    # Can't overfit here because the training batch type cannot both do state and action prediction as would be required here
    # Because of that, we use consistency_checks=False and just remember the output
    masks = {
        "state_masks": np.zeros(seq_len),
        "action_masks": np.zeros(seq_len),
        "rtg_masks": np.zeros(seq_len),
        # Reward only used for debugging output (not used by model)
        "reward_masks": np.zeros(seq_len),
    }
    masks["state_masks"][-1] = 1
    sampled_data = get_backward_completion(
        env,
        matching_data,
        masks,
        trainer_act,
        input_keys,
        argmax=True,
        consistency_checks=False,
    )
    sampled_actions = sampled_data.get_factor("action").inputs_hr[0]
    sampled_states = sampled_data.get_factor("state").inputs_hr[0]
    assert torch.all(tt([3, 1, 3, 0, 0, 0, 0, 1, 1, 3]) == sampled_actions), sampled_actions
    assert torch.all(tt([13, 2, 6, 6, 2, 6, 8, 12, 12, 15]) == sampled_states), sampled_states


def test_goal_conditioned():
    pass
    # You have two different trajs which only differ in the final state, overfit on them
    # and then test that the one-to-last action is perfectly predicted (this should be trivial)
    # there might be a slightly harder test to perform here: two trajs that have identical first N timesteps, and
    # you aim to predict the Nth action, based on the history so far (useless) and the final state (which should tell
    # you exactly where to go)
    # Verify this with works with both an overfit goal-conditioned model, and a rnd model


@pytest.mark.skipif(os.environ.get("CI") is not None, reason="Mujoco data not available in CI")
def test_maze_evals_dont_explode():
    """This test can't be run on CI because it requires having downloaded the mujoco dataset, which is too large to
    commit to the repo.

    Checks that simple runs with various settings don't break everything
    """
    from uniMASK.data.datasets.generate_maze2d import make_maze
    from uniMASK.envs.d4rl.d4rl_data import MUJOCO_GYM_ENV_NAMES

    # We make deterministic dataset of len 49 to not conflict with main dataset
    horizon = 49
    env_name = MUJOCO_GYM_ENV_NAMES["umaze"]
    train_data, test_data = MazeDataset.get_datasets(
        env_name,
        {"horizon": horizon},
        prop=0.0001,
        leave_for_eval=1,
    )
    make_env = lambda: make_maze("maze2d-umaze-v1", horizon, no_rnd=False)
    _test_all_batch_codes(make_env, horizon, seq_len=5, test_data=train_data, train_data=train_data)


def _test_all_batch_codes(make_env, max_ep_len, seq_len, test_data, train_data):
    """Not actually testing anything except for stuff breaking

    TODO: this is actually not currently testing all batch codes? Maybe discontinue this if we already have it
     tested in the dry runs?
    """
    # Doing timestep embedding here! (at least for this test it cuts the training time in half, surprisingly)
    loss_weights = {"state": 1, "action": 1, "timestep": np.nan}
    for rtg_masking_type in Batch.RTG_MASKING_TYPES:
        batch_params = imdict(
            {
                "type": FuturePred,
                "rtg_masking_type": rtg_masking_type,
                "logging_name": "future",
                # This is necessary for rew_batch_params, and is ignored for the training ones
            }
        )

        train_evaluator = Evaluator(
            make_env,
            rew_eval_num=2,
            eval_types=[BCEvalIndicator],
            max_ep_len=max_ep_len,
        )

        tp = base_training_params(
            loss_weights=loss_weights,
            loss_types="l2",
            seq_len=seq_len,
            epochs=2,
            batch_size=20,
            log_interval=2,
            eval_interval=2,
            rew_eval_interval=2,
            lr=0.002,
            stacked=True,
            run_name="test",
            train_batch_params_n=[batch_params],
            rew_batch_params_n=[batch_params],
        )
        train_from_params(
            train_data,
            test_data,
            base_model_params(seq_len=seq_len, timestep_encoding=True),
            tp,
            train_evaluator=train_evaluator,
        )


@pytest.mark.skipif(os.environ.get("CI") is not None, reason="Mujoco data not available in CI")
def test_maze_overfitting():
    """
    Overfit to more than 1 trajectory (with different rewards, but same state) and make sure the loss is 0 and
    evaluation reward perfectly matches across different models on the RC task.

    Checks that the models see the rtg and use that information appropriately.
    """
    from uniMASK.data.datasets.generate_maze2d import make_maze
    from uniMASK.envs.d4rl.d4rl_data import MUJOCO_GYM_ENV_NAMES
    from uniMASK.envs.d4rl.maze.data import MazeDataset

    seed = 3
    random.seed(seed)
    np.random.seed(seed)

    horizon = 9
    env_name = MUJOCO_GYM_ENV_NAMES["umaze"]
    train_data, test_data = MazeDataset.get_datasets(env_name, {"horizon": horizon}, prop=1, leave_for_eval=0)

    eval_horizon = horizon
    train_data = train_data.cat_t_and_rtg(eval_horizon)
    train_data = train_data[:2]

    seq_len = 2
    loss_weights = {"action": 1, "state": 0, "rtg": 0}

    unique_rewards = torch.stack(train_data.rtg)[:, 0, 0].ravel().unique().numpy()
    unique_rewards = [
        float(item) for item in unique_rewards
    ]  # TODO: have whole test for this: only use 1 traj and check auto can overfit + ["RC_auto"]

    make_env = lambda: make_maze("maze2d-umaze-v1", eval_horizon, no_rnd=True)

    evaluator = Evaluator(
        make_env,
        rew_eval_num=2,
        eval_types=unique_rewards,
        max_ep_len=horizon,
        rew_scale=1,
    )

    for model_class in ["FB", "DT", "NN"]:
        if model_class == "DT":
            batch_code = "DT_RC"
        else:
            batch_code = "RC"

        batch_params_n = batch_code_to_params_n_dict(None)[batch_code]
        tp = base_training_params(
            loss_weights=loss_weights,
            loss_types="l2",
            seq_len=seq_len,
            epochs=200,
            batch_size=100,
            log_interval=2,
            eval_interval=100,
            rew_eval_interval=100,
            lr=0.005,
            stacked=True,
            run_name="test",
            train_batch_params_n=batch_params_n,
            rew_batch_params_n=batch_params_n,
            rtg_cat_t=True,
            model_class=model_class,
        )
        trainer = train_from_params(
            train_data,
            train_data,
            base_model_params(seq_len=seq_len, embed_dim=32, nlayers=2, nheads=2),
            tp,
            train_evaluator=evaluator,
        )
        assert trainer.last_eval_metrics["RC_total"] < 0.01

        for rew_target in unique_rewards:
            eval_metrics = trainer.last_eval_metrics["RC"][rew_target]

            assert abs(eval_metrics["eval_avg_rew"] - rew_target) - 0.03, "Should be close to dataset performance"
            assert np.allclose(eval_metrics["eval_se_rew"], 0), "Env and policy should be deterministic"


@pytest.mark.skipif(os.environ.get("CI") is not None, reason="Maze is hard to install in CI")
def test_maze():
    """This test can't be run on CI because it requires having downloaded the mujoco dataset, which is too large to
    commit to the repo.

    TODO: eventually fix by just running this on a small subset of the mujoco data which can be saved
    """
    from uniMASK.envs.d4rl.maze.data import MazeDataset

    seq_len = 5

    horizon = 9
    env_name = MUJOCO_GYM_ENV_NAMES["umaze"]
    train_data, test_data = MazeDataset.get_datasets(env_name, {"horizon": horizon}, prop=0.9, leave_for_eval=1)

    # Doing timestep embedding here! (at least for this test it cuts the training time in half, surprisingly)
    loss_weights = {"state": 1, "action": 1, "timestep": np.nan}
    batch_params = imdict({"type": FuturePred, "rtg_masking_type": "Unchanged"})

    val_batch_params = imdict(
        {
            "type": FuturePred,
            "span_limits": (seq_len, seq_len),
            "rtg_masking_type": "Unchanged",
        }
    )
    tp = base_training_params(
        loss_weights=loss_weights,
        loss_types="l2",
        seq_len=seq_len,
        epochs=20,
        batch_size=100,
        log_interval=10,
        eval_interval=10,
        rew_eval_interval=20,
        lr=0.002,
        stacked=True,
        run_name="test_mujoco_overfitting",
        train_batch_params_n=[batch_params],
        val_batch_params_n=[val_batch_params],
        rew_batch_params_n=[batch_params],
    )
    trainer = train_from_params(
        train_data,
        test_data,
        base_model_params(seq_len=seq_len, timestep_encoding=True),
        tp,
    )
    last_loss = trainer.last_eval_metrics[f"{FuturePred.__name__}_total"]
    assert last_loss < 0.336, last_loss


@pytest.mark.skipif(os.environ.get("CI") is not None, reason="Mujoco is hard to install in CI")
def test_dt_submodules():
    """This tests that we did not regress to this bug https://github.com/micahcarroll/flexiBERT/pull/30.
    It tests that all neural networks are added as modules to the DT.
    The above bug informed us that modules that are not added to the DT will not be saved, and should therefore cause
    loaded model behavior to differ between reloads. This test verifies that reloaded models evaluate the same.
    """
    from uniMASK.data.datasets.generate_maze2d import make_maze
    seed = 0
    seq_len = 5
    rew_eval_num = 2
    max_episode_steps = 1000
    run_name = "test"
    horizon = 9
    env_name = MUJOCO_GYM_ENV_NAMES["umaze"]
    train_data, test_data = MazeDataset.get_datasets(env_name, {"horizon": horizon}, prop=0.9, leave_for_eval=1)
    loss_weights = {"state": 0, "action": 1, "rtg": 0}
    batch_params = batch_code_to_params_n_dict(None)["DT_BC"][0]

    make_env = lambda: make_maze("maze2d-umaze-v1", horizon, no_rnd=True)
    evaluator = Evaluator(
        make_env,
        rew_eval_num=rew_eval_num,
        eval_types=[BCEvalIndicator],
        max_ep_len=max_episode_steps,
        env_seeds=range(rew_eval_num),
    )
    tp = base_training_params(
        loss_weights=loss_weights,
        loss_types="l2",
        seq_len=seq_len,
        epochs=2,
        batch_size=100,
        log_interval=2,
        eval_interval=2,
        rew_eval_interval=2,
        lr=0.002,
        stacked=True,
        run_name="test",
        train_batch_params_n=[batch_params],
        rew_batch_params_n=[batch_params],
        model_class="DT",
        # save_best="rew",
        seed=seed,
    )
    trainer = train_from_params(
        train_data,
        test_data,
        base_model_params(seq_len=seq_len, timestep_encoding=False),
        tp,
        train_evaluator=evaluator,
    )

    trainers = {
        "original": trainer,
        "loaded1": Trainer.load(run_name, evaluator, seed=seed),
        "loaded2": Trainer.load(run_name, evaluator, seed=seed),
    }
    metrics = {}
    unwanted_keys = [
        "rew_eval_time_BCBC",
        "rew_eval_steps_per_sec_avg_BCBC",
        "rew_eval_steps_per_sec_max_BCBC",
    ]
    for tr_name, tr in trainers.items():
        metrics[tr_name], _ = tr.perform_rew_eval_and_get_metrics(evaluator, dataset=None)
        # remove keys that are allowed to be different (e.g. eval time)
        for unwanted_key in unwanted_keys:
            metrics[tr_name].pop(unwanted_key)
    trainer_names = list(metrics.keys())
    for i in range(len(trainer_names) - 1):
        tr_1, tr_2 = trainer_names[i], trainer_names[i + 1]
        assert metrics[tr_1] == metrics[tr_2], f"{tr_1} evaluated differently than {tr_2}!"

@pytest.mark.skipif(os.environ.get("CI") is not None, reason="Mujoco data not available in CI")
def test_dry_runs():
    # TODO: dry run tests for minigrid
    default_args = "maze -ep 2 --num_trajs 1 --rew_eval_freq 1 --rtg_loss 0 -et 3600 --env_spec medium --final_rew_eval_num 0 --embed_dim 8 -K 2 --train_rew_eval_num 2 --nlayers 2 --nheads 2"
    wandb_sweep_args = [
        # "FB,rnd,all,BC_RC,1",
        # "FB,future,all,BC_RC,1",
        # "FB,past,all,BC_RC,1",
        # "FB,BC,all,BC_RC,1",
        # "FB,RC,all,BC_RC,1",
        "FB,all,all,BC_RC,1",
        "DT,DT_BC,DT_all,DT_all,1",
        # "DT,DT_RC,DT_all,DT_all,1",
        "NN,BC,all,BC_RC,1",
        # "NN,RC,all,BC_RC,1"
    ]
    args = [f"{default_args} -sweep_p {sweep_args}" for sweep_args in wandb_sweep_args]
    for a in args:
        run(a.split(" "))


if __name__ == "__main__":
    # test_basic_overfitting()
    # test_overfitting()
    # test_no_immediate_errors()
    # test_finetuning()
    # test_overfitting_key()
    # test_custom_batch()
    # test_forward_sampling()
    # test_backwards_sampling()
    # test_mujoco()
    # test_mujoco_eval()
    # test_maze_overfitting()
    # test_mujoco_evals_dont_explode()
    # test_dataset_formatting()
    # test_dry_runs()
    # test_dt_loading()
    # test_dt_submodules()
    # test_maze_evals_dont_explode()
    pass
