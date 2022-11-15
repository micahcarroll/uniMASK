import argparse
import os
import pickle
from copy import deepcopy

import numpy as np
import seaborn
from matplotlib import pyplot as plt
from scipy.stats import variation
from tqdm import tqdm

from uniMASK.batches import Batch
from uniMASK.data import TEST_DATA_DIR
from uniMASK.envs.base_data import Dataset
from uniMASK.scripts.configs import batch_code_to_params_n_dict
from uniMASK.trainer import Trainer
from uniMASK.utils import set_style

SEQ_LEN = 10
MAX_VALUE = 1000


COLS = dict(
    [
        ("BC", {"display_name": "Behavior\nCloning", "state_loss": 0.1}),
        ("RC", {"display_name": "Reward\nConditioned", "state_loss": 1}),
        ("goal_conditioned", {"display_name": "Goal\nConditioned", "state_loss": 1}),
        ("waypoint", {"display_name": "Waypoint\nConditioned", "state_loss": 1}),
        ("past", {"display_name": "Past\nInference", "state_loss": 0.5}),
        ("future", {"display_name": "Future\nInference", "state_loss": 0.5}),
        ("forwards", {"display_name": "Forwards\nDynamics", "state_loss": 1}),
        ("backwards", {"display_name": "Inverse\nDynamics", "state_loss": 0.5}),
    ]
)

COLS_TO_DT_COLS = {"BC": "DT_BC", "RC": "DT_RC"}

FB_ROWS = dict(
    [
        (
            "all_w_dyna",
            {"display_name": "Multi-task\n(All the above)", "state_loss": 1},
        ),  # "$\\bf{All}$ the\nabove",
        ("rnd", {"display_name": "Random-mask", "state_loss": 1}),
        (
            "ft",
            {"display_name": "Random M.\n+Finetune", "state_loss": 1},
        ),
    ]
)

BASELINE_ROWS = dict(
    [
        ("DT_BC", {"display_name": "(DT) Behavior\nCloning", "state_loss": 0}),
        ("DT_RC", {"display_name": "(DT) Reward\nConditioned", "state_loss": 0}),
        ("NN_BC", {"display_name": "(NN) Behavior\nCloning", "state_loss": 0.5}),
        ("NN_rnd", {"display_name": "(NN) Random\nMasking", "state_loss": 0.5}),
    ]
)

# A design decision is to have a row corresponding to each column followed by remainder FB rows, followed by baselines.
# ROWS = cat_ordered_dicts(COLS, FB_ROWS, BASELINE_ROWS)
ROWS = deepcopy(COLS)
ROWS.update(FB_ROWS)
ROWS.update(BASELINE_ROWS)

ROWS_IN_ORDER = [
    "BC",
    "RC",
    "goal_conditioned",
    "waypoint",
    "past",
    "future",
    "forwards",
    "backwards",
    "all_w_dyna",
    "rnd",
    "ft",
    "NN_BC",
    "NN_rnd",
    "DT_BC",
    "DT_RC",
]

# assert set(ROWS_IN_ORDER) == set(ROWS.keys())

BC_TO_PARAMS_N = batch_code_to_params_n_dict({"seq_len": SEQ_LEN})


DATA_NAME = f"2000_keyenv_{SEQ_LEN}len"
TEST_DATA_PATH = os.path.join(TEST_DATA_DIR, DATA_NAME)

def get_run_name(row, num_trajs, finetune_col=None):
    if row == "ft":
        assert finetune_col is not None
        row = "rnd"
    state_loss = ROWS[row]["state_loss"]
    sl_str = "" if state_loss == 1 else f"_sl{state_loss}"
    train_task = row
    if "NN" in train_task:
        train_task = train_task[3:]
    run_name = f"{num_trajs}N_10len_{train_task}_rl{sl_str}"
    if finetune_col:
        run_name += f"_finetune_{finetune_col}"
    if "NN" in row:
        run_name += "_NN"
    if "DT" in row:
        run_name += "_DT"
    return run_name

def load_trainer_if_found(row, num_trajs, seed, finetune_col=None):
    trainer = None
    run_name = get_run_name(row, num_trajs, finetune_col)
    try:
        trainer = Trainer.load(run_name, best=True, seed=seed)
    except FileNotFoundError as e:
        print(f"{run_name} with seed {seed} not found.")
    if trainer is not None:
        trainer.model.eval()
    return trainer

def generate_row(row, num_trajs, test_data, seeds, test):
    """

    @param test:
    @param seeds:
    @param num_trajs:
    @param row:
    @param test_data:
    @return: np.array of shape (len(COLS))
    """
    # We want the same loss_weights for all rows and columns, and we chose to set them as follows.
    loss_weights = {
        "state": 1,
        "state_key_pos": 1,
        # "state_key": state_loss,
        "action": 1,
        "rtg": 0,
        # "timestep": np.nan,
    }
    row_data_per_seed = []
    missing_data = []
    for seed in tqdm(seeds, desc=" seed", position=2, leave=False):
        row_data = np.empty(shape=(len(COLS)))
        row_data[:] = np.nan  # nan because the heatmap will render this as a blank.
        # "ft" needs a  different model being loaded for each column, but others don't.
        run_name, trainer = None, None
        if row != "ft":
            trainer = load_trainer_if_found(row, num_trajs, seed)
            if trainer is None:
                missing_data.append(f"{get_run_name(row, num_trajs)} seed={seed}")
                continue
        for x, col in tqdm(enumerate(COLS), desc=" cols", position=3, leave=False, total=len(COLS)):
            # DT can only be evaluated on some cols. On the others, we'll continue (leaving the value as np.nan).
            if "DT" in row:
                if col in COLS_TO_DT_COLS:
                    # Basically maps BC->DT_BC, RC->DT_RC. But more correct.
                    col = COLS_TO_DT_COLS[col]
                else:
                    continue
            if row == "ft":
                trainer = load_trainer_if_found(row, num_trajs, seed, col)
                if trainer is None:
                    missing_data.append(f"{get_run_name(row, num_trajs, col)} seed={seed}")
                    continue
            # confirmed b.input_data.get_factor('state').input[0] same across different runs (for first two iters)
            b = Batch.get_dummy_batch_output(
                test_data.get_rnd_batch(
                    batch_size=10 if test else 5000,
                    seq_len=SEQ_LEN,
                    input_keys=set(loss_weights),
                    loss_types="sce",
                    stacked=True,
                ),
                batch_params=BC_TO_PARAMS_N[col][0],
                trainer=trainer,
            )
            row_data[x] = b.compute_loss_and_acc(loss_weights)["total"].item()
        row_data_per_seed.append(row_data)
    if missing_data:
        raise FileNotFoundError(missing_data)
    row_data_per_seed = np.vstack(row_data_per_seed)
    assert row_data_per_seed.shape == (len(seeds), len(COLS))
    return (
        row_data_per_seed.mean(0),
        row_data_per_seed.std(0),
        variation(row_data_per_seed, 0),
    )


def make_heatmaps(mean_data, std_data, num_trajs, output_format):
    mean_data = np.vstack([mean_data[row] for row in ROWS_IN_ORDER])
    std_data = np.vstack([std_data[row] for row in ROWS_IN_ORDER])

    make_heatmap(mean_data, f"{num_trajs}_heatmap", output_format, float_format=".3g")
    make_heatmap(std_data, f"{num_trajs}_heatmap_std", output_format, float_format=".3g")

    # Make normalized heatmaps
    nanmins = np.nanmin(mean_data, axis=0)
    mean_data /= nanmins
    std_data /= nanmins
    make_heatmap(
        mean_data,
        f"{num_trajs}_heatmap_normalized",
        output_format,
        float_format=".4g",
    )
    make_heatmap(
        std_data,
        f"{num_trajs}_heatmap_std_normalized",
        output_format,
        float_format=".4g",
    )


def make_heatmap(data, output_name, output_format, float_format):
    data = np.clip(data, a_min=None, a_max=MAX_VALUE)

    # TODO COLS is assumed to be in order, so should still used OrderedDict...
    x_labels = [col["display_name"] for col in COLS.values()]
    y_labels = [ROWS[row]["display_name"] for row in ROWS_IN_ORDER]
    plt.figure(figsize=(len(ROWS), len(COLS)))
    s = seaborn.heatmap(
        data,
        xticklabels=x_labels,
        yticklabels=y_labels,
        annot=True,
        fmt=float_format,
        vmax=1.5,
        cmap="viridis_r",
        cbar_kws={"pad": 0.02},
    )
    s.set_xticklabels(x_labels, size=11)
    s.set_yticklabels(y_labels, size=10)
    s.axhline(ROWS_IN_ORDER.index("ft") + 1, color="1")

    plt.xlabel("Evaluation task", fontsize=25)
    plt.xticks(rotation=0)
    plt.ylabel("Training task", fontsize=25)

    plt.savefig(
        f"{output_name}.{output_format}",
        format=output_format,
        dpi=500,
        bbox_inches="tight",
    )
    # plt.show()


def generate_data(num_trajs, test_data, seeds, test=False):
    """

    @param test:
    @param seeds:
    @param num_trajs:
    @param test_data:
    @return: dict {row: row_data}
    """
    mean_data = {}
    std_data = {}
    variation_data = {}
    missing_data = []
    for y, row in tqdm(enumerate(ROWS), desc=" rows", position=1, leave=False, total=len(ROWS)):
        mean_data[row], std_data[row], variation_data[row] = None, None, None
        try:
            mean_data[row], std_data[row], variation_data[row] = generate_row(row, num_trajs, test_data, seeds, test)
        except FileNotFoundError as e:
            missing_data.append(e.args[0])
            continue
        assert len(mean_data[row]) == len(COLS)
    if missing_data:
        raise FileNotFoundError(missing_data)
    assert len(mean_data) == len(ROWS)
    return mean_data, std_data, variation_data


def main(args):
    np.random.seed(1)

    if args.no_benchmarks:
        global ROWS_IN_ORDER, ROWS
        ROWS_IN_ORDER = [row_name for row_name in ROWS_IN_ORDER if row_name not in BASELINE_ROWS.keys()]
        ROWS = {k: v for k, v in ROWS.items() if k not in BASELINE_ROWS.keys()}

    dataset = Dataset.load(TEST_DATA_PATH)
    _, test_data = dataset.split_data(train_prop=0.5, num_val_trajs=1000)

    max_variations = {}
    for num_trajs in tqdm(args.num_trajs, desc=" num trajs", position=0, leave=False):
        if args.load:
            with open(f"heatmap_{num_trajs}_data.pkl", "rb") as handle:
                mean_data, std_data, variation_data = pickle.load(handle)
        else:
            mean_data, std_data, variation_data = generate_data(num_trajs, test_data, range(args.num_seeds), args.test)
            if args.save:
                with open(f"heatmap_{num_trajs}_data.pkl", "wb") as handle:
                    pickle.dump((mean_data, std_data, variation_data), handle, protocol=pickle.HIGHEST_PROTOCOL)

        set_style(font_scale=2, tex=False)
        make_heatmaps(mean_data, std_data, num_trajs, output_format="eps" if args.eps else "png")
        max_variations[num_trajs] = np.nanmax(list(variation_data.values()))
    print(f"Max variation(s): {max_variations}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    data_group = parser.add_mutually_exclusive_group(required=False)
    data_group.add_argument(
        "--load",
        default=False,
        action="store_true",
        help="Load data instead of recreating it.",
    )
    data_group.add_argument(
        "--save",
        default=False,
        action="store_true",
        help="Save the evaluated data.",
    )
    parser.add_argument(
        "--num_trajs",
        default=[500],
        nargs="+",
        help="num_trajs for which heatmaps will be made",
        type=int,
    )
    parser.add_argument(
        "--num_seeds",
        default=3,
        type=int,
        help="Rows will average out on range(num_seeds) seeds.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Gives out garbage results, but very quickly",
    )
    parser.add_argument(
        "--eps",
        action="store_true",
        help="Save as a high definition .eps",
    )
    parser.add_argument(
        "--no_benchmarks",
        action="store_true",
        help="Omit benchmark rows (DT / NN).",
    )
    # parser.add_argument(
    #     "--err",
    #     action="store_true",
    #     help="Report std error",
    # )

    args = parser.parse_args()

    main(args)
