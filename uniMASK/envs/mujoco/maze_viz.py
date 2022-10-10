import random

import numpy as np
from matplotlib import pyplot as plt

from uniMASK.data.datasets.generate_maze2d import make_maze
from uniMASK.envs.d4rl.d4rl_data import MUJOCO_GYM_ENV_NAMES
from uniMASK.envs.d4rl.maze.data import MazeDataset
from uniMASK.envs.evaluator import Evaluator
from uniMASK.trainer import RCAutoEvalIndicator, Trainer
from uniMASK.utils import dict_mean_and_std_err

load_name = "900N_5len_BC_RC_rl_sl0.5_2656a"  # "900N_10len_rnd_BC_rl"
seed = 71709
env_type = "maze2d-medium-v1"
horizon = 200
no_randomness = False
rew_eval_num = 200
target_rewards = [RCAutoEvalIndicator]
max_ep_len = 1000
reward_scale = 1
render = False

random.seed(seed)
np.random.seed(seed)
horizon = 200
env_name = MUJOCO_GYM_ENV_NAMES["medium"]
train_data, _ = MazeDataset.get_datasets(env_name, {"horizon": horizon}, prop=1, leave_for_eval=0)

make_env = lambda: make_maze(env_type, horizon, no_randomness)

evaluator = Evaluator(make_env, rew_eval_num, target_rewards, max_ep_len, reward_scale, render)
trainer = Trainer.load(load_name, seed=seed, best=True, train_evaluator=evaluator)
trainer.rew_batch_params_n.pop(0)

# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)
#
# from uniMASK.envs.d4rl.d4rl_data import MUJOCO_GYM_ENV_NAMES
# from uniMASK.envs.d4rl.maze.data import MazeDataset
#
# # Could potentially have less code reuse if we make MazeDataSet and
# # dataset_info an attribute (the rest is the same), but I think this
# # is clearer.
# train_data, test_data = MazeDataset.get_datasets(
#     env_name=MUJOCO_GYM_ENV_NAMES["medium"],
#     dataset_info={"horizon": 20},
#     prop=0.5,
#     leave_for_eval=5,
# )
#
# validation_losses = trainer.val_loss_evaluation(test_data, 0)

rewards = {}
for x in np.arange(0.0, 1.3, 0.1):
    evaluator.target_factor = x
    results = evaluator.evaluate(train_data)
    rewards[x] = results["RC"][RCAutoEvalIndicator]["ep_rews"]

avged_rews = dict_mean_and_std_err(rewards)
vals = [x[0] for x in list(avged_rews.values())]
plt.scatter(list(avged_rews.keys()), vals)
plt.show()

results, _ = trainer.perform_rew_eval_and_get_metrics(evaluator, dataset=train_data)

evaluator.env_seeds = [int(item) for item in np.random.randint(0, 10e6, size=(rew_eval_num,))]
results, _ = trainer.perform_rew_eval_and_get_metrics(evaluator, dataset=train_data)
print(results)
