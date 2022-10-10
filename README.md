# Introduction 

uniMASK is a generalization of BERT models with flexible abstractions for performing inference on subportions of 
sequences. Masking and prediction can occur both on the token level (as in traditional transformer), or even on 
subportions of tokens.

File structure:
- `batches.py`: has all data pipeline processing classes (`FactorSeq, TokenSeq, FullTokenSeq, Batch, SubBatch`)
- `transformer.py`: contains the transformer model class itself
- `transformer_train.py`: interface and config setting for training a transformer, through `Trainer` class
- `transformer_eval.py`: interface for getting predictions from transformer (currently empty)

# Getting Started

###	Installation process
For installing, use the following commands:

```bash
conda create -n uniMASK python=3.7

conda activate uniMASK

conda install -y pytest
conda install -y -c pytorch pytorch
conda install -y -c conda-forge matplotlib ipywidgets seaborn pickle5 black wandb tqdm gym gym-box2d transformers
conda install -y -c ericmjl isort

pip install -e .
```

You should also [install Mujoco as detailed here](https://github.com/openai/mujoco-py). You can also not install it and ignore the failing Mujoco tests. To have the Mujoco tests to work, you need to download the Mujoco data with the script in from the `decision-transformer` repo, `download_d4rl_dataset.py`. You can then copy paste it into the `data/datasets` folder of uniMASK.

# Build and Test

To verify one's local installation, run `pytest` from the root directory.