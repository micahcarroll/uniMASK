# Introduction 

uniMASK is a generalization of BERT models with flexible abstractions for performing inference on subportions of 
sequences. Masking and prediction can occur both on the token level (as in traditional transformer), or even on 
subportions of tokens.

File structure:
- `data/`: where rollouts (`datasets`) and trained models (`transformer_runs`) are stored. 
- `envs/`: data-handling and evaluation for each supported environment. Currently
- `scripts/`: 
- `base.py`:
- `batches.py`: has all data pipeline processing classes (`FactorSeq, TokenSeq, FullTokenSeq, Batch, SubBatch`)
- `distributions.py`
- `sequences.py`:
- `trainer.py`:
- `transformer.py`: contains the transformer model class itself
- `transformer_train.py`: interface and config setting for training a transformer, through `Trainer` class
- `utils.py`
- `transformer_eval.py`: interface for getting predictions from transformer (currently empty)
- 

# Getting Started

###	Installation
To install uniMASK, run:

```bash
conda create -n uniMASK python=3.7

conda activate uniMASK

pip install -e .
```

uniMASK requires [D4RL](https://github.com/Farama-Foundation/D4RL), a library for offline reinforcement environments.
You may install as detailed [in the D4RL repo](https://github.com/Farama-Foundation/D4RL#setup), e.g., by running:
```bash
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
```

For CUDA support, you may need to reinstall `pytorch` in CUDA mode:
`pip install torch --extra-index-url https://download.pytorch.org/whl/cu116`

To verify that the installation was successful, run `pytest`.