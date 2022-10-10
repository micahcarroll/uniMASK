#!/bin/bash

# Script initialization
BASHRC="$HOME/.bashrc"
MUJOCO_DIR="$HOME/.mujoco"
MUJOCO_EXPORT='export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia'
MUJOCO_EXPORT=${MUJOCO_EXPORT}:${MUJOCO_DIR}/mujoco210/bin

# exit when any command fails
set -e

# Uncomment next block to echo the last command + exit status before exiting.
#trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
#trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

# cd "${0%/*}" # Change to script directory (doesn't work rn)

echo "This script installs uniMASK as well as all necessary dependencies."
echo "NOTE: It is intended for use on a Linux server."
read -p "Press ENTER to proceed. (Ctrl+c to exit)" yn

# Install conda
if command -v conda
then
	echo "Found conda"
else
	echo "Downloading Miniconda3..."
	tmpfile_miniconda=$(mktemp miniconda3.sh.XXXXXX)
	curl https://repo.anaconda.com/miniconda/Miniconda3-py39_4.11.0-Linux-x86_64.sh > $tmpfile_miniconda
	yes | bash $tmpfile_miniconda -b
	rm $tmpfile_miniconda
	echo "Conda installed!"
	echo "Now restart your shell (or run source ~/.bashrc) and run this script again."
	exit 0
	# Should be possible to do this without restarting the shell but I've exhausted my servers trying to figure it out.
fi

# Initialize conda for bash
if [ -z $CONDA_SHLVL ]
then
  echo "Initializing conda..."
  conda init bash
#  $HOME/miniconda3/bin/conda init bash
  source $HOME/.bashrc
fi

# Install mujoco library
# Doing this before installing uniMASK because maybe that saves the export steps?
if [ -d ${MUJOCO_DIR}/mujoco210/ ]
then
	echo "Found mujoco library"
else
	echo "Downloading mujoco..."
	mkdir -p $MUJOCO_DIR
	tmpfile_mujoco=$(mktemp mujoco.tar.gz.XXXXXX)
	curl https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz > $tmpfile_mujoco
	tar --directory $MUJOCO_DIR -xf $tmpfile_mujoco
	rm $tmpfile_mujoco
	cat $BASHRC | grep -qxF "$MUJOCO_EXPORT" || echo $MUJOCO_EXPORT >> $HOME/.bashrc # Add mujoco export statements (if not exist)
	source $HOME/.bashrc # reload .bashrc after adding the export statements
fi

# Create uniMASK env
conda create -y -n uniMASK python=3.7
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate uniMASK
conda install -y pytest
conda install -y -c pytorch pytorch
conda install -y -c conda-forge matplotlib ipywidgets seaborn pickle5 black wandb tqdm gym gym-box2d transformers
conda install -y -c ericmjl isort

# Install uniMASK package
pip list | grep -F uniMASK || pip install -e . # install uniMASK if not already installed


# Install D4RL
if [ -d d4rl/.git/ ]
then
  echo "Found d4rl"
else
  # TODO: add flag / if-else so that it clones https only when run from GitHub Actions, else use ssh.
  git clone https://github.com/rail-berkeley/d4rl.git
  cp d4rl_setup.py d4rl/setup.py # Currently d4rl/setup.py has a bug, so we overwrite it with a fixed version.
fi
pip list | grep -F d4rl || pip install -e d4rl # install d4rl if not already installed


# Download Mujoco datasets
cd uniMASK/data/datasets
python parse_d4rl_datasets.py --download mujoco

echo "uniMASK installation complete! Don't forget to conda activate uniMASK"
