#!/bin/zsh
source ~/.zshrc
# install D4RLp
# cd ~/workspace/benchmarks/d4rl && pip install -e .
# install simxarm
# cd ~/workspace/benchmarks/simxarm && pip install -e .
# install unimask
cd ~/workspace && pip install -e .  
# error catching
pip install mujoco==2.3.2
cd ~/

# https://stackoverflow.com/questions/30209776/docker-container-will-automatically-stop-after-docker-run-d
tail -f /dev/null
