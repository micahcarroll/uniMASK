#!/bin/bash
cd "$(dirname "$0")" # Change to current directory
python ../envs/minigrid/heatmaps.py --num_trajs 500 --num_seeds 6