# A very minimal version of final_minigrid_exps.sh, used to test that minigrid-related changes didn't break Jessy'lsb_release -a
# benchmark runs.

# parallel -k --lb -j 6 -a lite_final_minigrid_exps.sh --delay 10
# NOTE: Were all run with dropout on (except finetunings)!! Currently, re-running doesn't include dropout.
#  Should re-run everything without dropout (as it makes comparisons harder)? Or with dropout. Not sure. In any case, worth re-running everything at once.
# NOTE: should also change s_loss and a_loss to be 1. This shouldn't change any of the results (as long as the LR is re-scaled to be 5 times less)
# TODO: more hyperparam search for baselines
# TODO: more fine-tuning hyperparam search
# TODO: add minigrid rewards / success-rates


###############
# Medium data #
###############
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 10 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -wt LITE -s 0 #

# DT
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_BC -vbc DT_all -K 10 -ep 10 --lr 1e-4 -data_p 0.25 --model_class DT -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -wt LITE -s 0

# NN
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 10 --lr 1e-4 -data_p 0.25 --model_class NN -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -wt LITE -s 0 #

#############
# High data #
#############
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 10 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -wt LITE -s 0 #

# DT
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_BC -vbc DT_all -K 10 -ep 10 --lr 1e-4 -data_p 0.5 --model_class DT -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -wt LITE -s 0 #
#
## NN
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 10 --lr 5e-4 -data_p 0.5 --model_class NN -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -wt LITE -s 0 #

##############
# V low data #
##############
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 10 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -wt LITE -s 0
#
## DT
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_BC -vbc DT_all -K 10 -ep 10 --lr 1e-4 -data_p 0.025 --model_class DT -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -wt LITE -s 0 #
#
## NN
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 10 --lr 1e-4 -data_p 0.025 --model_class NN -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -wt LITE -s 0 #