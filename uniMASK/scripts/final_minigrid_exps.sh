# parallel -k --lb -j 6 -a final_minigrid_exps.sh --delay 10
# NOTE: Were all run with dropout on (except finetunings)!! Currently, re-running doesn't include dropout.
#  Should re-run everything without dropout (as it makes comparisons harder)? Or with dropout. Not sure. In any case, worth re-running everything at once.
# NOTE: should also change s_loss and a_loss to be 1. This shouldn't change any of the results (as long as the LR is re-scaled to be 5 times less)
# TODO: more hyperparam search for baselines
# TODO: more fine-tuning hyperparam search
# TODO: add minigrid rewards / success-rates


###############
# Medium data #
###############
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 10000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 10000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 10000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc future -vbc all_w_dyna -K 10 -ep 2000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc future -vbc all_w_dyna -K 10 -ep 2000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc future -vbc all_w_dyna -K 10 -ep 2000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc past -vbc all_w_dyna -K 10 -ep 2000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc past -vbc all_w_dyna -K 10 -ep 2000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc past -vbc all_w_dyna -K 10 -ep 2000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc forwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc forwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc forwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc backwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc backwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc backwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc waypoint -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc waypoint -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc waypoint -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc all_w_dyna -vbc all_w_dyna -K 10 -ep 10000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc all_w_dyna -vbc all_w_dyna -K 10 -ep 10000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc all_w_dyna -vbc all_w_dyna -K 10 -ep 10000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2 #

# DT
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_BC -vbc DT_all -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 --model_class DT -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 0
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_BC -vbc DT_all -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 --model_class DT -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 1
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_BC -vbc DT_all -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 --model_class DT -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_RC -vbc DT_all -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 --model_class DT -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 0
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_RC -vbc DT_all -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 --model_class DT -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 1
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_RC -vbc DT_all -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 --model_class DT -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 2

# Finetuning
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc future -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.25 --finetune 500N_10len_rnd_rl_sl0.2_al0.2_seed0 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc future -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.25 --finetune 500N_10len_rnd_rl_sl0.2_al0.2_seed1 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc future -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.25 --finetune 500N_10len_rnd_rl_sl0.2_al0.2_seed2 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc past -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.25 --finetune 500N_10len_rnd_rl_sl0.2_al0.2_seed0 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc past -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.25 --finetune 500N_10len_rnd_rl_sl0.2_al0.2_seed1 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc past -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.25 --finetune 500N_10len_rnd_rl_sl0.2_al0.2_seed2 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.25 --finetune 500N_10len_rnd_rl_sl0.2_al0.2_seed0 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.25 --finetune 500N_10len_rnd_rl_sl0.2_al0.2_seed1 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.25 --finetune 500N_10len_rnd_rl_sl0.2_al0.2_seed2 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.25 --finetune 500N_10len_rnd_rl_sl0.2_al0.2_seed0 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.25 --finetune 500N_10len_rnd_rl_sl0.2_al0.2_seed1 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.25 --finetune 500N_10len_rnd_rl_sl0.2_al0.2_seed2 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc forwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 --finetune 500N_10len_rnd_rl_sl0.2_al0.2_seed0 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc forwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 --finetune 500N_10len_rnd_rl_sl0.2_al0.2_seed1 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc forwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 --finetune 500N_10len_rnd_rl_sl0.2_al0.2_seed2 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc backwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 --finetune 500N_10len_rnd_rl_sl0.2_al0.2_seed0 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc backwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 --finetune 500N_10len_rnd_rl_sl0.2_al0.2_seed1 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc backwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 --finetune 500N_10len_rnd_rl_sl0.2_al0.2_seed2 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 --finetune 500N_10len_rnd_rl_sl0.2_al0.2_seed0 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 --finetune 500N_10len_rnd_rl_sl0.2_al0.2_seed1 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 --finetune 500N_10len_rnd_rl_sl0.2_al0.2_seed2 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc waypoint -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 --finetune 500N_10len_rnd_rl_sl0.2_al0.2_seed0 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc waypoint -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 --finetune 500N_10len_rnd_rl_sl0.2_al0.2_seed1 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc waypoint -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 --finetune 500N_10len_rnd_rl_sl0.2_al0.2_seed2 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready

# NN
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 10000 --lr 1e-4 -data_p 0.25 --model_class NN -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 0 0.1 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 10000 --lr 1e-4 -data_p 0.25 --model_class NN -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 1 0.1 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 10000 --lr 1e-4 -data_p 0.25 --model_class NN -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 2 0.1 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.25 --model_class NN -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 0 0.1 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.25 --model_class NN -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 1 0.1 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.25 --model_class NN -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 2 0.1 #

#############
# High data #
#############
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 10000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 10000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 10000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc future -vbc all_w_dyna -K 10 -ep 6000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc future -vbc all_w_dyna -K 10 -ep 6000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc future -vbc all_w_dyna -K 10 -ep 6000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc past -vbc all_w_dyna -K 10 -ep 2000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc past -vbc all_w_dyna -K 10 -ep 2000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc past -vbc all_w_dyna -K 10 -ep 2000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc forwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc forwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc forwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc backwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc backwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc backwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc waypoint -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc waypoint -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc waypoint -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc all_w_dyna -vbc all_w_dyna -K 10 -ep 10000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc all_w_dyna -vbc all_w_dyna -K 10 -ep 10000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc all_w_dyna -vbc all_w_dyna -K 10 -ep 10000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2 #

# DT
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_BC -vbc DT_all -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 --model_class DT -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 0 0.1 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_BC -vbc DT_all -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 --model_class DT -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 1 0.1 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_BC -vbc DT_all -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 --model_class DT -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 2 0.1 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_RC -vbc DT_all -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 --model_class DT -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 0 0.1 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_RC -vbc DT_all -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 --model_class DT -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 1 0.1 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_RC -vbc DT_all -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 --model_class DT -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 2 0.1 #

# Finetuning
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc future -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.5 --finetune 1000N_10len_rnd_rl_sl0.2_al0.2_seed0 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc future -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.5 --finetune 1000N_10len_rnd_rl_sl0.2_al0.2_seed1 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc future -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.5 --finetune 1000N_10len_rnd_rl_sl0.2_al0.2_seed2 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc past -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.5 --finetune 1000N_10len_rnd_rl_sl0.2_al0.2_seed0 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc past -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.5 --finetune 1000N_10len_rnd_rl_sl0.2_al0.2_seed1 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc past -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.5 --finetune 1000N_10len_rnd_rl_sl0.2_al0.2_seed2 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.5 --finetune 1000N_10len_rnd_rl_sl0.2_al0.2_seed0 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.5 --finetune 1000N_10len_rnd_rl_sl0.2_al0.2_seed1 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.5 --finetune 1000N_10len_rnd_rl_sl0.2_al0.2_seed2 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.5 --finetune 1000N_10len_rnd_rl_sl0.2_al0.2_seed0 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.5 --finetune 1000N_10len_rnd_rl_sl0.2_al0.2_seed1 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.5 --finetune 1000N_10len_rnd_rl_sl0.2_al0.2_seed2 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc forwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 --finetune 1000N_10len_rnd_rl_sl0.2_al0.2_seed0 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc forwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 --finetune 1000N_10len_rnd_rl_sl0.2_al0.2_seed1 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc forwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 --finetune 1000N_10len_rnd_rl_sl0.2_al0.2_seed2 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc backwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 --finetune 1000N_10len_rnd_rl_sl0.2_al0.2_seed0 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc backwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 --finetune 1000N_10len_rnd_rl_sl0.2_al0.2_seed1 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc backwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 --finetune 1000N_10len_rnd_rl_sl0.2_al0.2_seed2 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 --finetune 1000N_10len_rnd_rl_sl0.2_al0.2_seed0 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 --finetune 1000N_10len_rnd_rl_sl0.2_al0.2_seed1 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 --finetune 1000N_10len_rnd_rl_sl0.2_al0.2_seed2 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc waypoint -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 --finetune 1000N_10len_rnd_rl_sl0.2_al0.2_seed0 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc waypoint -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 --finetune 1000N_10len_rnd_rl_sl0.2_al0.2_seed1 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc waypoint -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 --finetune 1000N_10len_rnd_rl_sl0.2_al0.2_seed2 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready

# NN
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 10000 --lr 5e-4 -data_p 0.5 --model_class NN -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 0 0.1 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 10000 --lr 5e-4 -data_p 0.5 --model_class NN -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 1 0.1 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 10000 --lr 5e-4 -data_p 0.5 --model_class NN -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 2 0.1 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 3000 --lr 5e-4 -data_p 0.5 --model_class NN -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 0 0.1 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 3000 --lr 5e-4 -data_p 0.5 --model_class NN -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 1 0.1 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 3000 --lr 5e-4 -data_p 0.5 --model_class NN -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 2 0.1 #

##############
# V low data #
##############
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 6000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 6000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 6000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc future -vbc all_w_dyna -K 10 -ep 2000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc future -vbc all_w_dyna -K 10 -ep 2000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc future -vbc all_w_dyna -K 10 -ep 2000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc past -vbc all_w_dyna -K 10 -ep 2000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc past -vbc all_w_dyna -K 10 -ep 2000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc past -vbc all_w_dyna -K 10 -ep 2000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc forwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc forwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc forwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc backwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc backwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc backwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc waypoint -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc waypoint -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc waypoint -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc all_w_dyna -vbc all_w_dyna -K 10 -ep 6000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 0
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc all_w_dyna -vbc all_w_dyna -K 10 -ep 6000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 1
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc all_w_dyna -vbc all_w_dyna -K 10 -ep 6000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.2 -a_loss 0.2 -wp minigrid_camera_ready -s 2

# DT
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_BC -vbc DT_all -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 --model_class DT -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 0 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_BC -vbc DT_all -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 --model_class DT -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 1 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_BC -vbc DT_all -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 --model_class DT -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 2 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_RC -vbc DT_all -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 --model_class DT -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 0 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_RC -vbc DT_all -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 --model_class DT -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 1 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_RC -vbc DT_all -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 --model_class DT -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 2 #

# Finetuning
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc future -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.025 --finetune 50N_10len_rnd_rl_sl0.2_al0.2_seed0 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc future -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.025 --finetune 50N_10len_rnd_rl_sl0.2_al0.2_seed1 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc future -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.025 --finetune 50N_10len_rnd_rl_sl0.2_al0.2_seed2 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc past -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.025 --finetune 50N_10len_rnd_rl_sl0.2_al0.2_seed0 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc past -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.025 --finetune 50N_10len_rnd_rl_sl0.2_al0.2_seed1 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc past -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.025 --finetune 50N_10len_rnd_rl_sl0.2_al0.2_seed2 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.025 --finetune 50N_10len_rnd_rl_sl0.2_al0.2_seed0 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.025 --finetune 50N_10len_rnd_rl_sl0.2_al0.2_seed1 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.025 --finetune 50N_10len_rnd_rl_sl0.2_al0.2_seed2 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.025 --finetune 50N_10len_rnd_rl_sl0.2_al0.2_seed0 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.025 --finetune 50N_10len_rnd_rl_sl0.2_al0.2_seed1 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 500 --lr 5e-5 -data_p 0.025 --finetune 50N_10len_rnd_rl_sl0.2_al0.2_seed2 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc forwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 --finetune 50N_10len_rnd_rl_sl0.2_al0.2_seed0 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc forwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 --finetune 50N_10len_rnd_rl_sl0.2_al0.2_seed1 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc forwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 --finetune 50N_10len_rnd_rl_sl0.2_al0.2_seed2 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc backwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 --finetune 50N_10len_rnd_rl_sl0.2_al0.2_seed0 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc backwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 --finetune 50N_10len_rnd_rl_sl0.2_al0.2_seed1 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc backwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 --finetune 50N_10len_rnd_rl_sl0.2_al0.2_seed2 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 --finetune 50N_10len_rnd_rl_sl0.2_al0.2_seed0 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 --finetune 50N_10len_rnd_rl_sl0.2_al0.2_seed1 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 --finetune 50N_10len_rnd_rl_sl0.2_al0.2_seed2 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc waypoint -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 --finetune 50N_10len_rnd_rl_sl0.2_al0.2_seed0 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc waypoint -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 --finetune 50N_10len_rnd_rl_sl0.2_al0.2_seed1 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc waypoint -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 --finetune 50N_10len_rnd_rl_sl0.2_al0.2_seed2 -sb loss -a_loss 0.2 -s_loss 0.2 -wp minigrid_camera_ready

# NN
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 6000 --lr 1e-4 -data_p 0.025 --model_class NN -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 0 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 6000 --lr 1e-4 -data_p 0.025 --model_class NN -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 1 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 6000 --lr 1e-4 -data_p 0.025 --model_class NN -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 2 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 --model_class NN -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 0 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 --model_class NN -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 1 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.025 --model_class NN -s_loss 0.2 -sb loss -a_loss 0.2 -wp minigrid_camera_ready -s 2 #