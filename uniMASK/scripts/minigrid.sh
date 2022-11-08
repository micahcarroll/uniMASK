# parallel -k --lb -j 25 -a minigrid.sh --delay 10

# hyperparams inferred from https://wandb.ai/fbert/minigrid_sweeps/sweeps/38rgzey3?workspace=user-orrp


###############
# Medium data #
###############
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 10000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --embed_dim 128 --batch_size 100 --feedforward_nhid 128 --nlayers 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 10000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --embed_dim 128 --batch_size 100 --feedforward_nhid 128 --nlayers 2  #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 10000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --embed_dim 128 --batch_size 100 --feedforward_nhid 128 --nlayers 2  #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc future -vbc all_w_dyna -K 10 -ep 2000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --nlayers 2 --nheads 4 --feedforward_nhid 32
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc future -vbc all_w_dyna -K 10 -ep 2000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --nlayers 2 --nheads 4 --feedforward_nhid 32
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc future -vbc all_w_dyna -K 10 -ep 2000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --nlayers 2 --nheads 4 --feedforward_nhid 32
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc past -vbc all_w_dyna -K 10 -ep 2000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --embed_dim 32 --nlayers 4 --nheads 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc past -vbc all_w_dyna -K 10 -ep 2000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --embed_dim 32 --nlayers 4 --nheads 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc past -vbc all_w_dyna -K 10 -ep 2000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --embed_dim 32 --nlayers 4 --nheads 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --embed_dim 32 --feedforward_nhid 128 --nheads 4 --nlayers 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --embed_dim 32 --feedforward_nhid 128 --nheads 4 --nlayers 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --embed_dim 32 --feedforward_nhid 128 --nheads 4 --nlayers 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --embed_dim 32 --batch_size 50 --nheads 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --embed_dim 32 --batch_size 50 --nheads 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --embed_dim 32 --batch_size 50 --nheads 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc forwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc forwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc forwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc backwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --batch_size 50
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc backwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --batch_size 50
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc backwards -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --batch_size 50
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc waypoint -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --feedforward_nhid 128
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc waypoint -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --feedforward_nhid 128
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc waypoint -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --feedforward_nhid 128
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc all_w_dyna -vbc all_w_dyna -K 10 -ep 10000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --embed_dim 32 --nheads 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc all_w_dyna -vbc all_w_dyna -K 10 -ep 10000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --embed_dim 32 --nheads 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc all_w_dyna -vbc all_w_dyna -K 10 -ep 10000 --lr 1e-4 -data_p 0.25 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --embed_dim 32 --nheads 4

# DT
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_BC -vbc DT_all -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 --model_class DT -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --embed_dim 32 --feedforward_nhid 128
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_BC -vbc DT_all -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 --model_class DT -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --embed_dim 32 --feedforward_nhid 128
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_BC -vbc DT_all -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 --model_class DT -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --embed_dim 32 --feedforward_nhid 128
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_RC -vbc DT_all -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 --model_class DT -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --embed_dim 32 --feedforward_nhid 128
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_RC -vbc DT_all -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 --model_class DT -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --embed_dim 32 --feedforward_nhid 128
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_RC -vbc DT_all -K 10 -ep 1500 --lr 1e-4 -data_p 0.25 --model_class DT -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --embed_dim 32 --feedforward_nhid 128

# NN
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 10000 --lr 1e-4 -data_p 0.25 --model_class NN -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --embed_dim 32 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 10000 --lr 1e-4 -data_p 0.25 --model_class NN -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --embed_dim 32 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 10000 --lr 1e-4 -data_p 0.25 --model_class NN -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --embed_dim 32 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.25 --model_class NN -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --embed_dim 32 --batch_size 100 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.25 --model_class NN -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --embed_dim 32 --batch_size 100 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 3000 --lr 1e-4 -data_p 0.25 --model_class NN -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --embed_dim 32 --batch_size 100 #

##############
## High data #
##############
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 5000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --embed_dim 128 --batch_size 100 --feedforward_nhid 128 --nlayers 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 5000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --embed_dim 128 --batch_size 100 --feedforward_nhid 128 --nlayers 2  #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 5000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --embed_dim 128 --batch_size 100 --feedforward_nhid 128 --nlayers 2  #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc future -vbc all_w_dyna -K 10 -ep 1000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --nlayers 2 --nheads 4 --feedforward_nhid 32
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc future -vbc all_w_dyna -K 10 -ep 1000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --nlayers 2 --nheads 4 --feedforward_nhid 32
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc future -vbc all_w_dyna -K 10 -ep 1000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --nlayers 2 --nheads 4 --feedforward_nhid 32
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc past -vbc all_w_dyna -K 10 -ep 1000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --embed_dim 32 --nlayers 4 --nheads 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc past -vbc all_w_dyna -K 10 -ep 1000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --embed_dim 32 --nlayers 4 --nheads 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc past -vbc all_w_dyna -K 10 -ep 1000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --embed_dim 32 --nlayers 4 --nheads 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 750 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --embed_dim 32 --feedforward_nhid 128 --nheads 4 --nlayers 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 750 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --embed_dim 32 --feedforward_nhid 128 --nheads 4 --nlayers 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 750 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --embed_dim 32 --feedforward_nhid 128 --nheads 4 --nlayers 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 750 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --embed_dim 32 --batch_size 50 --nheads 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 750 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --embed_dim 32 --batch_size 50 --nheads 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 750 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --embed_dim 32 --batch_size 50 --nheads 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc forwards -vbc all_w_dyna -K 10 -ep 750 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc forwards -vbc all_w_dyna -K 10 -ep 750 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc forwards -vbc all_w_dyna -K 10 -ep 750 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc backwards -vbc all_w_dyna -K 10 -ep 750 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --batch_size 50
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc backwards -vbc all_w_dyna -K 10 -ep 750 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --batch_size 50
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc backwards -vbc all_w_dyna -K 10 -ep 750 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --batch_size 50
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc waypoint -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --feedforward_nhid 128
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc waypoint -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --feedforward_nhid 128
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc waypoint -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --feedforward_nhid 128
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc all_w_dyna -vbc all_w_dyna -K 10 -ep 5000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --embed_dim 32 --nheads 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc all_w_dyna -vbc all_w_dyna -K 10 -ep 5000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --embed_dim 32 --nheads 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc all_w_dyna -vbc all_w_dyna -K 10 -ep 5000 --lr 1e-4 -data_p 0.5 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --embed_dim 32 --nheads 4

# DT
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_BC -vbc DT_all -K 10 -ep 750 --lr 1e-4 -data_p 0.5 --model_class DT -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --embed_dim 32 --feedforward_nhid 128
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_BC -vbc DT_all -K 10 -ep 750 --lr 1e-4 -data_p 0.5 --model_class DT -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --embed_dim 32 --feedforward_nhid 128
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_BC -vbc DT_all -K 10 -ep 750 --lr 1e-4 -data_p 0.5 --model_class DT -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --embed_dim 32 --feedforward_nhid 128
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_RC -vbc DT_all -K 10 -ep 750 --lr 1e-4 -data_p 0.5 --model_class DT -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --embed_dim 32 --feedforward_nhid 128
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_RC -vbc DT_all -K 10 -ep 750 --lr 1e-4 -data_p 0.5 --model_class DT -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --embed_dim 32 --feedforward_nhid 128
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_RC -vbc DT_all -K 10 -ep 750 --lr 1e-4 -data_p 0.5 --model_class DT -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --embed_dim 32 --feedforward_nhid 128

# NN
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 5000 --lr 1e-4 -data_p 0.5 --model_class NN -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --embed_dim 32 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 5000 --lr 1e-4 -data_p 0.5 --model_class NN -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --embed_dim 32 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 5000 --lr 1e-4 -data_p 0.5 --model_class NN -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --embed_dim 32 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 --model_class NN -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --embed_dim 32 --batch_size 100 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 --model_class NN -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --embed_dim 32 --batch_size 100 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 1500 --lr 1e-4 -data_p 0.5 --model_class NN -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --embed_dim 32 --batch_size 100 #

##############
# V low data #
##############
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 100000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --embed_dim 128 --batch_size 100 --feedforward_nhid 128 --nlayers 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 100000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --embed_dim 128 --batch_size 100 --feedforward_nhid 128 --nlayers 2  #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 100000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --embed_dim 128 --batch_size 100 --feedforward_nhid 128 --nlayers 2  #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc future -vbc all_w_dyna -K 10 -ep 20000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --nlayers 2 --nheads 4 --feedforward_nhid 32
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc future -vbc all_w_dyna -K 10 -ep 20000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --nlayers 2 --nheads 4 --feedforward_nhid 32
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc future -vbc all_w_dyna -K 10 -ep 20000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --nlayers 2 --nheads 4 --feedforward_nhid 32
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc past -vbc all_w_dyna -K 10 -ep 20000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --embed_dim 32 --nlayers 4 --nheads 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc past -vbc all_w_dyna -K 10 -ep 20000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --embed_dim 32 --nlayers 4 --nheads 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc past -vbc all_w_dyna -K 10 -ep 20000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --embed_dim 32 --nlayers 4 --nheads 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 15000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --embed_dim 32 --feedforward_nhid 128 --nheads 4 --nlayers 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 15000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --embed_dim 32 --feedforward_nhid 128 --nheads 4 --nlayers 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 15000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --embed_dim 32 --feedforward_nhid 128 --nheads 4 --nlayers 2
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 15000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --embed_dim 32 --batch_size 50 --nheads 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 15000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --embed_dim 32 --batch_size 50 --nheads 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 15000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --embed_dim 32 --batch_size 50 --nheads 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc forwards -vbc all_w_dyna -K 10 -ep 15000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc forwards -vbc all_w_dyna -K 10 -ep 15000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc forwards -vbc all_w_dyna -K 10 -ep 15000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc backwards -vbc all_w_dyna -K 10 -ep 15000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --batch_size 50
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc backwards -vbc all_w_dyna -K 10 -ep 15000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --batch_size 50
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc backwards -vbc all_w_dyna -K 10 -ep 15000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --batch_size 50
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 30000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 30000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 30000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc waypoint -vbc all_w_dyna -K 10 -ep 30000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --feedforward_nhid 128
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc waypoint -vbc all_w_dyna -K 10 -ep 30000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --feedforward_nhid 128
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc waypoint -vbc all_w_dyna -K 10 -ep 30000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --feedforward_nhid 128
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc all_w_dyna -vbc all_w_dyna -K 10 -ep 100000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --embed_dim 32 --nheads 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc all_w_dyna -vbc all_w_dyna -K 10 -ep 100000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --embed_dim 32 --nheads 4
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc all_w_dyna -vbc all_w_dyna -K 10 -ep 100000 --lr 1e-4 -data_p 0.025 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --embed_dim 32 --nheads 4

# DT
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_BC -vbc DT_all -K 10 -ep 15000 --lr 1e-4 -data_p 0.025 --model_class DT -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --embed_dim 32 --feedforward_nhid 128
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_BC -vbc DT_all -K 10 -ep 15000 --lr 1e-4 -data_p 0.025 --model_class DT -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --embed_dim 32 --feedforward_nhid 128
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_BC -vbc DT_all -K 10 -ep 15000 --lr 1e-4 -data_p 0.025 --model_class DT -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --embed_dim 32 --feedforward_nhid 128
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_RC -vbc DT_all -K 10 -ep 15000 --lr 1e-4 -data_p 0.025 --model_class DT -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --embed_dim 32 --feedforward_nhid 128
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_RC -vbc DT_all -K 10 -ep 15000 --lr 1e-4 -data_p 0.025 --model_class DT -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --embed_dim 32 --feedforward_nhid 128
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc DT_RC -vbc DT_all -K 10 -ep 15000 --lr 1e-4 -data_p 0.025 --model_class DT -sb loss -s_loss 1 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --embed_dim 32 --feedforward_nhid 128

# NN
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 100000 --lr 1e-4 -data_p 0.025 --model_class NN -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --embed_dim 32 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 100000 --lr 1e-4 -data_p 0.025 --model_class NN -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --embed_dim 32 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc rnd -vbc all_w_dyna -K 10 -ep 100000 --lr 1e-4 -data_p 0.025 --model_class NN -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --embed_dim 32 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 30000 --lr 1e-4 -data_p 0.025 --model_class NN -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 3 --embed_dim 32 --batch_size 100 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 30000 --lr 1e-4 -data_p 0.025 --model_class NN -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 4 --embed_dim 32 --batch_size 100 #
python train.py minigrid --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 30000 --lr 1e-4 -data_p 0.025 --model_class NN -sb loss -s_loss 0.5 -a_loss 1 -wp minigrid_neurips_after_mistake -s 5 --embed_dim 32 --batch_size 100 #