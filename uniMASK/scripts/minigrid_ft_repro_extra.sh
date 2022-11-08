# parallel -k --lb -j 20 -a minigrid_ft_repro_extra.sh --delay 10
# Medium data #

python train.py minigridbig --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 20000 --lr 1e-7 -data_p 0.25 --finetune 500N_10len_rnd_rl_seed0 -s 3 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_repro_ft_extra --embed_dim 32 --feedforward_nhid 128 --nheads 4 --nlayers 2
python train.py minigridbig --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 20000 --lr 1e-7 -data_p 0.25 --finetune 500N_10len_rnd_rl_seed1 -s 3 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_repro_ft_extra --embed_dim 32 --feedforward_nhid 128 --nheads 4 --nlayers 2
python train.py minigridbig --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 20000 --lr 1e-7 -data_p 0.25 --finetune 500N_10len_rnd_rl_seed2 -s 3  -sb loss -s_loss 1 -a_loss 1 -wp minigrid_repro_ft_extra --embed_dim 32 --feedforward_nhid 128 --nheads 4 --nlayers 2
python train.py minigridbig --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 20000 --lr 1e-7 -data_p 0.25 --finetune 500N_10len_rnd_rl_seed3 -s 3  -sb loss -s_loss 1 -a_loss 1 -wp minigrid_repro_ft_extra --embed_dim 32 --feedforward_nhid 128 --nheads 4 --nlayers 2
python train.py minigridbig --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 20000 --lr 1e-7 -data_p 0.25 --finetune 500N_10len_rnd_rl_seed4 -s 3  -sb loss -s_loss 1 -a_loss 1 -wp minigrid_repro_ft_extra --embed_dim 32 --feedforward_nhid 128 --nheads 4 --nlayers 2
python train.py minigridbig --final_rew_eval_num 0 --dropout 0.1 -tbc BC -vbc all_w_dyna -K 10 -ep 20000 --lr 1e-7 -data_p 0.25 --finetune 500N_10len_rnd_rl_seed5 -s 3  -sb loss -s_loss 1 -a_loss 1 -wp minigrid_repro_ft_extra --embed_dim 32 --feedforward_nhid 128 --nheads 4 --nlayers 2

python train.py minigridbig --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 20000 --lr 1e-7 -data_p 0.25 --finetune 500N_10len_rnd_rl_seed0 -s 3 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_repro_ft_extra --embed_dim 32 --batch_size 50 --nheads 4
python train.py minigridbig --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 20000 --lr 1e-7 -data_p 0.25 --finetune 500N_10len_rnd_rl_seed1 -s 3 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_repro_ft_extra --embed_dim 32 --batch_size 50 --nheads 4
python train.py minigridbig --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 20000 --lr 1e-7 -data_p 0.25 --finetune 500N_10len_rnd_rl_seed2 -s 3  -sb loss -s_loss 1 -a_loss 1 -wp minigrid_repro_ft_extra --embed_dim 32 --batch_size 50 --nheads 4
python train.py minigridbig --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 20000 --lr 1e-7 -data_p 0.25 --finetune 500N_10len_rnd_rl_seed3 -s 3  -sb loss -s_loss 1 -a_loss 1 -wp minigrid_repro_ft_extra --embed_dim 32 --batch_size 50 --nheads 4
python train.py minigridbig --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 20000 --lr 1e-7 -data_p 0.25 --finetune 500N_10len_rnd_rl_seed4 -s 3  -sb loss -s_loss 1 -a_loss 1 -wp minigrid_repro_ft_extra --embed_dim 32 --batch_size 50 --nheads 4
python train.py minigridbig --final_rew_eval_num 0 --dropout 0.1 -tbc RC -vbc all_w_dyna -K 10 -ep 20000 --lr 1e-7 -data_p 0.25 --finetune 500N_10len_rnd_rl_seed5 -s 3  -sb loss -s_loss 1 -a_loss 1 -wp minigrid_repro_ft_extra --embed_dim 32 --batch_size 50 --nheads 4

python train.py minigridbig --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 20000 --lr 1e-7 -data_p 0.25 --finetune 500N_10len_rnd_rl_seed0 -s 3 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_repro_ft_extra
python train.py minigridbig --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 20000 --lr 1e-7 -data_p 0.25 --finetune 500N_10len_rnd_rl_seed1 -s 3 -sb loss -s_loss 1 -a_loss 1 -wp minigrid_repro_ft_extra
python train.py minigridbig --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 20000 --lr 1e-7 -data_p 0.25 --finetune 500N_10len_rnd_rl_seed2 -s 3  -sb loss -s_loss 1 -a_loss 1 -wp minigrid_repro_ft_extra
python train.py minigridbig --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 20000 --lr 1e-7 -data_p 0.25 --finetune 500N_10len_rnd_rl_seed3 -s 3  -sb loss -s_loss 1 -a_loss 1 -wp minigrid_repro_ft_extra
python train.py minigridbig --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 20000 --lr 1e-7 -data_p 0.25 --finetune 500N_10len_rnd_rl_seed4 -s 3  -sb loss -s_loss 1 -a_loss 1 -wp minigrid_repro_ft_extra
python train.py minigridbig --final_rew_eval_num 0 --dropout 0.1 -tbc goal_conditioned -vbc all_w_dyna -K 10 -ep 20000 --lr 1e-7 -data_p 0.25 --finetune 500N_10len_rnd_rl_seed5 -s 3  -sb loss -s_loss 1 -a_loss 1 -wp minigrid_repro_ft_extra