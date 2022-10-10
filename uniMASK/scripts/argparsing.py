def parse_common_args(parser, possible_batch_codes):
    parser.add_argument(
        "--train_batch_code",
        "-tbc",
        type=str,
        default="rnd",
        choices=possible_batch_codes,
    )
    parser.add_argument("--rew_batch_code", "-rbc", type=str, choices=possible_batch_codes)
    parser.add_argument(
        "--val_batch_code",
        "-vbc",
        type=str,
        default="all",
        choices=possible_batch_codes,
    )
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data_prop", "-data_p", type=float)
    data_group.add_argument("--num_trajs", type=int)
    epochs_group = parser.add_mutually_exclusive_group(required=True)
    epochs_group.add_argument("--epochs", "-ep", type=int)
    epochs_group.add_argument("--timesteps", "-ts", type=int)
    parser.add_argument(
        "--finetune",
        type=str,
        help="Name of run folder (in ./data/) to finetune. If unspecified, no finetuning will be done.",
    )
    parser.add_argument(
        "--seq_len",
        "-K",
        type=int,
        default=5,
        help="Context length to feed to the network",
    )
    parser.add_argument(
        "--embed_dim",
        "-edim",
        type=int,
        default=128,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--feedforward_nhid",
        "-fnhid",
        type=int,
        default=128,
        help="Feedforward dimension",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=0,
        help="Training seed",
    )
    parser.add_argument(
        "--state_loss",
        "-s_loss",
        type=float,
        default=1,
        help="Loss for state",
    )
    parser.add_argument(
        "--action_loss",
        "-a_loss",
        type=float,
        default=1,
        help="Loss for action",
    )
    parser.add_argument(  # NOTE: 1e-07 is proportional to action_loss 1 in hopper medium.
        "--rtg_loss",
        "-r_loss",
        type=float,
        default=0,
        help="Loss for return-to-go",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--dropout",
        "-drop",
        type=float,
        default=0,
        help="Dropout",
    )
    parser.add_argument(
        "--t_enc",
        "-te",
        dest="timestep_encoding",
        default=False,
        action="store_true",
        help="Use timestep encoding rather than positional encoding",
    )
    parser.add_argument(
        "--unstacked",
        default=False,
        action="store_true",
        help="Don't use stacked tokens",
    )
    parser.add_argument(
        "--nlayers",
        type=int,
        default=3,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--nheads",
        type=int,
        default=8,
        help="Number of transformer heads (this doesn't do anything for an NN)",
    )
    parser.add_argument(
        "--action_tanh",
        default=False,
        action="store_true",
        help="Don't use stacked tokens",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="If not set, will use the default defined in the subtrainer",
    )

    ##############
    # wandb args #
    ##############
    parser.add_argument(
        "--wandb_project",
        "-wp",
        type=str,
        help="wandb project name. If none specified, will not log to wandb",
    )
    parser.add_argument(
        "--wandb_notes",
        "-wn",
        type=str,
        help='wandb notes. "This helps you remember what you were doing when you ran this run." ',
    )
    parser.add_argument(
        "--wandb_tags",
        "-wt",
        type=str,
        nargs="*",
        help='"Tags can be used to label runs with particular features..."',
    )
    parser.add_argument(
        "--model_class",
        "-mc",
        type=str,
        default="FB",
        choices=["FB", "NN", "DT"],
        help="Model type, transformer or NN",
    )
    parser.add_argument(
        "--save_best",
        "-sb",
        type=str,
        choices=["loss", "rew"],
        help="Save best model (with respect to loss or reward) to disk.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Name suffix",
    )

    parser.add_argument(
        "--rnd_suffix",
        default=False,
        action="store_true",
        help="Add a random suffix to the name (to prevent name overlaps).",
    )

    ####################
    # wandb sweep args #
    ####################
    parser.add_argument(
        "--wandb_sweep_params",
        "-sweep_p",
        type=str,
        help="Comma-separated model,tbc,vbc,rbc,epochs. Leave any of these empty to have them undefined.",
    )

    parser.add_argument(
        "--wandb_sweep_t_enc",
        "-sweep_te",
        dest="sweep_timestep_encoding",
        type=int,
        help="Specify timestep encoding as int (cast to bool). Useful for wandb sweeps.",
    )

    #################
    # Rew eval args #
    #################
    parser.add_argument(
        "--train_rew_eval_num",
        type=int,
        default=10,
        help="Number of reward evaluation rollouts for reward evals during training",
    )
    parser.add_argument(
        "--final_rew_eval_num",
        type=int,
        help="Number of reward evaluation rollouts for final reward evaluation",
    )
    parser.add_argument(
        "--sequential_eval",
        "-seq",
        default=False,
        action="store_true",
        help="Makes reward evaluations use sequential method instead of parallel. Likely only useful for debugging",
    )
    parser.add_argument(
        "--rew_eval_freq",
        type=int,
        default=10,
        help="Frequency of reward evaluation rollouts",
    )
    parser.add_argument(
        "--eval_types",
        "-et",
        type=str,
        nargs="+",
        default=[],
        help="List of eval types. Can be strings corresponding to eval types (BC, auto) or target rewards e.g., -tr 1800 3600",
    )
    parser.add_argument(
        "--reward_scale",
        "-rs",
        type=float,
        help="Factor by which to shrink rewards (and reward-to-go) when evaluating trajectories",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render rollouts.",
    )
    parser.add_argument(
        "--torch_cpus",
        type=int,
        help="Set number of CPUs used by torch.",
    )
