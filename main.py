import os
from parser import Parser
from datetime import datetime

from misc.utils import DATA_PATH, set_seed
from modules.multiprocs import ParentProcess


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(args):

    args = set_config(args)

    if args.model in ['fedavg', 'ffa']:
        from fedmm.fedavg.server import Server
        from fedmm.fedavg.client import Client
    else:
        print('incorrect model was given: {}'.format(args.model))
        os._exit(0)

    print(args)
    set_seed(args.seed)
    pp = ParentProcess(args, Server, Client)
    pp.start()

def set_config(args):
    if args.recalculate_svd_period != 0:
        assert args.model == 'ffa', "SVD is not supported for non-FFA models"

    if args.n_clients == 1:
        print(args.n_clients)
        print("="*50)
        args.use_all = True
    else:
        args.use_all = False

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial = f'{now}_{args.task}_{args.model}' \
            if args.trial == None else f'{now}_{args.task}_{args.model}_{args.trial}'

    args.data_path = f'{DATA_PATH}/{args.task}/Questions/{args.dist}_split_{args.n_clients}'
    if args.resume_from_round is not None:
        args.checkpt_path = f'{args.base_path}/checkpoints/{args.resume_from_checkpoint_path}'
        args.log_path = f'{args.base_path}/logs/{args.resume_from_checkpoint_path}'
    else:
        args.checkpt_path = f'{args.base_path}/checkpoints/{trial}'
        args.log_path = f'{args.base_path}/logs/{trial}'
    return args

if __name__ == '__main__':
    main(Parser().parse())










