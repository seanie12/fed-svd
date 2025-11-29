import argparse
class Parser:

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()

    def set_arguments(self):

        self.parser.add_argument('--gpu', type=str, default='0')
        self.parser.add_argument('--seed', type=int, default=1234)

        self.parser.add_argument(
            '--model',
            type=str,
            default='fedavg',
            choices=['fedavg', 'ffa'],
            help="""
            fedavg: FedAvg (Train: A, B | Share: A, B)
            ffa: Freeze A (Train: B | Share: B)
            """
        )
        self.parser.add_argument('--agg-flora', action='store_true')
        self.parser.add_argument('--agg-fedex', action='store_true')
        self.parser.add_argument('--task', type=str, default='mnli', choices=["mnli", "qqp", "snli", "sst2", "qnli"])
        self.parser.add_argument('--dist', type=str, default='non_iid', choices=['iid', 'non_iid'])

        self.parser.add_argument(
            '--backbone',
            type=str,
            default='roberta-large')
        self.parser.add_argument('--n-workers', type=int, default=1)
        self.parser.add_argument('--n-clients', type=int, default=1)
        self.parser.add_argument('--n-rnds', type=int, default=100)
        self.parser.add_argument('--n-eps', type=int, default=1)
        self.parser.add_argument("--n-iter", type=int, default=10)
        self.parser.add_argument('--frac', type=float, default=None)
        self.parser.add_argument('--alpha', type=float, default=0.5)    # alpha for Dirichlet distribution
        self.parser.add_argument('--balanced', action='store_true')     # balanced split for SNLI + GLUE

        self.parser.add_argument('--trial', type=str, default=None)
        self.parser.add_argument('--data-path', type=str, default='./')
        self.parser.add_argument('--base-path', type=str, default='./')
        self.parser.add_argument('--report-period', type=int, default=20)

        self.parser.add_argument('--peft', type=str, default='lora', choices=["lora", "pissa"])
        self.parser.add_argument('--lora-alpha', type=int, default=8)
        self.parser.add_argument('--lora-rank', type=int, default=8)
        self.parser.add_argument("--dropout", type=float, default=0.05)

        # Optimizer & Scheduler Reset
        self.parser.add_argument('--scheduler-warmup-steps', type=int, default=20)
        self.parser.add_argument("--scheduler-warmup-ratio", type=float, default=0.1)
        self.parser.add_argument('--scheduler-type', type=str, default='constant', choices=['cosine-decay', 'linear-decay', 'constant'])

        # Training
        self.parser.add_argument('--accumulation-steps', type=int, default=1)
        self.parser.add_argument('--clip-grad-norm', type=float, default=1.0)
        self.parser.add_argument("--weight-decay", type=float, default=0)
        self.parser.add_argument('--lr', type=float, default=3e-4)
        self.parser.add_argument('--batch_size', type=int, default=8)

        # Resume
        self.parser.add_argument('--resume-from-checkpoint-path', type=str, default=None)
        self.parser.add_argument('--resume-from-round', type=int, default=None)

        # FeDual
        self.parser.add_argument('--recalculate-svd-period', type=int, default=1)
        self.parser.add_argument("--svd-warmup-steps", type=int, default=0)

        # DP-SGD
        self.parser.add_argument("--dp", action="store_true")
        self.parser.add_argument("--eps", type=float, default=6)
        self.parser.add_argument("--delta", type=float, default=1e-5)
        self.parser.add_argument("--max-grad-norm", type=float, default=2.0)
        
    def parse(self):
        args, unparsed  = self.parser.parse_known_args()
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        return args
