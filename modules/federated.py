from misc.utils import *
from dataset.dataset import GlueDataLoader
from modules.logger import Logger
import time
from typing import Dict


class ServerModule:
    def __init__(self, args, sd, gpu_server):

        self.args = args
        self._args = vars(self.args)
        self.gpu_id = gpu_server
        self.sd = sd
        self.loader = GlueDataLoader(self.args, is_server=True)
        self.logger = Logger(self.args, self.gpu_id, is_server=True)
        self.tokenizer = self.loader.te_loader.dataset.tokenizer

    def aggregate(self, local_weights, ratio=None):
        st = time.time()
        aggr_theta = OrderedDict([(k, None) for k in local_weights[0].keys()])
        if isinstance(ratio, list):
            for name, _ in aggr_theta.items():
                aggr_theta[name] = np.sum(
                    [theta[name]*ratio[j] for j, theta in enumerate(local_weights)], 0)
        else:
            ratio = 1/len(local_weights)
            for name, _ in aggr_theta.items():
                aggr_theta[name] = np.sum(
                    [theta[name] * ratio for j, theta in enumerate(local_weights)], 0)
        self.logger.print(f'aggregation done ({round(time.time()-st, 3)} s)')
        return aggr_theta

    @torch.no_grad()
    def evaluate(self):
        with torch.no_grad():
            self.model.to(self.gpu_id)
            pred_dict = {}
            for i, batch in enumerate(self.loader.te_loader):

                labels = batch.pop("labels")
                if self.args.task == "mnli":
                    is_matched = batch.pop("is_matched")
                batch = {k: v.to(self.gpu_id) for k, v in batch.items()}
                self.model.eval()
                outputs = self.model(**batch)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                if i == 0:
                    pred_dict["preds"] = []
                    pred_dict["labels"] = []
                    if self.args.task == "mnli":
                        pred_dict["is_matched"] = []

                pred_dict["preds"].append(preds.detach().cpu())
                pred_dict["labels"].append(labels)
                if self.args.task == "mnli":
                    pred_dict["is_matched"].append(is_matched)

            pred_dict = {k: torch.cat(v, dim=0) for k, v in pred_dict.items()}
            result_dict = self.accuracy(pred_dict=pred_dict)

        return result_dict

    @torch.no_grad()
    def accuracy(self, pred_dict: Dict[str, str]):
        if self.args.task == "mnli":
            return mnli_score(pred_dict=pred_dict)
        elif self.args.task in GLUE_TASKS:
            return glue_score(pred_dict)
        else:
            raise NotImplementedError()

    @torch.no_grad()
    def fl_score(self, preds, targets):
        num_classes = self.args.num_labels

        f1_per_class = []
        for class_idx in range(num_classes):
            tp = ((preds == class_idx) & (targets == class_idx)).sum().float()
            fp = ((preds == class_idx) & (targets != class_idx)).sum().float()
            fn = ((preds != class_idx) & (targets == class_idx)).sum().float()

            precision = tp / (tp + fp + 1e-8)  # mitigate division by zero
            recall = tp / (tp + fn + 1e-8)

            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            f1_per_class.append(f1)

        # macro-averaging
        f1_score = torch.mean(torch.stack(f1_per_class)) * 100.0
        return f1_score.item()

    def save_log(self):
        save(self.args.log_path, f'server.txt', {
            'args': self._args,
            'log': self.log
        })


class ClientModule:
    def __init__(self, args, w_id, g_id, sd):
        self.sd = sd
        self.gpu_id = g_id
        self.worker_id = w_id
        self.args = args
        self._args = vars(self.args)
        self.loader = GlueDataLoader(self.args)
        self.logger = Logger(self.args, self.gpu_id)

    def switch_state(self, client_id, curr_rnd):
        self.client_id = client_id
        self.loader.switch(client_id, curr_rnd=curr_rnd)
        self.logger.switch(client_id)
        time.sleep(0.1)
        if self.is_initialized():
            self.load_state()
        else:
            self.init_state()

    def is_initialized(self):
        return os.path.exists(os.path.join(self.args.checkpt_path, f'{self.client_id}_state.pt'))

    @property
    def init_state(self):
        raise NotImplementedError()

    @property
    def save_state(self):
        raise NotImplementedError()

    @property
    def load_state(self):
        raise NotImplementedError()

    @torch.no_grad()
    def evaluate(self):
        with torch.no_grad():
            pred_dict = {}
            for i, batch in enumerate(self.loader.te_loader):

                labels = batch.pop("labels")
                if self.args.task == "mnli":
                    is_matched = batch.pop("is_matched")
                batch = {k: v.to(self.gpu_id) for k, v in batch.items()}
                self.model.eval()
                outputs = self.model(**batch)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                if i == 0:
                    pred_dict["preds"] = []
                    pred_dict["labels"] = []
                    if self.args.task == "mnli":
                        pred_dict["is_matched"] = []
                pred_dict["preds"].append(preds.detach().cpu())
                pred_dict["labels"].append(labels)
                if self.args.task == "mnli":
                    pred_dict["is_matched"].append(is_matched)

            pred_dict = {k: torch.cat(v, dim=0)
                            for k, v in pred_dict.items()}
            result_dict = self.accuracy(pred_dict=pred_dict)
            

        return result_dict

    @torch.no_grad()
    def accuracy(self, pred_dict: Dict[str, str]):
        if self.args.task == "mnli":
            return mnli_score(pred_dict=pred_dict)
        elif self.args.task in GLUE_TASKS:
            return glue_score(pred_dict)
        else:
            raise NotImplementedError()

    def get_lr(self):
        return self.scheduler.get_last_lr()[0]

    def save_log(self):
        save(self.args.log_path, f'client_{self.client_id}.txt', {
            'args': self._args,
            'log': self.log
        })
