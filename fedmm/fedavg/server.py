import math
import time
from collections import OrderedDict

import numpy as np
import wandb
from misc.utils import (assert_layer_num, freeze_target_layers,
                        get_layer_num, get_lora_config, get_state_dict,
                        init_model_and_tokenizer, print_trainable_parameters,
                        reinit_lora, set_state_dict, torch_load, torch_save)
from modules.federated import ServerModule
from peft import get_peft_model
from tqdm import tqdm
import torch

class Server(ServerModule):
    def __init__(self, args, sd, gpu_server):
        super().__init__(args, sd, gpu_server)

        self.model, self.tokenizer = init_model_and_tokenizer(args)
        self.lora_config = get_lora_config(args)

        self.model = get_peft_model(self.model, self.lora_config)

        self.model = self.model.cuda(self.gpu_id)
        freeze_target_layers(args, self.model)
        print_trainable_parameters(self.model)

        self.log = {
            'rnd_acc': [], 'rnd_lss': []
        }

        self.wandb_run = wandb.init(
            project="fedmm",
            name=f"{self.args.trial}_server",
            group=self.args.trial,
            config=vars(self.args),
        )

        self.scaling_factor = self.args.lora_alpha / self.args.lora_rank

    def on_round_begin(self, selected, curr_rnd):  # selected parameter unused but required by interface
        self.round_begin = time.time()
        self.curr_rnd = curr_rnd
        if curr_rnd == 0:
            self.sd['global'] = self.get_weights()
        if self.args.resume_from_round is not None and curr_rnd == self.args.resume_from_round:
            print(f'resume from checkpoint... re-initialize Server')
            prev_weights = torch_load(self.args.checkpt_path, f'server_state.pt')['model']
            set_state_dict(self.model, prev_weights, self.gpu_id)
            self.sd['global'] = self.get_weights()

    def on_round_complete(self, updated, curr_rnd):
        self.update(updated)
        self.sd['global'] = self.get_weights()

        if (curr_rnd + 1) % self.args.report_period == 0 or curr_rnd == self.args.n_rnds - 1:
            st = time.time()
            result_dict = self.evaluate()
            print(f'evaluate: {time.time() - st:.2f}s')

            acc = result_dict['accuracy'] if self.args.task != "mnli" else result_dict["matched_accuracy"]
            self.log['rnd_acc'].append(acc)
            self.logger.print(
                f'rnd: {self.curr_rnd}, acc: {acc:.4f}%, ({time.time() - self.round_begin:.2f}s)')

            log_dict = {
                'round': self.curr_rnd + 1
            }
            if self.args.task == "mnli":
                log_dict["matched_accuracy"] = result_dict["matched_accuracy"]
                log_dict["mismatched_accuracy"] = result_dict["mismatched_accuracy"]
            else:
                log_dict['accuracy'] = result_dict['accuracy']

            wandb.log(log_dict, step=self.curr_rnd + 1)
            self.save_log()

        # rerun svd
        if self.args.recalculate_svd_period != 0:
            if ((curr_rnd + 1) % self.args.recalculate_svd_period == 0
                    and (curr_rnd + 1) > self.args.svd_warmup_steps):
                print(f"Server : rerun svd for round {curr_rnd + 1}")
                reinit_lora(self.model)
                self.sd['global'] = self.get_weights()

        # Save server model periodically and at last round
        CHECKPOINT_PERIOD = 300
        if (curr_rnd > 0 and curr_rnd % CHECKPOINT_PERIOD == 0) or curr_rnd == self.args.n_rnds - 1:
            torch_save(self.args.checkpt_path, f'server_state.pt', {
                'model': get_state_dict(self.model),
                'log': self.log,
            })

    def update(self, updated):
        st = time.time()
        local_weights_diff = {}
        local_train_sizes = []
        for c_id in updated:
            local_weights_diff[c_id] = self.sd[c_id]['theta_diff'].copy()
            local_train_sizes.append(self.sd[c_id]['train_size'])
            del self.sd[c_id]
        self.logger.print(
            f'all clients have been uploaded ({time.time() - st:.2f}s)')

        st = time.time()
        ratio = None if self.args.balanced else (
            np.array(local_train_sizes) / np.sum(local_train_sizes)).tolist()
        self.set_weights(self.model, self.aggregate(local_weights_diff, ratio))
        self.logger.print(
            f'global model has been updated ({time.time() - st:.2f}s)')

    def set_weights(self, model, state_dict):
        set_state_dict(model, state_dict, self.gpu_id)

    def get_weights(self):
        return {
            'model': get_state_dict(self.model),
        }

    def _agg_flora(self, local_weights_diff, aggr_theta, orig_theta, ratio):
        print("Server: flora aggregation")
        updated_layer_num = set()
        for name, _ in tqdm(self.model.named_parameters(), desc="Aggregating weights"):
            if 'base_layer' in name and ('bias' not in name):
                name_A = name.replace('base_layer', 'lora_A.default')
                name_B = name.replace('base_layer', 'lora_B.default')

                # stack up clients' A and B
                updates_A = [(orig_theta[name_A] + theta[name_A]) * ratio[j]
                             for j, theta in enumerate(local_weights_diff.values())]
                updates_B = [(orig_theta[name_B] + theta[name_B])
                             for theta in local_weights_diff.values()]
                updates_A, updates_B = np.concatenate(
                    updates_A, 0), np.concatenate(updates_B, 1)

                # sum of clients' B @ A
                updates_all = updates_B @ updates_A  # dim: (d_out, d_in)
                aggr_theta[name] = orig_theta[name] + \
                    self.scaling_factor * updates_all
                updated_layer_num.add(get_layer_num(name))
            elif 'lora_A' in name:
                assert_layer_num(name, updated_layer_num)
                aggr_theta[name] = torch.zeros_like(
                    torch.tensor(orig_theta[name]))
                torch.nn.init.kaiming_uniform_(
                    aggr_theta[name], a=math.sqrt(5))
            elif 'lora_B' in name:
                assert_layer_num(name, updated_layer_num)
                aggr_theta[name] = torch.zeros_like(
                    torch.tensor(orig_theta[name]))
            else:
                aggr_theta[name] = orig_theta[name]

        return aggr_theta


    def _agg_fedex(self, local_weights_diff, aggr_theta, orig_theta, ratio):
        print("Server: fedex aggregation")
        updated_layer_num = set()
        for name, _ in tqdm(self.model.named_parameters(), desc="Aggregating weights"):
            if ('base_layer' in name) and ('bias' not in name):
                name_A = name.replace('base_layer', 'lora_A.default')
                name_B = name.replace('base_layer', 'lora_B.default')

                # sum of clients' B @ A
                updates_all = [(orig_theta[name_B] + theta[name_B]) @ (orig_theta[name_A] +
                    theta[name_A]) * ratio[j] for j, theta in enumerate(local_weights_diff.values())]
                updates_all = np.sum(updates_all, 0)

                # sum of clients' B x sum of clients' A
                updates_B = [(orig_theta[name_B] + theta[name_B]) * ratio[j]
                             for j, theta in enumerate(local_weights_diff.values())]
                updates_A = [(orig_theta[name_A] + theta[name_A]) * ratio[j]
                             for j, theta in enumerate(local_weights_diff.values())]
                updates_B_A = np.sum(updates_B, 0) @ np.sum(updates_A, 0)

                updates_res = updates_all - updates_B_A
                aggr_theta[name] = orig_theta[name] + \
                    self.scaling_factor * updates_res
                updated_layer_num.add(get_layer_num(name))
            elif 'lora' in name:
                assert_layer_num(name, updated_layer_num)
                updates = np.sum([theta[name] * ratio[j]
                                 for j, theta in enumerate(local_weights_diff.values())], 0)
                aggr_theta[name] = orig_theta[name] + updates
            else:
                aggr_theta[name] = orig_theta[name]

        return aggr_theta

    def aggregate(self, local_weights_diff, ratio=None):
        st = time.time()
        aggr_theta = OrderedDict([(k, None) for k in next(
            iter(local_weights_diff.values())).keys()])
        orig_theta = self.get_weights()["model"]
        ratio = ratio if isinstance(ratio, list) else [
            1 / len(local_weights_diff)] * len(local_weights_diff)

        # flora or fedex aggregation
        if self.args.agg_flora or self.args.agg_fedex:
            assert self.args.model == 'fedavg', "flora and fedex aggregation only support fedavg"
            args = (local_weights_diff, aggr_theta, orig_theta, ratio)
            if self.args.agg_flora:
                aggr_theta = self._agg_flora(*args)
            else:
                aggr_theta = self._agg_fedex(*args)

            self.logger.print(f"aggregation done ({round(time.time() - st, 3)} s)")
            return aggr_theta

        for name, _ in tqdm(self.model.named_parameters(), desc="Aggregating weights"):

            # aggregate weights according to the model type
            # update is theta_0 + \sum_k (n_k/n) * (theta^k_T - theta^k_0)
            if self.args.model == 'fedavg':
                if 'lora' in name:
                    update = np.sum(
                        [theta[name] * ratio[j] for j, theta in enumerate(local_weights_diff.values())], 0)
                    aggr_theta[name] = orig_theta[name] + update
                else:
                    aggr_theta[name] = orig_theta[name]

            elif self.args.model == 'ffa':
                if 'lora_B' in name or 'lora_embedding_B' in name:
                    update = np.sum(
                        [theta[name] * ratio[j] for j, theta in enumerate(local_weights_diff.values())], 0)
                    aggr_theta[name] = orig_theta[name] + update
                else:
                    aggr_theta[name] = orig_theta[name]

        self.logger.print(f'aggregation done ({round(time.time() - st, 3)} s)')
        return aggr_theta