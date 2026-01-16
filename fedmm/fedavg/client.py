import math
import time
from copy import deepcopy

import torch
import torch.nn as nn
import wandb
from misc.utils import (NUM_LABELS, freeze_target_layers, get_lora_config,
                         init_model_and_tokenizer, print_trainable_parameters,
                         receive_state_dict_from_server, set_seed, torch_load,
                         torch_save)
from modules.federated import ClientModule
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.utils.batch_memory_manager import BatchMemoryManager
from peft import get_peft_model
from tqdm import tqdm
from transformers import (get_constant_schedule_with_warmup,
                          get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup)


class Client(ClientModule):
    def __init__(self, args, w_id, g_id, sd):
        super().__init__(args, w_id, g_id, sd)

        self.model, self.tokenizer = init_model_and_tokenizer(args)
        self.lora_config = get_lora_config(args)

        self.model = get_peft_model(self.model, self.lora_config)

        self.model = self.model.cuda(self.gpu_id)
        freeze_target_layers(args, self.model)
        print_trainable_parameters(self.model)
        self.tensor_dataset = None

    def init_state(self):
        set_seed(self.args.seed)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        self.log = {
            'lr': [], 'train_lss': [],
            'ep_lss': [], 'ep_acc': [],
            'rnd_lss': [], 'rnd_acc': [],
        }
        self.wandb_run = wandb.init(
            project="fedsvd",
            name=f"{self.args.trial}_client_{self.client_id}",
            group=self.args.trial,
            config=vars(self.args),
        )
        if self.args.dp:
            self.dp_init()

    def dp_init(self):
        batch_size = self.args.batch_size * self.args.accumulation_steps
        self.tensor_dataset = self.loader.get_tensor_dataset()
        num_batches = math.ceil(len(self.tensor_dataset) / batch_size)
        assert self.args.frac > 0
        sample_rate = (1 / num_batches) 

        self.noise_multiplier = get_noise_multiplier(
            sample_rate=sample_rate,
            steps=self.args.n_iter * self.args.n_rnds * self.args.frac,
            target_delta=self.args.delta,
            target_epsilon=self.args.eps,
            accountant="prv"
        )
        print(f"noise multiplier: {self.noise_multiplier:.4f}")

    def save_state(self):
        torch_save(self.args.checkpt_path, f'{self.client_id}_state.pt', {
            'log': self.log,
            'tensor_dataset': self.tensor_dataset if self.args.dp else None,
            'noise_multiplier': self.noise_multiplier if self.args.dp else None,
        })

    def load_state(self):

        if self.args.resume_from_round is not None:
            set_seed(self.args.seed)
            self.wandb_run = wandb.init(
                project="fedsvd",
                name=f"{self.args.trial}_client_{self.client_id}",
                group=self.args.trial,
                config=vars(self.args),
            )

        loaded_state = torch_load(
            self.args.checkpt_path, f'{self.client_id}_state.pt')
        self.log = loaded_state['log']
        if self.args.dp:
            self.noise_multiplier = loaded_state['noise_multiplier']
            self.tensor_dataset = loaded_state['tensor_dataset']

    def on_receive_message(self, curr_rnd):
        self.curr_rnd = curr_rnd

        # Update state dict
        self.update(self.sd['global'])

        # optimizer reset for every round
        self.optimizer = self.get_optimizer(self.model)
        print(f"client {self.client_id}: re-init the optimizer")

        # Scheduler reset for every round
        self.scheduler = self.get_scheduler(
            self.optimizer, self.loader.pa_loader)
        t_total = self.args.n_eps * len(self.loader.pa_loader)
        warmup_steps = int(t_total * self.args.scheduler_warmup_ratio)
        print(
            f"client {self.client_id} : scheduler reset with warmup_steps {warmup_steps}")

    def update(self, update):
        receive_state_dict_from_server(
            args=self.args,
            model=self.model,
            state_dict=update['model'],
            gpu_id=self.gpu_id,
            client_id=self.client_id,
            skip_stat=True,
        )

    def get_optimizer(self, model):
        if "roberta" in self.args.backbone or "vit" in self.args.backbone:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay)
        return optimizer

    def get_scheduler(self, optimizer, dataloader):
        t_total = self.args.n_eps * len(dataloader)
        warmup_steps = int(t_total * self.args.scheduler_warmup_ratio)

        kwargs = {
            "optimizer": optimizer,
            "num_warmup_steps": warmup_steps if self.args.task not in NUM_LABELS else 0,
            "num_training_steps": t_total
        }

        if self.args.scheduler_type == 'cosine-decay':
            scheduler = get_cosine_schedule_with_warmup(**kwargs)
        elif self.args.scheduler_type == "linear-decay":
            scheduler = get_linear_schedule_with_warmup(**kwargs)
        elif self.args.scheduler_type == 'constant':
            kwargs.pop("num_training_steps")
            scheduler = get_constant_schedule_with_warmup(**kwargs)
        return scheduler

    def on_round_begin(self, curr_rnd):
        # store initial parameter
        init_state_dict = {k: v.clone().detach().cpu()
                           for k, v in self.model.state_dict().items() if 'lora' in k}
        if self.args.dp:
            self.dp_train()
        else:
            self.train()
        self.transfer_to_server(init_state_dict)

    def train(self):
        accumulation_steps = self.args.accumulation_steps
        result_dict = None
        avg_epoch_loss = None

        for ep in tqdm(range(self.args.n_eps), desc=f'Client {self.client_id} Training'):
            st = time.time()
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            for i, batch in tqdm(enumerate(self.loader.pa_loader), desc=f'Client {self.client_id} Training - Batchwise'):

                if i >= self.args.n_iter * self.args.accumulation_steps:
                    break

                batch = {k: v.to(self.gpu_id) for k, v in batch.items()}
                outputs = self.model(**batch)
                # Scale loss by accumulation steps
                loss = outputs['loss'] / accumulation_steps
                loss.backward()

                if (i + 1) % accumulation_steps == 0:  # Perform optimizer step after accumulation
                    # Calculate gradient norm
                    grad_norm = torch.norm(torch.stack(
                        [torch.norm(p.grad) for p in self.model.parameters() if p.grad is not None]), p=2)
                    self.wandb_run.log({
                        'grad_norm': grad_norm.item()
                    })
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.args.clip_grad_norm)
                    self.optimizer.step()

                    self.optimizer.zero_grad()

                epoch_loss += loss.item() * accumulation_steps  # Accumulate the actual loss
                num_batches += 1
                if (i + 1) % accumulation_steps == 0:
                    self.scheduler.step()

                self.wandb_run.log({
                    'batch_loss': outputs['loss'].item(),
                    'learning_rate': self.get_lr(),
                })

            # Evaluate at last epoch, in every 2 rounds
            if ep == self.args.n_eps - 1 and (self.curr_rnd + 1) % self.args.report_period == 0:
                avg_epoch_loss = epoch_loss / num_batches
                result_dict = self.evaluate()
                log_dict = {
                    'round': self.curr_rnd + 1,
                    'epoch': ep + 1,
                    'epoch_loss': avg_epoch_loss,
                    'learning_rate': self.get_lr(),
                }
                if self.args.task == "mnli":
                    log_dict["matched_accuracy"] = result_dict["matched_accuracy"]
                    log_dict["mismatched_accuracy"] = result_dict["mismatched_accuracy"]
                else:
                    log_dict['accuracy'] = result_dict['accuracy']
                self.wandb_run.log(log_dict)
                # epoch-wise log
                self._log_metrics(result_dict, avg_epoch_loss, st, ep)

        # round-wise log
        if (self.curr_rnd + 1) % self.args.report_period == 0 and result_dict is not None:
            self._log_metrics(result_dict, avg_epoch_loss, st)
            self.save_log()

    def dp_train(self):
        self.model.train()
        orig_model = deepcopy(self.model)
        result_dict = None
        avg_epoch_loss = None

        # Do not carry previous optimizer state and always init new optimizer
        optimizer = self.get_optimizer(self.model)

        batch_size = self.args.batch_size * self.args.accumulation_steps

        dataloader = self.loader.get_dp_dataloader(self.tensor_dataset,
                                                  batch_size)

        privacy_engine = PrivacyEngine()
        criterion = nn.CrossEntropyLoss(reduction="mean")
        model, optimizer, criterion, dataloader = privacy_engine.make_private(
            module=self.model,
            optimizer=optimizer,
            criterion=criterion,
            data_loader=dataloader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.args.max_grad_norm,
            grad_sample_mode="ghost",
        )

        scheduler = self.get_scheduler(optimizer, dataloader)

        ep = 0
        st = time.time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        num_updates = 0
        with BatchMemoryManager(
            data_loader=dataloader,
            optimizer=optimizer,
            max_physical_batch_size=self.args.batch_size,
        ) as memory_safe_dataloader:
            t = tqdm(memory_safe_dataloader,
                     total=self.args.n_iter,
                     desc=f'Client {self.client_id} Training - Batchwise')
            for batch in t:
                optimizer.zero_grad()

                input_ids, attention_mask, labels = batch
                batch = {"input_ids": input_ids,
                         "attention_mask": attention_mask,
                         "labels": labels}
                # skip empty batch
                if input_ids.size(0) == 0:
                    continue

                batch = {k: v.to(self.gpu_id)
                             for k, v in batch.items()}
                labels = batch.pop("labels")
                if self.args.task == "hellaswag":
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                    )
                    logits = outputs.logits
                    shift_logits = logits[:, :-1, :]
                    shift_labels = labels[:, 1:]
                    b, t, _ = shift_logits.size()
                    loss = criterion(
                        shift_logits.reshape(b * t, -1),
                        shift_labels.reshape(-1),
                        shift_logits.shape
                    )
                else:
                    outputs = model(**batch)
                    logits = outputs.logits
                    loss = criterion(logits, labels)

                loss.backward()

                optimizer.step()
                if not optimizer._is_last_step_skipped:
                    scheduler.step()
                    num_updates += 1
                epoch_loss += loss.item()
                num_batches += 1

                self.wandb_run.log({
                    'batch_loss': loss.item(),
                    'learning_rate': scheduler.get_last_lr()[0],
                })
                if num_updates == self.args.n_iter:
                    break

            # Evaluate at last epoch, in every 2 rounds
            if (self.curr_rnd + 1) % self.args.report_period == 0 or self.curr_rnd == self.args.n_rnds - 1:
                eps = privacy_engine.get_epsilon(self.args.delta)
                avg_epoch_loss = epoch_loss / num_batches
                result_dict = self.evaluate()
                log_dict = {
                    'round': self.curr_rnd + 1,
                    'epoch': ep + 1,
                    'epoch_loss': avg_epoch_loss,
                    'learning_rate': self.get_lr(),
                    "eps": eps
                }
                if self.args.task == "mnli":
                    log_dict["matched_accuracy"] = result_dict["matched_accuracy"]
                    log_dict["mismatched_accuracy"] = result_dict["mismatched_accuracy"]
                else:
                    log_dict['accuracy'] = result_dict['accuracy']
                self.wandb_run.log(log_dict)
                # log metrics epoch-wise
                self._log_metrics(result_dict, avg_epoch_loss, st, ep)

        state_dict = model._module.state_dict()
        orig_model.load_state_dict(state_dict)
        self.model = orig_model

        del privacy_engine
        # round-wise log
        if (self.curr_rnd + 1) % self.args.report_period == 0 and result_dict is not None:
            self._log_metrics(result_dict, avg_epoch_loss, st)
            self.save_log()

    def _log_metrics(self, result_dict, avg_epoch_loss, st, ep=None):
        log_keyword = "ep" if ep is not None else "rnd"

        acc = result_dict['accuracy'] if self.args.task != "mnli" else result_dict["matched_accuracy"]
        self.log[f'{log_keyword}_acc'].append(acc)
        if ep is not None:
            self.logger.print(
                f'rnd: {self.curr_rnd + 1}, ep: {ep + 1:>2}, acc: {acc:.4f}%, lr: {self.get_lr():.4f} ({time.time() - st:.2f}s)'
            )
        self.log[f'{log_keyword}_lss'].append(avg_epoch_loss)

    def transfer_to_server(self, init_state_dict):
        state_dict = {k: v.clone().detach().cpu()
                      for k, v in self.model.state_dict().items() if 'lora' in k}
        theta_diff = {k: (state_dict[k] - init_state_dict[k])
                      for k in state_dict.keys()}
        # convert it to numpy
        theta_diff = {k: v.numpy() for k, v in theta_diff.items()}
        self.sd[self.client_id] = {
            'train_size': len(self.loader.partition),
            "theta_diff": theta_diff
        }

    def __del__(self):
        # Cleanup wandb run when client is destroyed
        if hasattr(self, 'wandb_run'):
            self.wandb_run.finish()
