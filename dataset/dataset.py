import os

import tokenizers
import torch
from misc.utils import *
from packaging import version
from rich import print
from torch.utils.data import DataLoader, Dataset, RandomSampler, TensorDataset
from transformers import AutoTokenizer, DataCollatorWithPadding

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# seed = 1234
# set_seed(seed)

class GlueDataLoader(object):
    def __init__(self, args, is_server=False) -> None:
        self.args = args
        self.n_workers = 1
        self.client_id = None
        if is_server:
            self.test = GlueDataset(client_id=-1, mode='test', dataset=args.task, args=args) if self.args.task in ['snli', ] else GlueDataset(client_id=-1, mode='val', dataset=args.task, args=args)
            self.te_loader = DataLoader(dataset=self.test, batch_size=self.args.batch_size,
                shuffle=False, num_workers=self.n_workers, pin_memory=True, collate_fn=DataCollatorWithPadding(self.test.tokenizer))

    def get_sampler(self, dataset, curr_rnd):
        g = torch.Generator()
        g.manual_seed(1234 + curr_rnd)
        return RandomSampler(dataset, generator=g)

    def switch(self, client_id, curr_rnd=None):
        if not self.client_id == client_id:
            self.client_id = client_id
            self.curr_rnd = curr_rnd

            # Switch Dataset
            self.partition = GlueDataset(client_id=client_id, mode='partition', dataset=self.args.task, args=self.args)
            self.valid = GlueDataset(client_id=client_id, mode='val', dataset=self.args.task, args=self.args)
            self.test = GlueDataset(client_id=client_id, mode='test', dataset=self.args.task, args=self.args) #if self.args.task != 'vqa' else None

            # Switch DataLoader as well
            self.pa_loader = DataLoader(
                dataset=self.partition, batch_size=self.args.batch_size, num_workers=self.n_workers, pin_memory=True,
                collate_fn=DataCollatorWithPadding(self.partition.tokenizer), worker_init_fn=worker_init_fn, sampler=self.get_sampler(self.partition, curr_rnd))

            if self.args.task in ['snli']:
                self.va_loader = DataLoader(
                    dataset=self.valid, batch_size=self.args.batch_size,
                    shuffle=False, num_workers=self.n_workers, pin_memory=True,
                    collate_fn=DataCollatorWithPadding(self.valid.tokenizer), worker_init_fn=worker_init_fn)

                self.te_loader = DataLoader(
                    dataset=self.test, batch_size=self.args.batch_size,
                    shuffle=False, num_workers=self.n_workers, pin_memory=True,
                    collate_fn=DataCollatorWithPadding(self.test.tokenizer), worker_init_fn=worker_init_fn)
            else:
                self.te_loader = DataLoader(
                    dataset=self.valid, batch_size=self.args.batch_size,
                    shuffle=False, num_workers=self.n_workers, pin_memory=True,
                    collate_fn=DataCollatorWithPadding(self.valid.tokenizer), worker_init_fn=worker_init_fn)

    def get_tensor_dataset(self):
        dataset = self.pa_loader.dataset
        tokenizer = dataset.tokenizer
        full_batch_size = len(dataset)

        _dataloader = DataLoader(
            dataset, full_batch_size, drop_last=False, shuffle=False,
            collate_fn=DataCollatorWithPadding(tokenizer),
            worker_init_fn=worker_init_fn)

        iterator = iter(_dataloader)
        full_batch = next(iterator)
        input_ids = full_batch["input_ids"]
        attention_mask = full_batch["attention_mask"]
        labels = full_batch["labels"]

        dataset = TensorDataset(input_ids, attention_mask, labels)

        return dataset

    def get_dp_dataloder(self, dataset, batch_size):
        dataloader = DataLoader(
            dataset, batch_size,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            sampler=self.get_sampler(dataset, self.curr_rnd)
        )

        return dataloader


class GlueDataset(Dataset):
    def __init__(self, client_id=-1, mode='partition', dataset=None, args=None):
        super().__init__()

        text_dir = os.path.join(DATA_PATH, dataset, 'Questions')

        # train
        if mode == 'partition':
            if args.dist == 'iid':
                data = jsonl_load(os.path.join(text_dir, f'iid_split_{args.n_clients}_{args.alpha}'), f'{dataset}_iid_partition_{client_id}.jsonl')
            else:
                if args.balanced:
                    data = jsonl_load(os.path.join(text_dir, f'non_iid_balanced_split_{args.n_clients}_{args.alpha}'), f'{dataset}_non_iid_partition_{client_id}.jsonl')
                else:
                    data = jsonl_load(os.path.join(text_dir, f'non_iid_split_{args.n_clients}_{args.alpha}'), f'{dataset}_non_iid_partition_{client_id}.jsonl')
        # val, test
        else:
            if args.task in NUM_LABELS.keys():
                data = jsonl_load(text_dir, f'{mode}.jsonl')    # val, test data doesn't change along with alpha
            else:
                data = jsonl_load(text_dir, f'{mode}_{args.alpha}.jsonl')

        self.data = data
        model_path = os.path.join(args.backbone)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.mode = mode
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        sentence1 = item["sentence1"]
        sentence2 = item["sentence2"]
        label = item["label"]

        inputs = self.tokenizer(
            sentence1, sentence2, padding=False,
            add_special_tokens=True, truncation=True, max_length=128)
        inputs["labels"] = label
        if self.args.task == "mnli" and self.mode != "partition":
            inputs["is_matched"] = item["is_matched"]
        return inputs
