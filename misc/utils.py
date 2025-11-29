import glob
import json
import os
import random
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import torch
from peft import LoraConfig, TaskType
from peft.tuners.lora import LoraLayer
from rich import print
from torch import nn
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer)

from misc.forked_pdb import ForkedPdb

ACT_TYPE = {
    'relu': nn.ReLU,
    'gelu': nn.GELU
}
GLUE_TASKS = ["qnli", "qqp", "sst2", "mnli"]
NUM_LABELS = {
    "qnli": 2,
    "qqp": 2,
    "sst2": 2,
    "mnli": 3,
}

DATA_PATH = './data'
MODEL_PATH = './models'

#######################
# From Original Project
#######################
def str2bool(v):
    return v.lower() in ['true', 't']

def torch_save(base_dir, filename, data):
    os.makedirs(base_dir, exist_ok=True)
    fpath = os.path.join(base_dir, filename)
    torch.save(data, fpath)

def torch_load(base_dir, filename):
    fpath = os.path.join(base_dir, filename)
    return torch.load(fpath, map_location=torch.device('cpu'), weights_only=False)

def save(base_dir, filename, data):
    os.makedirs(base_dir, exist_ok=True)
    with open(os.path.join(base_dir, filename), 'w+') as outfile:
        json.dump(data, outfile)

def exists(base_dir, filename):
    return os.path.exists(os.path.join(base_dir, filename))

def join_glob(base_dir, filename):
    return glob.glob(os.path.join(base_dir, filename))

def remove_if_exist(base_dir, filename):
    targets = join_glob(base_dir, filename)
    if len(targets) > 0:
        for t in targets:
            os.remove(t)

def debugger():
    ForkedPdb().set_trace()

AVAIL_PARAM_SETS = ['A', 'B']
def partial_initialize_select_fn(model_type):
    if model_type == 'fedavg':
        return 'all'
    elif model_type == 'ffa':
        return 'A'
    elif model_type == 'ffb':
        return 'B'

def get_param_groups(model):
    lora_a_params, lora_b_params, other_params = [], [], []
    for name, param in model.named_parameters():
        if "lora_A" in name:
            lora_a_params.append(param)
        elif "lora_B" in name:
            lora_b_params.append(param)
        else:
            other_params.append(param)
    return lora_a_params, lora_b_params, other_params

def get_state_dict(model):
    state_dict = convert_tensor_to_np(model.state_dict())
    return state_dict

def set_state_dict(model, state_dict, gpu_id, skip_stat=False):
    state_dict = convert_np_to_tensor(state_dict, gpu_id, skip_stat=skip_stat, model=model.state_dict())
    model.load_state_dict(state_dict, strict=False)

def receive_state_dict_from_server(
    args,
    model,
    state_dict,
    gpu_id,
    client_id=-1,
    skip_stat=False,
    ):
    """
    Set the state dictionary of the model.
    Used by clients, when they receive the state dictionary from the server.
    """
    receive_all = False
    model_type = args.model
    state_dict = convert_np_to_tensor(state_dict, gpu_id, skip_stat=skip_stat, model=model.state_dict())

    # Case where both lora_A and lora_B are loaded
    if model_type == 'fedavg':
        print(f"client {client_id} : FedAvg - Loading all")
        receive_all = True
    elif args.recalculate_svd_period:
        print(f"client {client_id} : Server re-calculated svd - Loading all")
        receive_all = True
    # Case where only lora_A or lora_B is loaded
    else:
        if model_type == 'ffa':
            print(f"client {client_id} : FFA - Loading lora_B")
            filtered_state_dict = {k: v for k, v in state_dict.items() if 'lora_B' in k or 'lora_embedding_B' in k}
        if args.task in NUM_LABELS:
            print(f"client {client_id} : Loading classifier weights for {args.task}")
            for k, v in state_dict.items():
                if 'classifier' in k:
                    filtered_state_dict[k] = v

    if receive_all:
        model.load_state_dict(state_dict, strict=False)
        del state_dict
    else:
        model.load_state_dict(filtered_state_dict, strict=False)
        del filtered_state_dict
        del state_dict

def convert_tensor_to_np(state_dict):
    return OrderedDict([(k,v.clone().detach().cpu().numpy()) for k, v in state_dict.items()])

def convert_np_to_tensor(state_dict, gpu_id, skip_stat=False, model=None):
    _state_dict = OrderedDict()
    for k,v in state_dict.items():
        if skip_stat:
            if 'running' in k or 'tracked' in k:
                _state_dict[k] = model[k]
                continue

        if len(np.shape(v)) == 0:
            _state_dict[k] = torch.tensor(v).cuda(gpu_id)
        else:
            if isinstance(v, torch.Tensor):
                _state_dict[k] = v.requires_grad_().cuda(gpu_id)
            else:
                _state_dict[k] = torch.tensor(v).requires_grad_().cuda(gpu_id)
    return _state_dict

def convert_np_to_tensor_cpu(state_dict):
    _state_dict = OrderedDict()
    for k,v in state_dict.items():
        _state_dict[k] = torch.tensor(v)
    return _state_dict


def json_load(path, file_name):
    with open(os.path.join(path, file_name), 'r') as f:
        return json.load(f)

def jsonl_save(path, file_name, data):
    with open(os.path.join(path, file_name), 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def jsonl_load(path, file_name):
    with open(os.path.join(path, file_name), 'r') as f:
        return [json.loads(line) for line in f]

def shuffle(seed, data):
    np.random.seed(seed)
    np.random.shuffle(data)
    return data

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param :.5f}"
    )

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def reinit_lora(model):
    # Recalculate SVD and initialize lora_A and lora_B
    for module in model.modules():
        if isinstance(module, LoraLayer):
            for adapter_name in module.active_adapters:
                lora_A = module.lora_A[adapter_name].weight.data
                lora_B = module.lora_B[adapter_name].weight.data
                prod = lora_B @ lora_A
                V, S, Uh = torch.linalg.svd(prod, full_matrices=False)
                r = lora_A.size(0)
                Vr = V[:, :r]
                Sr = S[:r]
                Uhr = Uh[:r]

                lora_A = Uhr
                lora_B = Vr @ torch.diag(Sr)

                module.lora_A[adapter_name].weight.data = lora_A
                module.lora_B[adapter_name].weight.data = lora_B


def get_layer_num(name):
    if "encoder" in name:
        return int(name.split("layer.")[1].split(".")[0])
    else:
        return int(name.split('layers.')[1].split('.')[0])

def assert_layer_num(name, updated_layer_num):
    assert get_layer_num(name) in updated_layer_num, \
        f"Base layer {get_layer_num(name)} must be updated before LoRA A layer {name}"


def init_model_and_tokenizer(args):
    # assume we are using RoBERTa
    num_labels = NUM_LABELS[args.task]
    config = AutoConfig.from_pretrained(args.backbone, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.backbone, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    return model, tokenizer

def get_lora_config(args):
    target_modules = ["query", "value"]
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.dropout,
        init_lora_weights="pissa" if args.peft == 'pissa' else True,
        task_type=TaskType.SEQ_CLS,
        use_dora=True if args.peft == 'dora' else False
    )
    return lora_config


def freeze_target_layers(args, model):
    # Freeze classification head
    if "roberta" in args.backbone or "vit" in args.backbone:
        for n, p in model.named_parameters():
            if 'classifier' in n or "attention.output.dense" in n:
                p.requires_grad = False

    # Define target layers to freeze
    target_freeze = None
    if args.model == 'ffa':
        target_freeze = ['lora_A', 'lora_embedding_A']
    for name, param in model.named_parameters():
        if target_freeze:
            if any(freeze_key in name for freeze_key in target_freeze):
                param.requires_grad = False


def glue_score(pred_dict: Dict[str, List]):
    preds = pred_dict["preds"]
    labels = pred_dict["labels"]
    acc = torch.mean((preds == labels).float()).item()
    metric = {"accuracy": acc * 100}
    return metric

def mnli_score(pred_dict: Dict[str, List]):
    preds = pred_dict["preds"]
    labels = pred_dict["labels"]
    is_matched = pred_dict["is_matched"]

    matched_preds = preds[is_matched == 1]
    matched_labels = labels[is_matched == 1]
    matched_acc = torch.mean((matched_preds == matched_labels).float()).item()

    mismatched_preds = preds[is_matched == 0]
    mismatched_labels = labels[is_matched == 0]
    mismatched_acc = torch.mean((mismatched_preds == mismatched_labels).float()).item()

    metric = {
        "matched_accuracy": matched_acc * 100,
        "mismatched_accuracy": mismatched_acc * 100
    }
    return metric
