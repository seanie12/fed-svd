from misc.utils import DATA_PATH
from misc.utils import *

from datasets import load_dataset
import pandas as pd
import numpy as np
import argparse
import os
import time
from typing import Union, List, Dict
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))


def generate_data(dataset: str, n_clients: int, alpha: float, seed: int):
    st = time.time()
    global iid_path, non_iid_path
    iid_path = os.path.join(DATA_PATH, dataset, 'Questions',
                            f'iid_split_{n_clients}_{alpha}')
    
    non_iid_path = os.path.join(
        DATA_PATH, dataset, 'Questions', f'non_iid_split_{n_clients}_{alpha}')

    non_iid_fn = "non_iid"
    non_iid_path = os.path.join(
        DATA_PATH, dataset, 'Questions', non_iid_fn + f'_split_{n_clients}_{alpha}')

    non_iid_path_exists = os.path.exists(non_iid_path)

    train_data, val_data, test_data = get_data(dataset, seed)    # download data

    if dataset in glue_features.keys():
        topics = set([data["label"] for data in train_data if data["label"] > -1])
        dict_topic_data = {topic: [] for topic in topics}
        sentence_keys = glue_features[dataset]
        for item in train_data:
            topic = item["label"]
            #  skip if there is no label
            if item["label"] == -1:
                continue
            dict_topic_data[topic].append({
                "sentence1": item[sentence_keys[0]],
                "sentence2": item[sentence_keys[1]] if len(sentence_keys) > 1 else None,
                "label": item["label"],
            })
        if not non_iid_path_exists:
            os.makedirs(non_iid_path)
        else:
            print(f'{non_iid_path} already exists, skipping...')

        # val data, test data
        for i, data_now in enumerate([val_data, test_data]):
            mode = 'val' if i == 0 else 'test'
            data_list = []
            for item in data_now:
                if item["label"] == -1:
                    continue
                if dataset == "mnli":
                    data_list.append({
                        "sentence1": item[sentence_keys[0]],
                        "sentence2": item[sentence_keys[1]] if len(sentence_keys) > 1 else None,
                        "label": item["label"],
                        "is_matched": item["is_matched"]
                    })
                else:
                    data_list.append({
                        "sentence1": item[sentence_keys[0]],
                        "sentence2": item[sentence_keys[1]] if len(sentence_keys) > 1 else None,
                        "label": item["label"],
                    })
            jsonl_save(os.path.join(DATA_PATH, dataset, 'Questions'),
                        f'{mode}.jsonl', data_list)
            print(f'{mode}.jsonl has been saved')


    else:
        raise ValueError(f'Unsupported dataset: {dataset}')

    print(f'done ({time.time()-st:.2f})')


def get_data(dataset: str, seed:int) -> Union[pd.DataFrame, List[Dict[str, str]]]:
    st = time.time()
    if dataset in glue_features.keys():
        if 'mnli' in dataset:
            ds = load_dataset("glue", "mnli")
            train_data = ds["train"]
            val_matched, val_mismatched = ds["validation_matched"], ds["validation_mismatched"]
            test_matched, test_mismatched = ds["test_matched"], ds["test_mismatched"]
            def _add_is_matched(data, matched):
                item = dict(data)
                item["is_matched"] = matched
                return item
            val_data, test_data = [], []
            val_data = list(map(_add_is_matched, val_matched, [1]*len(val_matched)))
            val_data.extend(list(map(_add_is_matched, val_mismatched, [0]*len(val_mismatched))))
            test_data = list(map(_add_is_matched, test_matched, [1]*len(test_matched)))
            test_data.extend(list(map(_add_is_matched, test_mismatched, [0]*len(test_mismatched))))
        else:
            ds = load_dataset("glue", dataset)
            valid_key = 'validation'
            test_key = 'test'
            train_data, val_data, test_data = ds["train"], ds[valid_key], ds[test_key]

    else:
        raise ValueError(f'Unsupported dataset: {dataset}')

    print(f'{dataset} have been loaded ({time.time()-st:.2f} sec)')
    return train_data, val_data, test_data


def split_iid(data_dict, n_clients, dataset, has_topics=True, seed=1234):
    st = time.time()

    if has_topics:
        # Original topic-based IID splitting
        n_inst_per_topic = {topic: round(len(data)/n_clients)
                            for topic, data in data_dict.items()}

        for client_id in range(n_clients):
            x_part = []
            for topic in data_dict.keys():
                start_idx = client_id * n_inst_per_topic[topic]
                end_idx = (client_id + 1) * n_inst_per_topic[topic]
                x_part.extend(data_dict[topic][start_idx:end_idx])
            x_part = shuffle(seed, x_part)
            # save dicts in list into jsonl format
            jsonl_save(
                iid_path, f'{dataset}_iid_partition_{client_id}.jsonl', x_part)
            print(
                f'client_id:{client_id}, iid, n_train:{len(x_part)} ({time.time()-st:.2f})')
    else:
        # Simple random splitting for datasets without topics
        x_all = data_dict['all']

        # Shuffle once before splitting
        x_all = shuffle(seed, x_all)

        # Calculate instances per client
        n_total = len(x_all)
        n_per_client = n_total // n_clients

        for client_id in range(n_clients):
            start_idx = client_id * n_per_client
            end_idx = start_idx + n_per_client if client_id < n_clients - 1 else n_total
            x_part = x_all[start_idx:end_idx]

            # save dicts in list into jsonl format
            jsonl_save(
                iid_path, f'{dataset}_iid_partition_{client_id}.jsonl', x_part)
            print(
                f'client_id:{client_id}, iid, n_train:{len(x_part)} ({time.time()-st:.2f})')


def split_non_iid(data_dict, n_clients, n_clss, dataset, alpha, seed):
    st = time.time()
    # Generate random distributions for each class using Dirichlet distribution
    dist = np.random.dirichlet([alpha for _ in range(n_clients)], n_clss)
    dist = {clss_id: dist[i] for i, clss_id in enumerate(data_dict.keys())}
    n_data_per_clss = {clss_id: len(x) for clss_id, x in data_dict.items()}
    csv_data = {}
    for client_id in range(n_clients):
        x_part = []
        _n_data_per_clss = {clss_id: 0 for clss_id in data_dict.keys()}
        for clss_id in data_dict.keys():
            # Calculate how many samples of this class go to this client based on Dirichlet distribution
            _n = int(n_data_per_clss[clss_id] * dist[clss_id][client_id])
            _n_data_per_clss[clss_id] += _n

            # Take the first _n samples of this class
            x_part = [*x_part, *data_dict[clss_id][:_n]]
            # Remove the taken samples
            data_dict[clss_id] = data_dict[clss_id][_n:]

        x_part = shuffle(seed, x_part)
        # save dicts in list into jsonl format
        jsonl_save(
            non_iid_path, f'{dataset}_non_iid_partition_{client_id}.jsonl', x_part)

        print(
            f'client_id:{client_id}, non_iid, n_train:{len(x_part)}, n_data_per_clss:{_n_data_per_clss}, ({time.time()-st:.2f})')
        st = time.time()

        csv_data[client_id] = _n_data_per_clss

    # data_list is a dictionary of dictionaries : {client_id: {clss_id: n_data}}
    # convert to dataframe - columns: class_id, rows: client_id, values: n_data
    df = pd.DataFrame.from_dict(csv_data)
    result = {}
    # Skip the first column which is 'class_superset'
    for client_id in df.columns:
        # Sum the values for each class_superset
        client_data = df[client_id].groupby(df.index).sum().to_dict()
        result[client_id] = client_data

    print(result)


def split_non_iid_balanced(data_dict, dataset, seed):
    # assume n_clients = 3
    # we use [0.1, 0.9], [0.9, 0.1], [0.5, 0.5] data split for binary classification tasks and [0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9] for three-class classification tasks

    n_classes = NUM_LABELS[dataset]
    if n_classes == 2:
        # Binary classification
        distributions = [
            [0.1, 0.9],  # Client 0 gets 10% class 0, 90% class 1
            [0.9, 0.1],  # Client 1 gets 90% class 0, 10% class 1
            [0.5, 0.5]   # Client 2 gets 50% of each class
        ]
    else:
        # Three-class classification
        distributions = [
            [0.9, 0.05, 0.05],  # Client 0 mostly gets class 0
            [0.05, 0.9, 0.05],  # Client 1 mostly gets class 1
            [0.05, 0.05, 0.9]   # Client 2 mostly gets class 2
        ]

    for client_id in range(3):
        client_data = []
        dist = distributions[client_id]

        for class_id in data_dict:
            n_samples = int(len(data_dict[class_id]) * dist[class_id])
            client_data.extend(data_dict[class_id][:n_samples])
            data_dict[class_id] = data_dict[class_id][n_samples:]

        # Shuffle client data
        client_data = shuffle(seed, client_data)

        # Save to jsonl
        jsonl_save(non_iid_path, f'{dataset}_non_iid_partition_{client_id}.jsonl', client_data)



glue_features = {
    "sst2": ("sentence",),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
}

def glue_split_features(x, task):
    features = tuple([x[name] for name in glue_features[task]])
    label = x["label"]
    return (features, label)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-dt',
        '--dataset',
        default='mnli',
        choices=list(glue_features.keys())
    )
    parser.add_argument(
        '-nc',
        '--n-clients',
        type=int,
        default=5,
    )
    parser.add_argument(
        '-a',
        '--alpha',
        type=float,
        default=0.5,
        help='Alpha for Dirichlet distribution'
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )

    args = parser.parse_args()
    generate_data(args.dataset, args.n_clients, args.alpha, args.seed)
