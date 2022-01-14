from transformers import AutoTokenizer, PreTrainedTokenizerFast, BatchEncoding
import torch
import json
from tqdm import tqdm
import os
import yaml
import time
from datetime import timedelta


def load_raw_texts(path: str) -> list():

    texts = []
    max_instances = 0
    with open(path, "r") as f:
        for line in tqdm(f, f'Loading texts from {path}'):
            instances = json.loads(line)
            texts.append(instances)

            if len(instances) > max_instances:
                max_instances = len(instances)

    return texts, max_instances


def tokenize(tokenizer: PreTrainedTokenizerFast, texts: list(), max_tokens) -> list():

    tokenized_instances_ids = []
    for instances in tqdm(texts, 'Tokenizing'):
        # Tokenize instances
        tokenized_instances_ids.append(tokenizer(
            instances, padding='max_length', max_length=max_tokens, truncation=True, return_attention_mask=False)['input_ids'])

    return tokenized_instances_ids


def truncate_pad_instances(tokenized_instances_ids, max_instances, max_tokens, padding_token):

    a = tokenized_instances_ids[0]
    b = a[:max_instances] + [[padding_token]*max_tokens] * \
        max(0, max_instances - len(a))

    return [
        instances[:max_instances] +
        [[padding_token]*max_tokens]*max(0, max_instances - len(instances)) for instances in tokenized_instances_ids
    ]


if __name__ == '__main__':
    print("Start preprocessing")
    t0 = time.time()

    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(
        description="Preprocess the texts.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory to store the preprocessed data. Must contain a params.yaml file.")

    # parser arguments
    args = parser.parse_args()
    output_path = args.output_dir

    # load preprocessing parameters
    with open(f"{args.output_dir}/params.yaml", "r") as f:
        params = yaml.load(f.read(), Loader=yaml.SafeLoader)['preprocess']

    dataset_path = params['dataset_path']
    max_instances = params['max_instances']
    max_tokens = params['max_tokens']
    tokenizer = AutoTokenizer.from_pretrained(
        f"sentence-transformers/{params['tokenizer']}")

    padding_idx = tokenizer.pad_token_id

    # Load raw texts
    train_texts, max_train_instances = load_raw_texts(
        os.path.join(dataset_path, 'train_texts.json'))
    val_texts, max_val_instances = load_raw_texts(
        os.path.join(dataset_path, 'val_texts.json'))
    test_texts, max_test_instances = load_raw_texts(
        os.path.join(dataset_path, 'test_texts.json'))

    if max_instances == -1:
        max_instances = max(
            max_train_instances, max_test_instances, max_val_instances)

    print(f"The maximum number of instances is {max_instances}")

    if max_tokens == -1:
        max_tokens = tokenizer.model_max_length
    else:
        max_tokens = min(tokenizer.model_max_length, max_tokens)

    print(
        f"Instances will be truncated to {max_tokens}")

    # Tokenize
    tokenized_train_instances_ids = tokenize(
        tokenizer, train_texts, max_tokens)
    tokenized_val_instances_ids = tokenize(
        tokenizer, val_texts, max_tokens)
    tokenized_test_instance_ids = tokenize(
        tokenizer, test_texts, max_tokens)

    # Pad and truncate instances
    tokenized_train_instances_ids = truncate_pad_instances(
        tokenized_train_instances_ids, max_instances, max_tokens, padding_idx)
    tokenized_val_instances_ids = truncate_pad_instances(
        tokenized_val_instances_ids, max_instances, max_tokens, padding_idx)
    tokenized_test_instance_ids = truncate_pad_instances(
        tokenized_test_instance_ids, max_instances, max_tokens, padding_idx)

    # Ids to tensor
    tokenized_train_instances_ids = torch.LongTensor(
        tokenized_train_instances_ids)
    tokenized_val_instances_ids = torch.LongTensor(
        tokenized_val_instances_ids)
    tokenized_test_instance_ids = torch.LongTensor(
        tokenized_test_instance_ids)

    print("Start saving to disk")

    # save train and test input ids to disk
    with open(os.path.join(dataset_path, 'train_labels.txt'), "r") as f:
        torch.save({
            'input-ids': tokenized_train_instances_ids,
            'labels': [labels.strip().split() for labels in f.readlines()]
        }, os.path.join(output_path, "train_data.pkl")
        )
    with open(os.path.join(dataset_path, 'val_labels.txt'), "r") as f:
        torch.save({
            'input-ids': tokenized_val_instances_ids,
            'labels': [labels.strip().split() for labels in f.readlines()]
        }, os.path.join(output_path, "val_data.pkl")
        )
    with open(os.path.join(dataset_path, 'test_labels.txt'), "r") as f:
        torch.save({
            'input-ids': tokenized_test_instance_ids,
            'labels': [labels.strip().split() for labels in f.readlines()]
        }, os.path.join(output_path, "test_data.pkl")
        )

    print(
        f"Preprocessing completed. Elapsed time: {str(timedelta(seconds=time.time() - t0)).split('.', 2)[0]}")
