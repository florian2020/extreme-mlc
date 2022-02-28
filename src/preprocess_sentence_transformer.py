from transformers import AutoTokenizer, PreTrainedTokenizerFast, BatchEncoding
import torch
import numpy as np
import json
from tqdm import tqdm
import os
import yaml
import time
from datetime import timedelta


def load_raw_texts(path: str, concatenate_main_body: bool) -> list():
    """
    Load data where each document is stored as a list of
    sections in a json lines format. Also count on the fly
    the max number of instances for any document.

    concatenate_main_body: indicates whether the subsetcions of the main body should be
    considered as one or multiple sections
    """

    texts = []
    max_instances = 4
    with open(path, "r") as f:
        for line in tqdm(f, f'Loading texts from {path}'):
            instances = json.loads(line)

            if concatenate_main_body:
                texts.append([instances[0], instances[1],
                             " ".join(instances[2:-1]), instances[-1]])

            else:
                texts.append(instances)

                if len(instances) > max_instances:
                    max_instances = len(instances)

    return texts, max_instances


def tokenize(tokenizer: PreTrainedTokenizerFast, texts: list(), max_tokens) -> list():
    """
    Apply sentence transformer tokenizer which returns a list of token ids (padded to max_tokens length)
    for each passed text
    """

    return [tokenizer(instances, padding='max_length', max_length=max_tokens, truncation=True, return_attention_mask=False)[
        'input_ids']for instances in tqdm(texts, 'Tokenizing')]


def truncate_pad_instances(tokenized_instances_ids, max_instances, max_tokens, padding_token):
    """
    Pad instances such that all documents have the same number of instances.
    Make sure that independent of the number of instances the attachments are included
    """

    return [
        [*instances[:min(max_instances-1, len(instances)-1)], instances[-1]] +
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
    with open(f"{args.output_dir}/params_preprocess.yaml", "r") as f:
        params = yaml.load(f.read(), Loader=yaml.SafeLoader)['preprocess']

    dataset_path = params['dataset_path']
    max_instances = params['max_instances']
    max_tokens = params['max_tokens']
    tokenizer = AutoTokenizer.from_pretrained(
        f"sentence-transformers/{params['tokenizer']}")

    padding_idx = tokenizer.pad_token_id

    # Load raw texts
    train_texts, max_train_instances = load_raw_texts(
        os.path.join(dataset_path, 'train_texts.json'), params['concatenate_main_body'])
    val_texts, _ = load_raw_texts(
        os.path.join(dataset_path, 'val_texts.json'), params['concatenate_main_body'])
    test_texts, _ = load_raw_texts(
        os.path.join(dataset_path, 'test_texts.json'), params['concatenate_main_body'])

    if max_instances == -1:
        max_instances = max_train_instances

    print(
        f"Maximum {max_instances} instances of each document will be considered.")

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
    tokenized_train_instances_ids = torch.IntTensor(
        tokenized_train_instances_ids)
    tokenized_val_instances_ids = torch.IntTensor(
        tokenized_val_instances_ids)
    tokenized_test_instance_ids = torch.IntTensor(
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
