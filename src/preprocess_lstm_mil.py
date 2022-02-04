import os
import json
import yaml
import time
from datetime import timedelta
import torch
import spacy
import numpy as np
from itertools import chain
from collections import Counter
from tqdm.auto import tqdm


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


def build_spacy_tokenizer(vocab):
    # get german tokenizer
    from spacy.lang.de import German
    # build tokenizer parameters
    prefixes = German.Defaults.prefixes
    suffixes = German.Defaults.suffixes
    infixes = German.Defaults.infixes
    prefix_search = spacy.util.compile_prefix_regex(
        prefixes).search if prefixes else None
    suffix_search = spacy.util.compile_suffix_regex(
        suffixes).search if suffixes else None
    infix_finditer = spacy.util.compile_infix_regex(
        infixes).finditer if infixes else None
    # add tokenizer exception for special tokens
    exc = German.Defaults.tokenizer_exceptions
    exc = spacy.util.update_exc(exc, {
        '[SEP]': [{spacy.symbols.ORTH: "[SEP]"}]
    })
    # create tokenizer
    return spacy.tokenizer.Tokenizer(
        vocab=spacy.vocab.Vocab(strings=vocab.keys()),
        rules=exc,
        prefix_search=prefix_search,
        suffix_search=suffix_search,
        infix_finditer=infix_finditer,
        token_match=German.Defaults.token_match,
        url_match=German.Defaults.url_match
    )


def build_spacy_english_tokenizer(vocab):
    # get german tokenizer
    from spacy.lang.en import English
    # build tokenizer parameters
    prefixes = English.Defaults.prefixes
    suffixes = English.Defaults.suffixes
    infixes = English.Defaults.infixes
    prefix_search = spacy.util.compile_prefix_regex(
        prefixes).search if prefixes else None
    suffix_search = spacy.util.compile_suffix_regex(
        suffixes).search if suffixes else None
    infix_finditer = spacy.util.compile_infix_regex(
        infixes).finditer if infixes else None
    # add tokenizer exception for special tokens
    exc = English.Defaults.tokenizer_exceptions
    exc = spacy.util.update_exc(exc, {
        '[SEP]': [{spacy.symbols.ORTH: "[SEP]"}]
    })
    # create tokenizer
    return spacy.tokenizer.Tokenizer(
        vocab=spacy.vocab.Vocab(strings=vocab.keys()),
        rules=exc,
        prefix_search=prefix_search,
        suffix_search=suffix_search,
        infix_finditer=infix_finditer,
        token_match=English.Defaults.token_match,
        url_match=English.Defaults.url_match
    )


def tokenize_doc(tokenizer, doc):
    """ tokenize all given texts """
    return [
        list(map(lambda t: str(t).lower(), tokenizer(text)))
        for text in doc
    ]


def tokenize(tokenizer, texts):

    return [tokenize_doc(tokenizer, doc) for doc in tqdm(texts, 'Tokenizing')]


def truncate_pad(tokenized_texts, max_instances, max_tokens, padding_token):

    # Pad each text first to the maximum length
    tokenized_texts = [[instance[:max_tokens] + [padding_token] * max(0, max_tokens - len(
        instance)) for instance in instances]for instances in tokenized_texts]

    # Pad missing instances
    return [
        instances[:max_instances] +
        [[padding_token]*max_tokens]*max(0, max_instances - len(instances)) for instances in tokenized_texts
    ]


def filter_vocab(vocab, embed, tokenized_texts, min_freq=1, max_size=200_000):
    # count token occurances and ignore tokens
    # that are not in the vocabulary
    counter = Counter(
        [token for instances in tokenized_texts for instance in instances for token in instance])
    # create filtered vocabulary containing the most frequent words
    filtered_vocab = [
        word
        for word, freq in counter.most_common()
        if (freq >= min_freq) and (word in vocab)
    ]
    filtered_vocab = filtered_vocab[: max_size]
    # filtered_vocab = counter.most_common(max_size)
    # filtered_vocab = [w for w, f in filtered_vocab if f >= min_freq]
    # add special tokens
    if "[SEP]".lower() in filtered_vocab:
        filtered_vocab.remove("[SEP]".lower())
    if "[UNK]".lower() in filtered_vocab:
        filtered_vocab.remove("[UNK]".lower())
    if "[PAD]".lower() in filtered_vocab:
        filtered_vocab.remove("[PAD]".lower())
    filtered_vocab.insert(0, "[SEP]".lower())
    filtered_vocab.insert(0, "[UNK]".lower())
    filtered_vocab.insert(0, "[PAD]".lower())
    # build embedding matrix for filtered vocab
    filtered_embed = [
        embed[vocab[token]
              ] if token in vocab else np.random.uniform(-1, 1, size=(embed.shape[1],))
        for token in filtered_vocab
    ]
    filtered_embed = np.stack(filtered_embed, axis=0)
    # create mapping for filtered vocab
    filtered_vocab = {token: i for i, token in enumerate(filtered_vocab)}
    assert len(filtered_vocab) == filtered_embed.shape[0]
    # return
    return filtered_vocab, filtered_embed


def convert_tokens_to_ids(vocab, tokenized_texts):
    unk_token_id = vocab["[unk]"]
    return [
        [[vocab.get(token.lower(), unk_token_id) for token in instance]
         for instance in instances] for instances in tokenized_texts
    ]


if __name__ == '__main__':
    print("Start preprocessing")
    t0 = time.time()

    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(
        description="Preprocess the texts and build the vocabulary.")
    parser.add_argument("--output-dir", type=str, help="Output directory.")
    # parser arguments
    args = parser.parse_args()

    # load preprocessing parameters
    with open(f"{args.output_dir}/params_preprocess.yaml", "r") as f:
        params = yaml.load(f.read(), Loader=yaml.SafeLoader)['preprocess']

    dataset_path = params['dataset_path']
    max_instances = params['max_instances']
    max_tokens = params['max_tokens']

    # load pretrained embeddings
    vocab = np.load(params['pretrained_vocab'])
    embed = np.load(params['pretrained_embed'])
    # change special tokens
    vocab[vocab == "<SEP>"] = "[SEP]"
    vocab[vocab == "<PAD>"] = "[PAD]"
    vocab[vocab == "<UNK>"] = "[UNK]"
    # convert vocab to list
    vocab = {token.lower(): i for i, token in enumerate(vocab.tolist())}

    # get the tokenizer
    tokenizer = {
        "Spacy_de": build_spacy_tokenizer,
        'Spacy_en': build_spacy_english_tokenizer
    }[params['tokenizer']](vocab)

    # load texts
    train_texts, max_train_instances = load_raw_texts(
        os.path.join(dataset_path, 'train_texts.json'))
    val_texts, _ = load_raw_texts(
        os.path.join(dataset_path, 'val_texts.json'))
    test_texts, _ = load_raw_texts(
        os.path.join(dataset_path, 'test_texts.json'))

    if max_instances == -1:
        max_instances = max_train_instances

    print(f"The maximum number of instances is {max_instances}")

    # tokenizer train texts
    train_tokenized = tokenize(tokenizer, train_texts)
    val_tokenized = tokenize(tokenizer, val_texts)
    test_tokenized = tokenize(tokenizer, test_texts)

    # Truncate and pad
    padding_token = "[PAD]".lower()
    train_tokenized = truncate_pad(
        train_tokenized, max_instances=max_instances, max_tokens=max_tokens, padding_token=padding_token)
    val_tokenized = truncate_pad(val_tokenized, max_instances=max_instances,
                                 max_tokens=max_tokens, padding_token=padding_token)
    test_tokenized = truncate_pad(
        test_tokenized, max_instances=max_instances, max_tokens=max_tokens, padding_token=padding_token)

    # filter vocabulary to keep only the tokens that actually occur in the training set
    filtered_vocab, filtered_embed = filter_vocab(
        vocab, embed, train_tokenized)
    pad_token_id = filtered_vocab["[PAD]".lower()]

    # build train input features
    train_input_ids = torch.IntTensor(
        convert_tokens_to_ids(filtered_vocab, train_tokenized))
    val_input_ids = torch.IntTensor(
        convert_tokens_to_ids(filtered_vocab, val_tokenized))
    test_input_ids = torch.IntTensor(
        convert_tokens_to_ids(filtered_vocab, test_tokenized))

    # save vocab and embeddings
    with open(os.path.join(args.output_dir, "vocab.json"), "w+") as f:
        f.write(json.dumps(filtered_vocab))
    np.save(os.path.join(args.output_dir, "vectors.npy"), filtered_embed)

    # save train and test input ids to disk
    with open(os.path.join(dataset_path, 'train_labels.txt'), "r") as f:
        torch.save({
            'input-ids': train_input_ids,
            'labels': [labels.strip().split() for labels in f.readlines()]
        }, os.path.join(args.output_dir, "train_data.pkl")
        )
    with open(os.path.join(dataset_path, 'val_labels.txt'), "r") as f:
        torch.save({
            'input-ids': val_input_ids,
            'labels': [labels.strip().split() for labels in f.readlines()]
        }, os.path.join(args.output_dir, "val_data.pkl")
        )
    with open(os.path.join(dataset_path, 'test_labels.txt'), "r") as f:
        torch.save({
            'input-ids': test_input_ids,
            'labels': [labels.strip().split() for labels in f.readlines()]
        }, os.path.join(args.output_dir, "test_data.pkl")
        )

        print(
            f"Preprocessing completed. Elapsed time: {str(timedelta(seconds=time.time() - t0)).split('.', 2)[0]}")
