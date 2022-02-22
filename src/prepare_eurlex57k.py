import os
import json
import re
import numpy as np
import gensim
from collections import Counter


def gensim_word_emb_to_vectors_and_vocab(path):
    glove = gensim.KeyedVectors.load(path)
    vocab = np.array([word for word in glove.index_to_key])
    vectors = np.array([glove[word] for word in glove.index_to_key])
    np.save('./lab_iais/data/word_embeddings/glove_attentionXML/vocab.npy', vocab)
    np.save('./lab_iais/data/word_embeddings/glove_attentionXML/vectors.npy', vectors)


def eurlex57k_to_extreme_milmlc_format(source_path, target_path, min_example_per_label):

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    train_text_rows, train_label_rows, all_train_labels = load_eurlex57k_instances(
        source_path, "train")
    dev_text_rows, dev_label_rows, all_dev_labels = load_eurlex57k_instances(
        source_path, "dev")
    test_text_rows, test_label_rows, all_test_labels = load_eurlex57k_instances(
        source_path, "test")

    labels_to_keep = find_frequent_labels(
        train_label_rows, min_example_per_label)

    train_text_rows, train_label_rows = remove_unfrequent_labels(
        train_text_rows, train_label_rows, labels_to_keep)

    dev_text_rows, dev_label_rows = remove_unfrequent_labels(
        dev_text_rows, dev_label_rows, labels_to_keep)

    test_text_rows, test_label_rows = remove_unfrequent_labels(
        test_text_rows, test_label_rows, labels_to_keep)

    train_label_rows = [" ".join(label_row) for label_row in train_label_rows]
    dev_label_rows = [" ".join(label_row) for label_row in dev_label_rows]
    test_label_rows = [" ".join(label_row) for label_row in test_label_rows]

    with open(os.path.join(target_path, f"labels_vocab.txt"), 'w') as fo:
        for label in labels_to_keep:
            fo.write(label)
            fo.write('\n')

    with open(os.path.join(target_path, f"test_texts.json"), 'w') as fo:
        for text in test_text_rows:
            fo.write(json.dumps(text))
            fo.write('\n')
    with open(os.path.join(target_path, f"test_labels.txt"), 'w') as fo:
        for text in test_label_rows:
            fo.write(text)
            fo.write('\n')

    with open(os.path.join(target_path, f"val_texts.json"), 'w') as fo:
        for text in dev_text_rows:
            fo.write(json.dumps(text))
            fo.write('\n')
    with open(os.path.join(target_path, f"val_labels.txt"), 'w') as fo:
        for text in dev_label_rows:
            fo.write(text)
            fo.write('\n')

    with open(os.path.join(target_path, f"train_texts.json"), 'w') as fo:
        for text in train_text_rows:
            fo.write(json.dumps(text))
            fo.write('\n')

    with open(os.path.join(target_path, f"train_labels.txt"), 'w') as fo:
        for text in train_label_rows:
            fo.write(text)
            fo.write('\n')


def eurlex57k_to_extreme_mlc_format(source_path, target_path, min_example_per_label):

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    train_text_rows, train_label_rows, all_train_labels = load_eurlex57k_texts(
        source_path, "train")
    dev_text_rows, dev_label_rows, all_dev_labels = load_eurlex57k_texts(
        source_path, "dev")
    test_text_rows, test_label_rows, all_test_labels = load_eurlex57k_texts(
        source_path, "test")

    labels_to_keep = find_frequent_labels(
        train_label_rows, min_example_per_label)

    train_text_rows, train_label_rows = remove_unfrequent_labels(
        train_text_rows, train_label_rows, labels_to_keep)

    dev_text_rows, dev_label_rows = remove_unfrequent_labels(
        dev_text_rows, dev_label_rows, labels_to_keep)

    test_text_rows, test_label_rows = remove_unfrequent_labels(
        test_text_rows, test_label_rows, labels_to_keep)

    train_label_rows = [" ".join(label_row) for label_row in train_label_rows]
    dev_label_rows = [" ".join(label_row) for label_row in dev_label_rows]
    test_label_rows = [" ".join(label_row) for label_row in test_label_rows]

    with open(os.path.join(target_path, f"labels_vocab.txt"), 'w') as fo:
        for label in labels_to_keep:
            fo.write(label)
            fo.write('\n')

    with open(os.path.join(target_path, f"test_texts.txt"), 'w') as fo:
        for text in test_text_rows:
            fo.write(text)
            fo.write('\n')
    with open(os.path.join(target_path, f"test_labels.txt"), 'w') as fo:
        for text in test_label_rows:
            fo.write(text)
            fo.write('\n')

    with open(os.path.join(target_path, f"val_texts.txt"), 'w') as fo:
        for text in dev_text_rows:
            fo.write(text)
            fo.write('\n')
    with open(os.path.join(target_path, f"val_labels.txt"), 'w') as fo:
        for text in dev_label_rows:
            fo.write(text)
            fo.write('\n')

    with open(os.path.join(target_path, f"train_texts.txt"), 'w') as fo:
        for text in train_text_rows:
            fo.write(text)
            fo.write('\n')

    with open(os.path.join(target_path, f"train_labels.txt"), 'w') as fo:
        for text in train_label_rows:
            fo.write(text)
            fo.write('\n')


def load_eurlex57k_texts(path, split_name):
    p = os.path.join(path, split_name)

    with open(os.path.join(path, 'EURLEX57K.json'), 'r') as f:
        label_dict = json.load(f)

    text_rows = []
    label_rows = []
    all_labels = []

    for filename in os.listdir(p):

        with open(os.path.join(p, filename), encoding='utf-8') as file:
            data_item = json.load(file)

            assert(type(data_item['attachments']) == str)
            assert(type(data_item['main_body']) == list)

            sections = []
            sections.append(data_item['header'])
            sections.append(data_item['recitals'])
            sections.extend([section
                            for section in data_item['main_body']])
            sections.append(data_item['attachments'])

            labels = [label_dict[concept_id]['label'].lower().replace(" ", "_")
                      for concept_id in data_item['concepts']]

            if len(labels) > 0:
                all_labels.extend(labels)
                text = clean_text(" ".join(sections))
                text_rows.append(text)
                label_rows.append(labels)

    all_labels = set(all_labels)

    return text_rows, label_rows, all_labels


def load_eurlex57k_instances(path, split_name):
    p = os.path.join(path, split_name)

    with open(os.path.join(path, 'EURLEX57K.json'), 'r') as f:
        label_dict = json.load(f)

    texts = []
    labels = []
    all_labels = []

    for filename in os.listdir(p):

        with open(os.path.join(p, filename), encoding='utf-8') as file:
            data_item = json.load(file)

            assert(type(data_item['attachments']) == str)
            assert(type(data_item['main_body']) == list)

            sections = []
            sections.append(data_item['header'])
            sections.append(data_item['recitals'])
            sections.extend([section
                            for section in data_item['main_body']])
            sections.append(data_item['attachments'])

            instance_labels = [label_dict[concept_id]['label'].lower().replace(" ", "_")
                               for concept_id in data_item['concepts']]

            if len(instance_labels) > 0:
                all_labels.extend(instance_labels)
                texts.append(sections)
                labels.append(instance_labels)

    all_labels = set(all_labels)

    return texts, labels, all_labels


def clean_text(text):
    return re.sub(' *[\W0-9]+ *', ' ', text).lower()
    # return text.lower()


def find_frequent_labels(train_label_rows, min_example_per_label):

    label_freq = Counter(
        [label for label_row in train_label_rows for label in label_row])

    labels_to_keep = {
        label for label, freq in label_freq.items() if freq >= min_example_per_label}

    print(
        f"Number of labels that appear less than {min_example_per_label} times in the training dataset: {len(label_freq)-len(labels_to_keep)}")
    print(
        f"Number of remaining labels: {len(labels_to_keep)}")

    return labels_to_keep


def remove_unfrequent_labels(text_rows, label_rows, labels_to_keep):

    new_label_rows = []
    docs_to_delete = set()
    num_labels_per_doc = []

    for i, label_row in enumerate(label_rows):
        new_label_row = []
        for label in label_row:
            if label in labels_to_keep:
                new_label_row.append(label)

        if len(new_label_row) > 0:
            num_labels_per_doc.append(len(new_label_row))
            new_label_rows.append(new_label_row)
        else:
            docs_to_delete.add(i)

    text_rows = [text_row for j, text_row in enumerate(
        text_rows) if j not in docs_to_delete]

    print(f"\nNumber of removed examples: {len(docs_to_delete)}")
    print(
        f"Number of remaining examples: {len(label_rows)-len(docs_to_delete)}")
    print(
        f"Average number of labels per remaining example: {np.array(num_labels_per_doc).mean()}")

    return text_rows, new_label_rows


if __name__ == '__main__':
    eurlex57k_to_extreme_mlc_format(source_path="./data/datasets/EURLEX57K_original/",
                                    target_path="./data/datasets/EURLEX57K_full_min10/",
                                    min_example_per_label=10)

    # eurlex57k_to_extreme_milmlc_format(source_path="./data/datasets/EURLEX57K_original/",
    #                                    target_path="./data/datasets/EURLEX57K_full_mil_min10/",
    #                                    min_example_per_label=10)
