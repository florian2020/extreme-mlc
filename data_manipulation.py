import os
import json
import re
import numpy as np
import gensim



def gensim_word_emb_to_vectors_and_vocab(path):
    glove = gensim.KeyedVectors.load(path)
    vocab = np.array([word for word in glove.index_to_key])
    vectors = np.array([glove[word] for word in glove.index_to_key])
    np.save('./lab_iais/data/word_embeddings/glove_attentionXML/vocab.npy',vocab)
    np.save('./lab_iais/data/word_embeddings/glove_attentionXML/vectors.npy',vectors)



def eurlex57k_to_extreme_mlc_format(source_path,target_path):

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    train_text_rows, train_label_rows, all_train_labels = load_eurlex57k(source_path, "train")
    dev_text_rows, dev_label_rows,all_dev_labels = load_eurlex57k(source_path, "dev")
    test_text_rows, test_label_rows,all_test_labels = load_eurlex57k(source_path, "test")


    all_labels = set().union(all_train_labels,all_dev_labels,all_test_labels)

    with open(os.path.join(target_path, f"labels_vocab.txt"), 'w') as fo:
        for label in all_labels:
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


def load_eurlex57k(path, split_name):
    p = os.path.join(path, split_name)

    with open(os.path.join(path, 'EURLEX57K.json'), 'r') as f:
        label_dict = json.load(f)

    text_rows = []
    label_rows = []
    all_labels = []

    for filename in os.listdir(p):

        with open(os.path.join(p, filename), encoding='utf-8') as file:
            data_item = json.load(file)

            sections = []
            sections.append(data_item['header'])
            sections.append(data_item['recitals'])
            sections.extend([section
                            for section in data_item['main_body']])
            sections.append(data_item['attachments'])

            assert(type(data_item['attachments']) == str)
            assert(type(data_item['main_body']) == list)

            labels = [label_dict[concept_id]['label'].lower().replace(" ", "_")
                      for concept_id in data_item['concepts']]

            if len(labels) > 0:
                all_labels.extend(labels)
                text = clean_text(" ".join(sections))
                text_rows.append(text)
                label_rows.append(" ".join(labels))

    all_labels = set(all_labels)

    return text_rows, label_rows,all_labels


def clean_text(text):
    return re.sub(' *[\W0-9]+ *', ' ', text).lower()
    # return text.lower()


if __name__ == '__main__':
    eurlex57k_to_extreme_mlc_format(source_path = "./data/datasets/EURLEX57K_original/",
                                     target_path = "./data/datasets/EURLEX57K_full/")
