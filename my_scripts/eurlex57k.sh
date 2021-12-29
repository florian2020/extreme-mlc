#!/usr/bin/env bash

data_dir=./data/datasets/EURLEX57K_full
preprocessed_dir=./data/prepcrocessed/EURLEX57K_full

# python src/preprocess.py \
# --train-texts $data_dir/train_texts.txt \
# --train-labels $data_dir/train_labels.txt \
# --val-texts $data_dir/val_texts.txt \
# --val-labels $data_dir/val_labels.txt \
# --test-texts $data_dir/test_texts.txt \
# --test-labels $data_dir/test_labels.txt \
# --tokenizer Spacy \
# --max-length 800 \
# --pretrained-vocab ./data/word_embeddings/glove_attentionXML/glove_vocab.npy \
# --pretrained-emb ./data/word_embeddings/glove_attentionXML/glove_all_vectors.npy \
# --output-dir $preprocessed_dir

# python src/build_label_tree.py \
# --labels-file $data_dir/labels_vocab.txt \
# --group-id-chars 0 \
# --output-file $preprocessed_dir/label_tree.pkl

python src/train.py \
--train-data $preprocessed_dir/train_data.pkl \
--val-data $preprocessed_dir/val_data.pkl \
--vocab $preprocessed_dir/vocab.json \
--embed $preprocessed_dir/vectors.npy \
--label-tree $preprocessed_dir/label_tree.pkl \
--output-dir ./results/EURLEX57K_full