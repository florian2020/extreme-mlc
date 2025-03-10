#!/usr/bin/env bash

data_dir=./data/datasets/EURLEX57K_full_min10
preprocessed_dir=./data/preprocessed/EURLEX57K_full_eng
model_output_dir=./results/EURLEX57K_full_intrabag_2_cont

python src/preprocess_lstm_doc.py \
--train-texts $data_dir/train_texts.txt \
--train-labels $data_dir/train_labels.txt \
--val-texts $data_dir/val_texts.txt \
--val-labels $data_dir/val_labels.txt \
--test-texts $data_dir/test_texts.txt \
--test-labels $data_dir/test_labels.txt \
--tokenizer Spacy_en \
--max-tokens 800 \
--pretrained-vocab ./data/word_embeddings/glove_attentionXML/glove_vocab.npy \
--pretrained-emb ./data/word_embeddings/glove_attentionXML/glove_all_vectors.npy \
--output-dir $preprocessed_dir

python src/build_label_tree.py \
--output-file $preprocessed_dir/flat_label_tree_min10.pkl

python src/train.py \
--train-data $preprocessed_dir/train_data.pkl \
--val-data $preprocessed_dir/val_data.pkl \
--vocab $preprocessed_dir/vocab.json \
--embed $preprocessed_dir/vectors.npy \
--label-tree $preprocessed_dir/flat_label_tree.pkl \
--output-dir $model_output_dir

python src/predict.py \
--test-data $preprocessed_dir/val_data.pkl \
--model-path $model_output_dir/model.bin \
--label-tree $preprocessed_dir/flat_label_tree.pkl \
--vocab $preprocessed_dir/vocab.json \
--embed $preprocessed_dir/vectors.npy \
--output-dir $model_output_dir

python src/evaluate.py \
--model-output $model_output_dir/predictions.pkl \
--sparse-targets $model_output_dir/targets.pkl \
--output-dir $model_output_dir