#!/usr/bin/env bash


preprocessed_dir=./data/preprocessed/EURLEX57K_full_mil_min10_all-MiniLM-L6-v2
preprocessed_dir=./data/preprocessed/EURLEX57K_mil_all-MiniLM-L6-v2
results_dir=./results/EURLEX57K_full_mil_unnormalized_4

python src/preprocess_sentence_transformer.py \
--output-dir $preprocessed_dir

python src/build_label_tree.py \
--output-file $preprocessed_dir/flat_label_tree_min10.pkl

python src/train.py \
--train-data $preprocessed_dir/train_data.pkl \
--val-data $preprocessed_dir/val_data.pkl \
--vocab $preprocessed_dir/vocab.json \
--embed $preprocessed_dir/vectors.npy \
--label-tree $preprocessed_dir/flat_label_tree.pkl \
--output-dir $results_dir

python src/predict.py \
--test-data $preprocessed_dir/test_data.pkl \
--model-path $results_dir/model.bin \
--vocab $preprocessed_dir/vocab.json \
--embed $preprocessed_dir/vectors.npy \
--label-tree $preprocessed_dir/flat_label_tree.pkl \
--output-dir $results_dir \
--record_attention_weights True

python src/evaluate.py \
--model-output $results_dir/predictions.pkl \
--sparse-targets $results_dir/targets.pkl \
--output-dir $results_dir