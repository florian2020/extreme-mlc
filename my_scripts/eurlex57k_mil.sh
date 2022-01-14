#!/usr/bin/env bash


preprocessed_dir=./data/preprocessed/EURLEX57K_tiny_mil
results_dir=./results/EURLEX57K_tiny_mil


python src/preprocess_mil.py \
--output-dir $preprocessed_dir

python src/build_label_tree.py \
--output-file $preprocessed_dir/flat_label_tree.pkl

python src/train.py \
--train-data $preprocessed_dir/train_data.pkl \
--val-data $preprocessed_dir/val_data.pkl \
--label-tree $preprocessed_dir/flat_label_tree.pkl \
--output-dir $results_dir

python src/predict.py \
--test-data $preprocessed_dir/test_data.pkl \
--model-path $results_dir/model.bin \
--label-tree $preprocessed_dir/flat_label_tree.pkl \
--output-dir $results_dir

python src/evaluate.py \
--model-output $results_dir/predictions.pkl \
--sparse-targets $results_dir/targets.pkl \
--output-dir $results_dir