1. Download the data here as described here: https://github.com/iliaschalkidis/lmtc-eurlex57k
2. Run prepare_eurlex57k.py (eurlex57k_to_extreme_milmlc_format for Multi-instance-learning)
3. Put a file named params_preprocess.yaml with the preprocessing configs in the folder where you want to store the preprocessed data
4. Put a file named params_tree.yaml with the tree building configs in the folder where you want to store the label tree
5. Put a file named params_training.yaml with the training configs into the folder where your training results should be stored
6. Run the script eurlex57k_mil.sh while specifying preprocessed_dir and out_dir to run the whole code.