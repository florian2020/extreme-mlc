# How to run this repository

1. Download the data here as described here: https://github.com/iliaschalkidis/lmtc-eurlex57k
2. Run prepare_eurlex57k.py (eurlex57k_to_extreme_milmlc_format for Multi-instance-learning)
3. Put a file named params_preprocess.yaml with the preprocessing configs in the folder where you want to store the preprocessed data
4. Put a file named params_tree.yaml with the tree building configs in the folder where you want to store the label tree
5. Put a file named params_training.yaml with the training configs into the folder where your training results should be stored
6. Run the script eurlex57k_mil.sh while specifying preprocessed_dir and out_dir to run the whole code.


# Best models


|Model|ndcg@1|ndcg@3|ndcg@5|
|---|---|---|---|
|AttentionXML original implementation (val set 2000)|93.2|86.9|83.2|
|AttentionXML original implementation (val set 6000)|92.7|86.4|83.0|
|AttentionXML own implementation (emb trainable) |92.0|85.7|82.1|
|AttentionXML own implementation (emb not trainable) |91.9|85.4|81.8|
|LSTM Encoder full text with intra bag|88.4|81.1|77.1|
|all-MiniLm-L6-v2 (lr 1e-5) and intra-bag (lr 1e-3)|91.7|85.3|81.1|
|all-MiniLm-L6-v2 (lr 1e-5) and mlp (lr 1e-3)|90.3|83.8|80.1|
|LSTM Sentence Encoder Mean and MLP (lr 1e-3)|90.0|83.2|79.2|
|LSTM Sentence Encoder Mean and intra bag (lr 1e-3)|89.3|80.6|75.5|
