# Summary
This repository integrates the deep multi-instance learning (MIL) components intra-bag and inter-bag attention [1] into our own implementation of AttentionXML [2], which is a model for extreme multi-label text classification (XMTC). We evaluate our proposed model on the EURLEX57K dataset [3]. We provide the code for two kind of models:
- MIL models which need the input texts to be encoded as a list of instances (our proposed model)
- Document level models which act on an entire single text (our implementation of AttentionXML)

# How to run this repository

1. Download the data the EURLEX57K dataset as described here: https://github.com/iliaschalkidis/lmtc-eurlex57k
2. Download the glove word embedding as described here: https://github.com/yourh/AttentionXML
2. Prepare the EURLEX57K dataset by runnning prepare_eurlex57k.py (eurlex57k_to_extreme_milmlc_format for MIL models and eurlex57k_to_extreme_mlc_format for document-level classifcation models)
3. Put a file named params_preprocess.yaml with the preprocessing configs in the folder where you want to store the preprocessed data (see examples data_dummy)
4. Put a file named params_tree.yaml with the tree building configs in the folder where you want to store the label tree (see example data_dummy)
5. Put a file named params_training.yaml with the training configs into the folder where your training results should be stored (see examples results_dummy)
6. Run the script eurlex57k_mil.sh (for MIL models) or eurlex57k_doc.sh (for doc classification models) while specifying preprocessed_dir and out_dir in the script to run the preprocessing, building of label tree, model training, model prediction and evaluation of predictions


# Best models

Performance comparison of different intra-bag models with AttentionXML on theEURLEX57K dataset. We combine the  sentence transformer (ST) 
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) or a BiLSTM (BiLSTM-avg) as instance encoder, the intra-bag attention layer and the biased dot product (dot) or twofully connected layers with ReLu activation function (mlp) to a MIL based classification model. The original AttentionXML model (AttentionXML-mlp) matches the reported results of BERT-BASE [3] and outperforms our best proposed model ST-intra-dot by a tiny margin.


|Model|ndcg@1|ndcg@3|ndcg@5|P@3|P@5|F@1|F@3|F@5|
|---|---|---|---|---|---|---|---|---|
|AttentionXML-mlp |**0.923**|**0.857**|**0.823**|**0.820**|**0.690**|0.340|**0.648**|**0.705**|
|BERT-BASE reported scores [3]|0.922|-|**0.823**|-|0.687|**0.341**|-|0.703|
|ST-intra-dot|0.919|0.855|0.812|0.819|0.676|0.339|0.648|0.691|
|ST-intra-mlp|0.903|0.838|0.801|0.800|0.667|0.335|0.634|0.682|
|BiLSTM-avg-intra-mlp|0.900|0.832|0.792|.792|0.657|0.334|0.629|0.673|
|BiLSTM-avg-intra-dot|0.894|0.806|0.755|0.761|0.619|0.331|0.599|0.631|
|AttentionXML-dot|0.891|0.822|0.782|0.785|0.650|0.328|0.620|0.665|


# Resources

[1] Ye, Zhi-Xiu, and Zhen-Hua Ling. “Distant Supervision Relation Extraction with Intra-Bag and Inter-Bag Attentions.” In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 2810–19. Minneapolis, Minnesota: Association for Computational Linguistics, 2019. ​

[2] You, Ronghui, Zihan Zhang, Ziye Wang, Suyang Dai, Hiroshi Mamitsuka, and Shanfeng Zhu. “AttentionXML: Label Tree-Based Attention-Aware Deep Model for High-Performance Extreme Multi-Label Text Classification.” ArXiv:1811.01727 [Cs], November 4, 2019. http://arxiv.org/abs/1811.01727​

[3] Chalkidis, Ilias, Emmanouil Fergadiotis, Prodromos Malakasiotis, and Ion Androutsopoulos. “Large-Scale Multi-Label Text Classification on EU Legislation.” In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 6314–22. Florence, Italy: Association for Computational Linguistics, 2019. https://doi.org/10.18653/v1/P19-1636.