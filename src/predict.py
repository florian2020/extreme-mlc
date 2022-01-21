import os
import json
import yaml
import pickle
import numpy as np
import torch
import time
from datetime import timedelta
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from xmlc.plt import ProbabilisticLabelTree, PLTOutput
from xmlc.utils import build_sparse_tensor
from xmlc.tree_utils import convert_labels_to_ids
from train import load_data, load_data_mil
from classifiers import ClassifierFactory
from transformers import AutoTokenizer


@torch.no_grad()
def predict(
    model: ProbabilisticLabelTree,
    dataset: Dataset,
    batch_size: int,
    k: int,
    device: str
):
    # set model to evaluation mode and move it to devoce
    model.eval()
    model = model.to(device)
    # create datalaoder
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # prediction loop
    outputs = []
    for inputs in tqdm(loader, desc="Predicting"):
        # apply model and collect all outputs
        inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
        model_out = model(**inputs, topk=k).topk(k=k)
        outputs.append(model_out.cpu())
    # concatenate all model outputs
    return PLTOutput(
        probs=torch.cat([out.probs for out in outputs], dim=0),
        candidates=torch.cat([out.candidates for out in outputs], dim=0),
        mask=torch.cat([out.mask for out in outputs], dim=0)
    )


if __name__ == '__main__':

    print("Start predicting")
    t0 = time.time()

    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Evaluate the model.")
    parser.add_argument("--test-data", type=str,
                        help="Path to the preprocessed test data.")
    parser.add_argument("--model-path", type=str,
                        help="Path to the trained model.")
    parser.add_argument("--model-type", type=str,
                        help="Type of the model which was trained")
    parser.add_argument("--label-tree", type=str,
                        help="Path to the label tree")
    parser.add_argument("--vocab", type=str, help="Path to the vocabulary.")
    parser.add_argument("--embed", type=str,
                        help="Path to the pretrained embedding vectors.")
    parser.add_argument("--output-dir", type=str,
                        help="Path to the output directory.")
    # parse arguments
    args = parser.parse_args()

    # load model parameters
    with open(os.path.join(os.path.dirname(args.model_path), "params_experiment.yaml"), "r") as f:
        params = yaml.load(f.read(), Loader=yaml.SafeLoader)

    # load label tree
    with open(args.label_tree, "rb") as f:
        tree = pickle.load(f)
    num_labels = len(set(n.data.level_index for n in tree.leaves()))

    # Load vocab depending on the encoder
    if params['model']['encoder']['type'] == 'sentence-transformer':

        assert(params['preprocess']['tokenizer'] ==
               params['model']['encoder']['name'])

        # Get parameters of pretrained sentence transformer
        padding_idx = AutoTokenizer.from_pretrained(
            f"sentence-transformers/{params['preprocess']['tokenizer']}").pad_token_id

        emb_init = None

        test_data = load_data_mil(data_path=args.test_data,
                                  padding_idx=padding_idx)

    elif params['model']['encoder']['type'] == 'lstm-mil':
        # load vocabulary
        with open(args.vocab, "r") as f:
            vocab = json.loads(f.read())
            padding_idx = vocab['[pad]']
            emb_init = np.load(args.embed)

        test_data = load_data_mil(data_path=args.test_data,
                                padding_idx=padding_idx)

    elif params['model']['encoder']['type'] == 'lstm':
        # load vocabulary
        with open(args.vocab, "r") as f:
            vocab = json.loads(f.read())
            padding_idx = vocab['[pad]']
            emb_init = np.load(args.embed)

        test_data = load_data(data_path=args.test_data,
                              padding_idx=padding_idx)

    # create the model
    model = ProbabilisticLabelTree(
        tree=tree,
        cls_factory=ClassifierFactory.from_params(
            model_params=params['model'],
            padding_idx=padding_idx,
            emb_init=emb_init
        )
    )

    # load model parameters
    state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    # predict
    output = predict(model,
                     dataset=test_data.inputs,
                     batch_size=params['trainer']['eval_batch_size'],
                     k=params['trainer']['topk'],
                     device='cuda' if torch.cuda.is_available() else 'cpu'
                     )
    # save predictions to disk
    torch.save(output.candidates, os.path.join(
        args.output_dir, "predictions.pkl"))

    # build sparse prediction tensor
    targets = convert_labels_to_ids(tree, test_data.labels)
    max_num_labels = max(map(len, targets))
    targets = [list(t) + [-1] * (max_num_labels - len(t)) for t in targets]
    targets = torch.LongTensor(targets)
    # save sparse targets to disk
    torch.save(targets, os.path.join(args.output_dir, "targets.pkl"))

    print(
        f"Predicting completed. Elapsed time: {str(timedelta(seconds=time.time() - t0)).split('.', 2)[0]}")
