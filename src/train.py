import os
import yaml
import json
import torch
import pickle
import time
from datetime import timedelta
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from treelib import Tree
from typing import Dict, Tuple, List, Any, Callable
# import attention-xml
from xmlc.trainer import (
    LevelTrainerModule,
    InputsAndLabels
)
from xmlc.metrics import *
from xmlc.plt import ProbabilisticLabelTree
from xmlc.dataset import NamedTensorDataset
from xmlc.utils import build_sparse_tensor
from classifiers import ClassifierFactory
from logger import LogHistory
from transformers import AutoTokenizer


def load_data_mil(
    data_path: str,
    padding_idx: int
) -> Tuple[InputsAndLabels, InputsAndLabels]:
    """
    Load pickled data of the shape: (num_examples, num_instances, num_tokens).
    """

    # Load data with multiple instances
    # load input ids and compute mask
    data = torch.load(data_path)
    input_ids, labels = data['input-ids'], data['labels']
    input_mask = (input_ids != padding_idx)

    # Mask needed for attention layer i.e. mask instances
    instances_mask = (input_mask.sum(dim=-1) !=
                      padding_idx*input_mask.shape[-1])
    # build data
    return InputsAndLabels(
        inputs=NamedTensorDataset(
            input_ids=input_ids, input_mask=input_mask, instances_mask=instances_mask),
        labels=labels
    )


def load_data(
    data_path: str,
    padding_idx: int
) -> Tuple[InputsAndLabels, InputsAndLabels]:
    """
    Load pickled data of the shape: (num_examples, num_tokens)
    """
    # Load data with only one instance
    # load input ids and compute mask
    data = torch.load(data_path)
    input_ids, labels = data['input-ids'], data['labels']
    input_mask = (input_ids != padding_idx)
    # build data
    return InputsAndLabels(
        inputs=NamedTensorDataset(input_ids=input_ids, input_mask=input_mask),
        labels=labels
    )


def concatenate_experiment_params(args):
    """
    Read all params files which have an impact on the trained model and
    write them into a single config file for clear documentation of the
    outcome of an experiment
    """

    # Get Paths
    label_tree_dir = os.path.dirname(args.label_tree)
    preprocessed_data_path = os.path.dirname(args.train_data)

    # Read preprocessing arguments to get padding id of tokenizer
    with open(f"{preprocessed_data_path}/params_preprocess.yaml", "r") as f:
        params_preprocess_string = f.read()

    # Read tree params
    with open(f"{label_tree_dir}/params_tree.yaml", "r") as f:
        params_tree_string = f.read()

    # Read training parameters
    with open(f"{args.output_dir}/params_train.yaml", "r") as f:
        params_string = f.read()

    # Write preprocessing and training parameters into same file
    with open(f"{args.output_dir}/params_experiment.yaml", 'w') as f:
        f.writelines(params_preprocess_string + '\n\n')
        f.writelines(params_tree_string + '\n\n')
        f.writelines(params_string)


def compute_metrics(
    preds: torch.LongTensor,
    targets: torch.LongTensor
):
    return {
        # first only the metrics that will be logged
        # in the progress bar
        "F3": f1_score(preds, targets, k=3),
        "nDCG3": ndcg(preds, targets, k=3),
    }, {
        # now all additional metrics that will be logged
        # to the logger of choice
        # precision @ k
        "P1": precision(preds, targets, k=1),
        "P3": precision(preds, targets, k=3),
        "P5": precision(preds, targets, k=5),
        # recall @ k
        "R1": recall(preds, targets, k=1),
        "R3": recall(preds, targets, k=3),
        "R5": recall(preds, targets, k=5),
        # f-score @ k
        "F1": f1_score(preds, targets, k=1),
        "F5": f1_score(preds, targets, k=5),
        # ndcg @ k
        "nDCG1": ndcg(preds, targets, k=1),
        "nDCG5": ndcg(preds, targets, k=5),
        # coverage @ k
        "C1": coverage(preds, targets, k=1),
        "C3": coverage(preds, targets, k=3),
        "C5": coverage(preds, targets, k=5),
        # hits @ k
        "H1": hits(preds, targets, k=1),
        "H3": hits(preds, targets, k=3),
        "H5": hits(preds, targets, k=5),
    }


def train_end2end(
    model: ProbabilisticLabelTree,
    train_data: InputsAndLabels,
    val_data: InputsAndLabels,
    params: Dict[str, Any]
):
    raise NotImplementedError()


def train_levelwise(
    tree: Tree,
    model: ProbabilisticLabelTree,
    train_data: InputsAndLabels,
    val_data: InputsAndLabels,
    params: Dict[str, Any],
    bag_group_size: int,
    output_dir: str
) -> LogHistory:
    # use gpu if possible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # train each level of the label-tree one after another
    for level in range(model.num_levels - 1):

        print("-" * 50)
        print("-" * 17 + ("Training Level %i" % level) + "-" * 17)
        print("-" * 50)

        # create logger
        # logger = pl.loggers.TensorBoardLogger("logs", name="attention-xml", sub_dir="level-%i" % level)
        logger = None  # pl.loggers.MLFlowLogger()
        history = LogHistory()
        # create the trainer module
        trainer_module = LevelTrainerModule(
            level=level,
            tree=tree,
            model=model,
            train_data=train_data,
            val_data=val_data,
            num_candidates=params['num_candidates'],
            topk=params['topk'],
            train_batch_size=params['train_batch_size'],
            val_batch_size=params['eval_batch_size'],
            metrics=compute_metrics,
            lr_encoder=params['lr_encoder'],
            lr_classifier=params['lr_classifier'],
            bag_group_size=bag_group_size
        )
        # create the trainer
        trainer = pl.Trainer(
            gpus=1,
            auto_select_gpus=True,
            max_steps=params['num_steps'],
            # val_check_interval=params['eval_interval'],
            num_sanity_val_steps=0,
            logger=[history],
            enable_checkpointing=False,
            callbacks=[
                pl.callbacks.early_stopping.EarlyStopping(
                    monitor="nDCG3",
                    patience=500,
                    mode="max",
                    verbose=False
                )
            ]
        )
        # train the model
        trainer.fit(trainer_module)

    # return the log-history instance of very last level
    return history


if __name__ == '__main__':

    print("Start training")
    t0 = time.time()

    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser("Train a model on the preprocessed data.")
    parser.add_argument("--model-type", type=str,
                        help="Choose the model type to train")
    parser.add_argument("--train-data", type=str,
                        help="Path to the preprocessed train data.")
    parser.add_argument("--val-data", type=str,
                        help="Path to the preprocessed validation data.")
    parser.add_argument("--vocab", type=str, help="Path to the vocab file.")
    parser.add_argument(
        "--embed", type=str, help="Path to the initial (pretrained) embedding vector file.")
    parser.add_argument("--label-tree", type=str,
                        help="Path to the label tree file.")
    parser.add_argument("--output-dir", type=str, help="Output directory.")

    # parse arguments
    args = parser.parse_args()

    # Write configuration files of preprocessing, tree building and training into the same file
    concatenate_experiment_params(args)

    # load parameters
    with open(f"{args.output_dir}/params_experiment.yaml", "r") as f:
        params = yaml.load(f.read(), Loader=yaml.SafeLoader)

    # load label tree
    label_tree_file = args.label_tree
    with open(label_tree_file, "rb") as f:
        tree = pickle.load(f)

    # Load data depending on the encoder
    if params['model']['encoder']['type'] == 'sentence-transformer':

        assert(params['preprocess']['tokenizer'] ==
               params['model']['encoder']['name'])

        # Get parameters of pretrained sentence transformer
        padding_idx = AutoTokenizer.from_pretrained(
            f"sentence-transformers/{params['preprocess']['tokenizer']}").pad_token_id

        emb_init = None

        # load train and validation data
        train_data = load_data_mil(data_path=args.train_data,
                                   padding_idx=padding_idx)
        val_data = load_data_mil(
            data_path=args.val_data, padding_idx=padding_idx)
        bag_group_size = params['model']['classifier']['bag_group_size']

    elif params['model']['encoder']['type'] == 'lstm-mil':
        with open(args.vocab, "r") as f:
            vocab = json.loads(f.read())
            padding_idx = vocab['[pad]']
            emb_init = np.load(args.embed)

        # load train and validation data
        train_data = load_data_mil(data_path=args.train_data,
                                   padding_idx=padding_idx)
        val_data = load_data_mil(
            data_path=args.val_data, padding_idx=padding_idx)
        bag_group_size = 1

    elif params['model']['encoder']['type'] == 'lstm':
        with open(args.vocab, "r") as f:
            vocab = json.loads(f.read())
            padding_idx = vocab['[pad]']
            emb_init = np.load(args.embed)

        # load train and validation data
        train_data = load_data(data_path=args.train_data,
                               padding_idx=padding_idx)
        val_data = load_data(data_path=args.val_data, padding_idx=padding_idx)
        bag_group_size = 1
    else:
        AssertionError("This is not a supported encoder type.")

    # create the model
    model = ProbabilisticLabelTree(
        tree=tree,
        cls_factory=ClassifierFactory.from_params(
            model_params=params['model'],
            padding_idx=padding_idx,
            emb_init=emb_init)
    )

    if "continue_training" in params['trainer']:
        model_path = params['trainer']['continue_training']
        print("Try to load model from: ", model_path)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)

    # check which training regime to use
    training_regime = {
        "levelwise": train_levelwise,
    }[params['trainer']['regime']]

    # train model
    history = training_regime(
        tree=tree,
        model=model,
        train_data=train_data,
        val_data=val_data,
        params=params['trainer'],
        bag_group_size=bag_group_size,
        output_dir=args.output_dir
    )

    # save model to disk
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.bin"))

    # save final metrics
    final_scores = {name: hist.values[-1]
                    for name, hist in history.history.items()}
    with open(os.path.join(args.output_dir, "validation-scores.json"), "w+") as f:
        f.write(json.dumps(final_scores))

    # plot metrics
    fig, axes = plt.subplots(7, 1, figsize=(12, 28), sharex=True)
    # plot losses
    axes[0].plot(history['train_loss'].steps,
                 history['train_loss'].values, label="train")
    axes[0].plot(history['val_loss'].steps,
                 history['val_loss'].values, label="validation")
    axes[0].set(
        title="Train and Validation Loss",
        xlabel="Loss",
        ylabel="Global Step"
    )
    axes[0].legend()
    axes[0].grid()
    # plot metrics
    for ax, name in zip(axes[1:], ["nDCG", "P", "R", "F", "C", "H"]):
        # plot ndcg
        ax.plot(history['%s1' % name].steps,
                history['%s1' % name].values, label="$k=1$")
        ax.plot(history['%s3' % name].steps,
                history['%s3' % name].values, label="$k=3$")
        ax.plot(history['%s5' % name].steps,
                history['%s5' % name].values, label="$k=5$")
        ax.set(
            title="%s @ k" % name,
            ylabel=name,
            xlabel="Global Step"
        )
        ax.legend()
        ax.grid()
    # save and show
    fig.savefig(os.path.join(args.output_dir, "validation-metrics.pdf"))

    print(
        f"Training completed. Elapsed time: {str(timedelta(seconds=time.time() - t0)).split('.', 2)[0]}")
