from xmlc.modules import (
    MLP,
    BagAttentionClassifier,
    SoftmaxAttention,
    MultiHeadAttention,
    LabelAttentionClassifierMLP
)
from xmlc.encoders import (LSTMSentenceEncoder,
                           LSTMEncoder,
                           SentenceTransformerEncoder
                           )
from sentence_transformers import SentenceTransformer

from typing import Iterator

from torch.nn.parameter import Parameter
import torch.nn as nn
import torch


class Classifier(nn.Module):
    """ Combination of an encoder and a simple attention-based
        Multi-label Classifier Module
    """

    def __init__(self,
                 num_labels: int,
                 model_params,
                 padding_idx,
                 emb_init,
                 record_attention_weights: bool,
                 ) -> None:
        # initialize module
        super(Classifier, self).__init__()

        self.encoder_type = model_params['encoder']['type']

        if self.encoder_type == 'sentence-transformer':

            # create lstm encoder
            self.enc = SentenceTransformerEncoder(
                sent_transformer_name=model_params['encoder']['name'],
                dropout=model_params['encoder']['dropout'],
                normalize_sentence_emb=model_params['encoder']['normalize_sentences']
            )

            classifier_input_dim = model_params['encoder']['output_dim']

        elif self.encoder_type == 'lstm-mil':
            self.enc = LSTMSentenceEncoder(
                hidden_size=model_params['encoder']['output_dim'],
                num_layers=model_params['encoder']['num_layers'],
                padding_idx=padding_idx,
                emb_init=torch.from_numpy(emb_init).float(),
                dropout=model_params['encoder']['dropout']
            )

            classifier_input_dim = 2 * model_params['encoder']['output_dim']

        elif self.encoder_type == 'lstm':

            # create lstm encoder
            self.enc = LSTMEncoder(
                vocab_size=emb_init.shape[0],
                embed_size=emb_init.shape[1],
                hidden_size=model_params['encoder']['output_dim'],
                num_layers=model_params['encoder']['num_layers'],
                padding_idx=padding_idx,
                emb_init=torch.from_numpy(emb_init).float(),
                dropout=model_params['encoder']['dropout']
            )

            classifier_input_dim = 2 * model_params['encoder']['output_dim']

        # Create and initialize attention module
        attention_module = {
            'softmax-attention': SoftmaxAttention,
            'multi-head-attention': MultiHeadAttention
        }[model_params['attention']['type']](record_attention_weights=record_attention_weights)

        if model_params['classifier']['type'] == 'mlp':
            self.mlp_layers = [
                classifier_input_dim,
                *model_params['classifier']['hidden_layers'],
                1
            ]
            self.mlp_kwargs = dict(
                bias=model_params['classifier']['bias'],
                act_fn={
                    'relu': torch.relu
                }[model_params['classifier']['activation']]
            )

            # create label-attention classifier
            self.cls = LabelAttentionClassifierMLP(
                hidden_size=classifier_input_dim,
                num_labels=num_labels,
                attention=attention_module,
                mlp=MLP(*self.mlp_layers, **self.mlp_kwargs)
            )
        elif model_params['classifier']['type'] == 'bag':

            self.cls = BagAttentionClassifier(
                encoder_hidden_size=classifier_input_dim,
                num_labels=num_labels,
                attention=attention_module,
                bag_group_size=model_params['classifier']['bag_group_size'],
                normalize_bags=model_params['classifier']['normalize_bags'],
                normalize_labels=model_params['classifier']['normalize_labels']
            )
        else:
            AssertionError("Your chosen classifier type is not supported")

        print("Total model parameters: ", sum(p.numel()
              for p in self.parameters()))
        print("Trainable model parameters: ", sum(p.numel()
                                                  for p in self.parameters() if p.requires_grad == True))

        print("Total encoder parameters: ", sum(p.numel()
              for p in self.enc.parameters()))
        print("Trainable encoder parameters: ", sum(p.numel()
                                                    for p in self.enc.parameters() if p.requires_grad == True))

        print("Total classfifier parameters: ", sum(p.numel()
              for p in self.cls.parameters()))
        print("Trainable classifier parameters: ", sum(p.numel()
                                                       for p in self.cls.parameters() if p.requires_grad == True))

    def forward(self, input_ids, input_mask, instances_mask=None, candidates=None):
        # apply encoder and pass output through classifer
        x = self.enc(input_ids, input_mask, instances_mask)

        if self.encoder_type in {'lstm'}:
            classifier_mask = input_mask
        elif self.encoder_type in {'sentence-transformer', 'lstm-mil'}:
            classifier_mask = instances_mask

        return self.cls(x, classifier_mask, candidates)


class ClassifierFactory(object):
    # Creates a model with sentence transformer as encoder, attention Layer and a
    # succeding linear layer

    def __init__(self, model_params,
                 padding_idx,
                 emb_init,
                 record_attention_weights: bool = False
                 ) -> None:

        self.model_params = model_params
        self.padding_idx = padding_idx
        self.emb_init = emb_init
        self.record_attention_weights = record_attention_weights

    def create(self, num_labels: int) -> Classifier:

        return Classifier(
            num_labels=num_labels,
            model_params=self.model_params,
            padding_idx=self.padding_idx,
            emb_init=self.emb_init,
            record_attention_weights=self.record_attention_weights
        )

    def __call__(self, num_labels: int) -> Classifier:
        return self.create(num_labels)

    @ staticmethod
    def from_params(model_params, padding_idx,
                    emb_init, record_attention_weights=False):
        return ClassifierFactory(model_params, padding_idx,
                                 emb_init, record_attention_weights)
