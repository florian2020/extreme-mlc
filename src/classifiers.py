from xmlc.modules import (
    MLP,
    IntraBagAttentionClassifier,
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
    """ Combination of a Sentence-Transformer-encoder and a simple attention-based
        Multi-label Classifier Module
    """

    def __init__(self,
                 num_labels: int,
                 model_params,
                 padding_idx,
                 emb_init
                 ) -> None:
        # initialize module
        super(Classifier, self).__init__()

        self.encoder_type = model_params['encoder']['type']

        if self.encoder_type == 'sentence-transformer':

            sentence_transformer = SentenceTransformer(
                model_params['encoder']['name'])

            # create lstm encoder
            self.enc = SentenceTransformerEncoder(
                sent_transformer=sentence_transformer
            )

            classifier_input_dim = model_params['encoder']['output_dim']

        elif self.encoder_type == 'lstm-mil':
            self.enc = LSTMSentenceEncoder(
                vocab_size=emb_init.shape[0],
                embed_size=emb_init.shape[1],
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
        }[model_params['attention']['type']]()

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
        elif model_params['classifier']['type'] == 'intra-bag':
            self.cls = IntraBagAttentionClassifier(
                hidden_size=classifier_input_dim,
                num_labels=num_labels,
                attention=attention_module
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
        x = self.enc(input_ids, input_mask)

        if self.encoder_type in {'lstm'}:
            classifier_mask = input_mask
        elif self.encoder_type in {'sentence-transformer'}:
            classifier_mask = instances_mask

        return self.cls(x, classifier_mask, candidates)


class ClassifierFactory(object):
    # Creates a model with sentence transformer as encoder, attention Layer and a
    # succeding linear layer

    def __init__(self, model_params, padding_idx,
                 emb_init
                 ) -> None:

        self.model_params = model_params
        self.padding_idx = padding_idx
        self.emb_init = emb_init

    def create(self, num_labels: int) -> Classifier:

        return Classifier(
            num_labels=num_labels,
            model_params=self.model_params,
            padding_idx=self.padding_idx,
            emb_init=self.emb_init
        )

    def __call__(self, num_labels: int) -> Classifier:
        return self.create(num_labels)

    @ staticmethod
    def from_params(model_params, padding_idx,
                    emb_init):
        return ClassifierFactory(model_params, padding_idx,
                                 emb_init)


# class DocClassifier(nn.Module):
#     """ Combination of a LSTM-encoder and a simple attention-based
#         Multi-label Classifier Module
#     """

#     def __init__(self,
#                  num_labels: int,
#                  # lstm
#                  hidden_size: int,
#                  num_layers: int,
#                  emb_init: np.ndarray,
#                  padding_idx: int,
#                  dropout: float,
#                  # attention module
#                  attention: nn.Module,
#                  # classifier module
#                  mlp: MLP
#                  ) -> None:
#         # initialize module
#         super(DocClassifier, self).__init__()
#         # create lstm encoder
#         self.enc = LSTMEncoder(
#             vocab_size=emb_init.shape[0],
#             embed_size=emb_init.shape[1],
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             padding_idx=padding_idx,
#             emb_init=torch.from_numpy(emb_init).float(),
#             dropout=dropout
#         )
#         # create label-attention classifier
#         self.cls = LabelAttentionClassifierMLP(
#             hidden_size=hidden_size * 2,  # x2 because lstm is bidirectional
#             num_labels=num_labels,
#             attention=attention,
#             mlp=mlp
#         )

#         print("Total encoder parameters: ", sum(p.numel()
#               for p in self.enc.parameters()))
#         print("Trainable encoder parameters: ", sum(p.numel()
#                                                     for p in self.enc.parameters() if p.requires_grad == True))

#         print("Total classfifier parameters: ", sum(p.numel()
#               for p in self.cls.parameters()))
#         print("Trainable classifier parameters: ", sum(p.numel()
#                                                        for p in self.cls.parameters() if p.requires_grad == True))

#     def forward(self, input_ids, input_mask, candidates=None):
#         # apply encoder and pass output through classifer
#         x = self.enc(input_ids, input_mask)
#         return self.cls(x, input_mask, candidates=candidates)


# class DocClassifierFactory(object):

#     def __init__(self,
#                  encoder_hidden_size: int,
#                  encoder_num_layers: int,
#                  attention_type: str,
#                  mlp_hidden_layers: list,
#                  mlp_bias: bool,
#                  mlp_activation: str,
#                  padding_idx: int,
#                  dropout: float,
#                  emb_init: np.ndarray,
#                  ) -> None:

#         # build classifier keyword-arguments
#         self.cls_kwargs = dict(
#             hidden_size=encoder_hidden_size,
#             num_layers=encoder_num_layers,
#             emb_init=emb_init,
#             padding_idx=padding_idx,
#             dropout=dropout
#         )

#         # get attention type
#         self.attention_module = {
#             'softmax-attention': SoftmaxAttention,
#             'multi-head-attention': MultiHeadAttention
#         }[attention_type]

#         # get classifier type
#         self.mlp_layers = [
#             encoder_hidden_size * 2,
#             *mlp_hidden_layers,
#             1
#         ]
#         self.mlp_kwargs = dict(
#             bias=mlp_bias,
#             act_fn={
#                 'relu': torch.relu
#             }[mlp_activation]
#         )

#     def create(self, num_labels: int) -> ProbabilisticLabelTree:

#         # create attention module
#         attention = self.attention_module()
#         # create multi-layer perceptron
#         mlp = MLP(*self.mlp_layers, **self.mlp_kwargs)
#         # create classifier
#         return DocClassifier(
#             num_labels=num_labels,
#             **self.cls_kwargs,
#             attention=attention,
#             mlp=mlp
#         )

#     def __call__(self, num_labels: int) -> ProbabilisticLabelTree:
#         return self.create(num_labels)

#     @ staticmethod
#     def from_params(params, padding_idx: int, emb_init: np.ndarray):
#         return DocClassifierFactory(
#             encoder_hidden_size=params['encoder']['output_dim'],
#             encoder_num_layers=params['encoder']['num_layers'],
#             attention_type=params['attention']['type'],
#             mlp_hidden_layers=params['classifier']['hidden_layers'],
#             mlp_bias=params['classifier']['bias'],
#             mlp_activation=params['classifier']['activation'],
#             dropout=params['encoder']['dropout'],
#             padding_idx=padding_idx,
#             emb_init=emb_init
#         )
