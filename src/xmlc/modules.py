import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Callable


class MLP(nn.Module):
    """ Multi-Layer Perceptron """

    def __init__(self,
                 *layers: Tuple[int],
                 act_fn: Callable[[Tensor], Tensor] = F.relu,
                 bias: bool = True
                 ) -> None:
        # initialize module
        super(MLP, self).__init__()
        # build all linear layers
        self.layers = nn.ModuleList([
            nn.Linear(n, m, bias=bias)
            for n, m in zip(layers[:-1], layers[1:])
        ])

        # initialize weights with xavier uniform
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
        # save activation function
        self.act_fn = act_fn

    def forward(self, x: Tensor) -> Tensor:
        # apply all but the last layer
        for l in self.layers[:-1]:
            x = self.act_fn(l(x))
        # apply last layer seperately to avoid activation function
        return self.layers[-1](x)


class SoftmaxAttention(nn.Module):
    """ Linear Softmax Attention Module used in original AttentionXML implementation """

    def forward(self,
                x: torch.FloatTensor,
                mask: torch.BoolTensor,
                label_emb: torch.FloatTensor
                ) -> torch.FloatTensor:

        # compute attention scores
        scores = x @ label_emb.transpose(1, 2)
        scores = scores.masked_fill(~mask.unsqueeze(-1), -1e5)
        scores = torch.softmax(scores, dim=-2)
        # compute label-aware embeddings
        return scores.transpose(1, 2) @ x


class InterBagAttention(nn.Module):

    def forward(self,
                x: torch.FloatTensor
                ) -> torch.FloatTensor:
        '''
        x: shape (num_bag_groups, bag_group_size, num_labels,encoder_hidden_size)
        '''

        # shape (num_bag_groups, num_labels, bag_group_size, encoder_hidden_size)
        x = x.transpose(1, 2)

        # shape (num_bag_groups, num_labels, bag_group_size, bag_group_size)
        bag_similarities = x @ x.transpose(2, 3)

        # Subtract one for similarity to oneself (which is 1 due to the normalization)
        bag_similarities = bag_similarities.sum(dim=-1) - 1

        # shape (num_bag_groups, num_labels, bag_group_size)
        bag_similarities = torch.softmax(bag_similarities, dim=-1)

        # shape (num_bag_groups, num_labels, encoder_hidden_size)
        return (bag_similarities[:, :, None, :] @ x).squeeze()


class MultiHeadAttention(nn.MultiheadAttention):
    """ Multi-Head Attention Module that can be used in a `LabelAttentionClassifier` Module """

    def forward(self,
                x: torch.FloatTensor,
                mask: torch.BoolTensor,
                label_emb: torch.FloatTensor
                ) -> torch.FloatTensor:
        # prepare inputs
        x = x.transpose(0, 1)
        label_emb = label_emb.transpose(0, 1)
        # apply multi-head attention
        attn_out, attn_weight = super(MultiHeadAttention, self).forward(
            query=label_emb,
            key=x,
            value=x,
            key_padding_mask=~mask
        )
        # reverse transpose
        return attn_out.transpose(0, 1)


class LabelAttentionClassifierMLP(nn.Module):
    """ Label-attention based Multi-Label Classifier """

    def __init__(self,
                 hidden_size: int,
                 num_labels: int,
                 attention: nn.Module,
                 mlp: MLP
                 ) -> None:
        # initialize module
        super(LabelAttentionClassifierMLP, self).__init__()
        # save the attention and mlp module
        self.att = attention
        self.mlp = mlp
        # create label embedding
        self.label_embed = nn.Embedding(
            num_embeddings=num_labels,
            embedding_dim=hidden_size,
            sparse=False
        )
        # use xavier uniform for initialization
        nn.init.xavier_uniform_(self.label_embed.weight)

    def forward(self,
                x: torch.FloatTensor,
                mask: torch.BoolTensor,
                candidates: torch.IntTensor = None
                ) -> torch.FloatTensor:
        # use all embeddings if no candidates are provided
        if candidates is None:
            n = self.label_embed.num_embeddings
            candidates = torch.arange(n).unsqueeze(0)
            candidates = candidates.repeat(x.size(0), 1)
            candidates = candidates.to(x.device)

        # get label embeddings and apply attention layer
        label_emb = self.label_embed(candidates)
        m = self.att(x, mask, label_emb)
        # apply classifier
        return self.mlp(m).squeeze(-1)


class BagAttentionClassifier(nn.Module):
    """ Label-attention based Multi-Label Classifier """

    def __init__(self,
                 encoder_hidden_size: int,
                 num_labels: int,
                 attention: nn.Module,
                 bag_group_size: int,
                 ) -> None:
        # initialize module
        super(BagAttentionClassifier, self).__init__()

        # Store params
        self.num_labels = num_labels
        self.encoder_hidden_size = encoder_hidden_size

        # save the attention and mlp module
        self.att = attention
        # create label embedding
        self.label_embed = nn.Embedding(
            num_embeddings=num_labels,
            embedding_dim=encoder_hidden_size,
            sparse=False
        )

        self.bias = nn.Embedding(
            num_embeddings=num_labels,
            embedding_dim=1,
            sparse=False
        )

        # None if inter_bag should not be used
        self.bag_group_size = bag_group_size
        # Interbag attention params
        self.inter_bag_att = InterBagAttention()

        # use xavier uniform for initialization
        nn.init.xavier_uniform_(self.label_embed.weight)
        nn.init.xavier_uniform_(self.bias.weight)

    def forward(self,
                x: torch.FloatTensor,
                mask: torch.BoolTensor,
                candidates: torch.IntTensor = None
                ) -> torch.FloatTensor:
        # use all embeddings if no candidates are provided
        if candidates is None:
            n = self.label_embed.num_embeddings
            candidates = torch.arange(n).to(x.device)
            # candidates = torch.arange(n).unsqueeze(0)
            # candidates = candidates.repeat(x.size(0), 1)
            # candidates = candidates.to(x.device)er

            # get label embeddings and apply attention layer
            label_emb = self.label_embed(candidates).unsqueeze(0)

        else:
            # Assume that candidates are the same for all samples in the batch
            label_emb = self.label_embed(candidates[0]).unsqueeze(0)

        # (num_bags,num_labels,hidden_dim)
        x = self.att(x, mask, label_emb)

        # Normalize each sample
        x = x / torch.norm(x, dim=2, keepdim=True)

        # Inter-Bag during training
        if self.training and self.bag_group_size is not None:
            # Form bag groups
            x = x.view(-1, self.bag_group_size, self.num_labels,
                       self.encoder_hidden_size)

            # shape (num_bag_groups, num_labels, hidden_dim)
            x = self.inter_bag_att(x)

        bias = self.bias(candidates[0]).squeeze(-1)

        # (num_bags,num_labels)
        dot_prod = torch.sum(x*label_emb, dim=-1)

        # apply classifier
        return dot_prod+bias
