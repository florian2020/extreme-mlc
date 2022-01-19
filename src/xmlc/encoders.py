import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


class LSTMEncoder(nn.Module):
    """ Basic LSTM Encoder """

    def __init__(self,
                 embed_size: int,
                 hidden_size: int,
                 num_layers: int,
                 vocab_size: int,
                 padding_idx: int,  # The vectors at padding_idx are not updated during training
                 emb_init: torch.FloatTensor = None,
                 dropout: float = 0.2
                 ) -> None:
        super(LSTMEncoder, self).__init__()
        self.dropout = dropout
        # create embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size,
            padding_idx=padding_idx,
            _weight=emb_init if emb_init is not None else None
        )
        # #Uncomment to make embeddings not trainable
        # self.embedding.weight.requires_grad = False

        # create lstm encoder
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        # initial hidden and cell states for lstm
        self.h0 = nn.Parameter(torch.zeros(num_layers*2, 1, hidden_size))
        self.c0 = nn.Parameter(torch.zeros(num_layers*2, 1, hidden_size))

    def forward(self,
                input_ids: torch.LongTensor,
                input_mask: torch.BoolTensor
                ) -> torch.Tensor:
        # flatten parameters
        self.lstm.flatten_parameters()
        # pass through embedding
        b, s = input_ids.size()
        x = self.embedding.forward(input_ids)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # pack padded sequences
        lengths = input_mask.sum(dim=-1).cpu()
        packed_x = nn.utils.rnn.pack_padded_sequence(
            input=x,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False
        )
        # apply lstm encoder
        h0 = self.h0.repeat_interleave(b, dim=1)
        c0 = self.c0.repeat_interleave(b, dim=1)
        packed_x, _ = self.lstm(packed_x, (h0, c0))
        # unpack packed sequences, dim x = (batch_size, text_length, encoder_hidden_size)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            sequence=packed_x,
            batch_first=True,
            padding_value=0,
            total_length=s
        )
        return F.dropout(x, p=self.dropout, training=self.training)


class LSTMSentenceEncoder(nn.Module):
    """ Basic LSTM Encoder """

    def __init__(self,
                 hidden_size: int,
                 num_layers: int,
                 padding_idx: int,  # The vectors at padding_idx are not updated during training
                 emb_init: torch.FloatTensor = None,
                 dropout: float = 0.2
                 ) -> None:

        super(LSTMSentenceEncoder, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = self.enc = LSTMEncoder(
            vocab_size=emb_init.shape[0],
            embed_size=emb_init.shape[1],
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            padding_idx=padding_idx,
            emb_init=torch.from_numpy(emb_init).float(),
            dropout=dropout
        )

    def forward(self,
                input_ids: torch.LongTensor,
                input_mask: torch.BoolTensor
                ) -> torch.Tensor:
        """
        To be tested
        """

        batch_size, num_instances, seq_length = input_ids.shape

        # Flatten instance dimension of input and mask
        input_ids_without_instances = input_ids.view(
            batch_size, seq_length * num_instances)
        input_mask_without_instances = input_mask.view(
            batch_size, seq_length * num_instances)

        x = self.lstm(input_ids_without_instances,
                      input_mask_without_instances)

        # Recover instance dimension
        x = x.view(batch_size, num_instances, seq_length, self.hidden_size*2)

        return x.mean(-2)


class SentenceTransformerEncoder(nn.Module):
    """ Basic Sentence Transformer Encoder """

    def __init__(self,
                 sent_transformer: SentenceTransformer
                 ) -> None:
        super(SentenceTransformerEncoder, self).__init__()

        self.sentence_transformer_model = sent_transformer

    def forward(self,
                input_ids: torch.LongTensor,
                input_mask: torch.BoolTensor
                ) -> torch.Tensor:

        batch_size, num_instances, seq_length = input_ids.shape

        # Flatten instance dimension of input and mask
        input_ids_without_instances = input_ids.view(
            batch_size * num_instances, seq_length)
        input_mask_without_instances = input_mask.view(
            batch_size * num_instances, seq_length)

        model_output = self.sentence_transformer_model(
            {'input_ids': input_ids_without_instances, 'attention_mask': input_mask_without_instances})

        sentence_embeddings = model_output['sentence_embedding']

        # Recover instance dimension
        sentence_embeddings = sentence_embeddings.view(
            batch_size, num_instances, -1)

        return sentence_embeddings
