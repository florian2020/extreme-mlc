# File to do small test on the behaviour of sentence transformers

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn.functional as F
from transformers.utils.logging import enable_default_handler
import json
from tqdm import tqdm


# Mean Pooling - Take attention mask into account for correct averaging
# Similar to https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2


def mean_pooling(x, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(x.size()).float()
    return torch.sum(x * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(
    'sentence-transformers/all-MiniLM-L6-v2')
model = SentenceTransformer('all-MiniLM-L6-v2')

model_hugging = AutoModel.from_pretrained(
    'sentence-transformers/all-MiniLM-L6-v2')


texts = ["This is a first sentence. And this is another one.", 'Second sentence']
texts_hugging = [
    "This is a first sentence. And this is another one.", 'Second sentence']


# Tokenize sentences
tokenizer_output = tokenizer(texts, padding=True,
                             truncation=True, return_tensors='pt')
tokenizer_output_hugging = tokenizer(texts_hugging, padding=True,
                                     truncation=True, return_tensors='pt')

model_out = model(tokenizer_output)
model_out_hugging = model_hugging(**tokenizer_output_hugging)

x = mean_pooling(
    model_out['token_embeddings'], tokenizer_output['attention_mask'])
# x = x/x.norm(dim=-1, keepdim=True)

x_hugging = mean_pooling(
    model_out_hugging[0], tokenizer_output_hugging['attention_mask'])
# x_hugging = x_hugging/x_hugging.norm(dim=-1, keepdim=True)


print("Normalized averaged token embeddings:")
print(x[0, :20])
print("Sentence embeddings:")
print(model_out['sentence_embedding'][0, :20])

# print("Encode")
# print(model.encode(texts)[0, :20])


print("HuggingFace Normalized averaged token embeddings:")
print(model_out_hugging[0][0, :20])
print("HuggingFace Sentence embeddings:")
print(x_hugging[0, :20])
