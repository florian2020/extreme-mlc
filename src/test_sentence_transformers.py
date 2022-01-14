# File to do small test on the behaviour of sentence transformers


from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn.functional as F
from transformers.utils.logging import enable_default_handler
import json
from tqdm import tqdm

# Mean Pooling - Take attention mask into account for correct averaging


def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


path = './data/datasets/EURLEX57K_tiny_mil/train_texts.json'

texts = []
max_instances = 0
# with open(path, "r") as f:
#     for line in tqdm(f, f'Loading texts from {path}'):
#         instances = json.loads(line)
#         texts.append(instances)

#         if len(instances) > max_instances:
#             max_instances = len(instances)

texts = []
# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(
    'sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
# model = AutoModel.from_pretrained(
#     'sentence-transformers/all-MiniLM-L12-v2')

# model = SentenceTransformer('all-MiniLM-L12-v2')


texts = ["This is a first sentence. And this is another one.",
         "This is a second paragraph"]

records = []


# print(type(tokenizer))
for property, value in vars(model).items():
    print(property, ":", value)

# Tokenize sentences
tokenizer_output = tokenizer(texts, padding=True,
                             truncation=True, return_tensors='pt')

# encoded_input = tokenizer_output['input_ids']

# encoded_input = tokenizer(sentences, padding=True,
#                           truncation=True, return_attention_mask=False)['input_ids']

# encoded_input.extend([[1]*len(encoded_input[0])])

# encoded_input = torch.LongTensor(encoded_input)

# attention_mask = (encoded_input != 1)

# Compute token embeddings
with torch.no_grad():
    # model_output = model(input_ids=encoded_input,
    #                      attention_mask=attention_mask)
    model_output = model(tokenizer_output)

# # Perform pooling
# sentence_embeddings = mean_pooling(
#     model_output, attention_mask)

# # Normalize embeddings
# sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

# records.append(sentence_embeddings)

print("Sentence embeddings:")
print(model_output)
# print(sentence_embeddings@sentence_embeddings.T)
# print(sentence_embeddings.shape)
