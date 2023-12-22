import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel


def getEmbeddingFromBert(input: str):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    encoded_input_1 = tokenizer(input, return_tensors='pt', padding= True, truncation=True)
    with torch.no_grad():
        model_output_1 = model(**encoded_input_1)
    embeddings_1 = model_output_1.last_hidden_state.mean(dim=1)
    print(np.shape(embeddings_1.detach().numpy()))
    print(embeddings_1.detach().numpy())
    return embeddings_1.detach().numpy()

