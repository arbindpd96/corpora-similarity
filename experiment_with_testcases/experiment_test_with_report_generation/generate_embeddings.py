import os

import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import RobertaTokenizer, RobertaModel

from config import EMBEDDINGS_MODEL, EMBEDDINGS_BATCH_SIZE
from experiment_with_testcases.experiment_test_with_report_generation.generate_plot_to_one_line import PhaseCommand

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

class Embedding(PhaseCommand) :
    def __init__(self, plot, method):
        self.plot = plot
        self.method = method

    def execute(self, *arg):
        return self.getEmbedding(arg[0] if arg[0] else self.plot)

    def getEmbedding(self, str):
        if self.method == 'openai':
            return self.getEmbeddingsUsingOpenAi(str)
        elif self.method == 'sbert':
            return self.getEmbeddingsUsingSentenceTransformers(str)
        elif self.method == 'bert' :
            return self.getEmbeddingFromBert(str)

    def getEmbeddingsUsingSentenceTransformers(self, inputStr: str):
        embeddings = []
        model = SentenceTransformer('gtr-t5-xl')
        for batch_start in range(0, len(inputStr), EMBEDDINGS_BATCH_SIZE):
            batch_end = batch_start + EMBEDDINGS_BATCH_SIZE
            batch = inputStr[batch_start:batch_end]
            query_embedding = model.encode(batch)
            embeddings.extend([e for e in query_embedding])

        return embeddings

    def getEmbeddingsUsingOpenAi(self, stringsArr: list):
        embeddings = []

        for batch_start in range(0, len(stringsArr), EMBEDDINGS_BATCH_SIZE):
            batch_end = batch_start + EMBEDDINGS_BATCH_SIZE
            batch = stringsArr[batch_start:batch_end]

            response = client.embeddings.create(
                model=EMBEDDINGS_MODEL,
                input=batch
            )
            embeddings.extend([e.embedding for e in response.data])

        return embeddings[0]

    def getEmbeddingFromBert(self, input: str):
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base')
        encoded_input_1 = tokenizer(input, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            model_output_1 = model(**encoded_input_1)
        embeddings_1 = model_output_1.last_hidden_state.mean(dim=1)
        return embeddings_1.detach().numpy()[0]



