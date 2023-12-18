from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

data_1 = "35 year old Vimal, a married man, falls in love with a young woman and ends up killing his wife, only to discover that it was planned by his wife all along to put him behind bars."
data_2 = "A young decoit, Shubham, is recruited in the gang. When a tough situation arises, Shubham kills a policeman, only to later discover that it was planned by the gang leader to eliminate the policeman and blame Shubham."
data_3 = " Taanshi, a young vibrant head of marketing at a company, decides to run for Prime Minister of the country. Her ambitions are met with problems from all over but she doesn't give up and ends up becoming Prime Minister of the company."
data_4 = "On his dead bead, Anubhav, a tech giant, decides to name his property to his employee, Mohan, who has planned this all along. Mohan faces legal problems but finally becomes the head of the tech company and owns the property."
data_5 = "Akshat, after loosing his father, starts seeing him in his dreams telling him to sleep calmly. But his sleep is disturbed to an extent that he ends up believing his father is alive. Later he kills himself and sleeps forever."


# Sentences we want sentence embeddings for
sentences = [data_1,data_2]


# Load AutoModel from huggingface model repository
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=1024, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, mean pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
sentence_embedding = F.normalize(sentence_embeddings, p=2, dim=1)
print(sentence_embedding[0])
print(sentence_embedding[1])


similarity = cosine_similarity([sentence_embedding[0]], [sentence_embedding[1]])

print("Cosine Similarity:", similarity)
