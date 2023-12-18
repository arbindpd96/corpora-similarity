from transformers import RobertaModel, RobertaTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Initialize tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

data_1 = "35 year old Vimal, a married man, falls in love with a young woman and ends up killing his wife, only to discover that it was planned by his wife all along to put him behind bars."
data_2 = "A young decoit, Shubham, is recruited in the gang. When a tough situation arises, Shubham kills a policeman, only to later discover that it was planned by the gang leader to eliminate the policeman and blame Shubham."
data_3 = " Taanshi, a young vibrant head of marketing at a company, decides to run for Prime Minister of the country. Her ambitions are met with problems from all over but she doesn't give up and ends up becoming Prime Minister of the company."
data_4 = "On his dead bead, Anubhav, a tech giant, decides to name his property to his employee, Mohan, who has planned this all along. Mohan faces legal problems but finally becomes the head of the tech company and owns the property."
data_5 = "Akshat, after loosing his father, starts seeing him in his dreams telling him to sleep calmly. But his sleep is disturbed to an extent that he ends up believing his father is alive. Later he kills himself and sleeps forever."



# Tokenize and encode sentences
encoded_input_1 = tokenizer(data_2, return_tensors='pt')
encoded_input_2 = tokenizer(data_5, return_tensors='pt')

# Get embeddings
with torch.no_grad():
    model_output_1 = model(**encoded_input_1)
    model_output_2 = model(**encoded_input_2)

# Calculate the mean of the last hidden states
embeddings_1 = model_output_1.last_hidden_state.mean(dim=1)
embeddings_2 = model_output_2.last_hidden_state.mean(dim=1)

# Convert to numpy arrays and calculate cosine similarity
embeddings_1_np = embeddings_1.detach().numpy()
embeddings_2_np = embeddings_2.detach().numpy()
similarity = cosine_similarity([embeddings_1_np[0]], [embeddings_2_np[0]])

print("Cosine Similarity:", similarity)
