import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


model = SentenceTransformer('gtr-t5-xl')

data_1 = "35 year old Vimal, a married man, falls in love with a young woman and ends up killing his wife, only to discover that it was planned by his wife all along to put him behind bars."
data_2 = "A young decoit, Shubham, is recruited in the gang. When a tough situation arises, Shubham kills a policeman, only to later discover that it was planned by the gang leader to eliminate the policeman and blame Shubham."
data_3 = " Taanshi, a young vibrant head of marketing at a company, decides to run for Prime Minister of the country. Her ambitions are met with problems from all over but she doesn't give up and ends up becoming Prime Minister of the company."
data_4 = "On his dead bead, Anubhav, a tech giant, decides to name his property to his employee, Mohan, who has planned this all along. Mohan faces legal problems but finally becomes the head of the tech company and owns the property."
data_5 = "Akshat, after loosing his father, starts seeing him in his dreams telling him to sleep calmly. But his sleep is disturbed to an extent that he ends up believing his father is alive. Later he kills himself and sleeps forever."


query_embedding = model.encode(data_2)
passage_embedding = model.encode(data_5)

analyzer = SentimentIntensityAnalyzer()
vs = analyzer.polarity_scores(data_1)
print(vs['neg'])
print(vs['neu'])
print(vs['pos'])


print(f" query_embedding : {np.shape(query_embedding)}")
print(f" passage_embedding : {np.shape(passage_embedding)}")
print("Similarity:", util.dot_score(query_embedding, passage_embedding))
