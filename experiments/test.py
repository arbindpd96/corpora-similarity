import os

import numpy as np
import pandas as pd
import chromadb

from ast import literal_eval
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from sklearn.metrics.pairwise import cosine_similarity

from config import EMBEDDINGS_MODEL


# Global variables
embedding_function = OpenAIEmbeddingFunction(
    api_key=os.environ.get('OPENAI_API_KEY'),
    model_name=EMBEDDINGS_MODEL
)
client = chromadb.EphemeralClient()

movies_collection = client.create_collection(
    name="movies",
    embedding_function=embedding_function
)

movies_collection.add(
    documents=[
        "35 year old Vimal, a married man, falls in love with a young woman and ends up killing his wife, only to discover that it was planned by his wife all along to put him behind bars.",
        " Taanshi, a young vibrant head of marketing at a company, decides to run for Prime Minister of the country. Her ambitions are met with problems from all over but she doesn't give up and ends up becoming Prime Minister of the company.",
        "On his dead bead, Anubhav, a tech giant, decides to name his property to his employee, Mohan, who has planned this all along. Mohan faces legal problems but finally becomes the head of the tech company and owns the property.",
        "Akshat, after loosing his father, starts seeing him in his dreams telling him to sleep calmly. But his sleep is disturbed to an extent that he ends up believing his father is alive. Later he kills himself and sleeps forever.",
        "I like mango",
        "Mangoes are liked by me",
        "I am stating a fact that i kind of have a mango allergy, in general they maybe fine but i am on the otherside of that spectrum",
        "i dont like mangoes",
        "i dislike cars"
    ],

    ids=["doc1", "doc2", "doc3", "doc4", "doc5", "doc6", "doc7", "doc8", "doc9"]
)
results = movies_collection.query(
    query_texts="I am stating a fact that I despise car, in general they maybe fine but i am on the otherside of that spectrum",
    include=['distances', 'embeddings']
)
print(results)
print(f"shape {np.shape(results['embeddings'])}")
final = pd.DataFrame({
    'score': results['distances'][0],
    'ids': results['ids'][0]
})
final['percentage_score'] = (1 - final['score']) * 100

print(final)
# similarity = cosine_similarity([results['embeddings'][0][0]], [results['embeddings'][0][1]])
# similarity_percentage = similarity[0][0] * 100
#
# print(f"Similarity Percentage: {similarity_percentage}%")

# "In \"Hamlet\" the title character, Prince Hamlet of Denmark, grapples with the recent death of his father, King Hamlet. His uncle, Claudius, not only claims the throne but also marries Hamlet's mother, Gertrude, swiftly after the king's death. Hamlet's distress intensifies when the ghost of his father appears, claiming Claudius murdered him and demanding vengeance.Hamlet, consumed by grief and a thirst for revenge, pretends to be mad to mask his investigative and plotting efforts against Claudius. This feigned madness causes turmoil in the court, especially affecting Ophelia, whom he loves, and her family.In a tragic turn of events, Hamlet's actions, coupled with his hesitation, lead to a chain of tragedies, including the accidental killing of Polonius, Ophelia's father. This incident drives Ophelia into madness and eventually death. Her brother, Laertes, seeks revenge for his family, targeting Hamlet.The story reaches its climax in a fencing match between Hamlet and Laertes. Both are fatally wounded in the duel, during which Laertes reveals Claudius's plot to kill Hamlet. In his final act, Hamlet kills Claudius before dying himself. The play concludes with the Norwegian Prince Fortinbras entering and taking control of a devastated Danish court.The play delves into themes like the complexity of action, revenge, and the impact of grief and betrayal.",
