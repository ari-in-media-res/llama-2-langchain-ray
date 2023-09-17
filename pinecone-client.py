import os
import pinecone
import demoConfig
from sentence_transformers import SentenceTransformer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SentenceTransformer('all-MiniLM-L6-v2', device=device)


# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = demoConfig.pinecone_api_key

# find your environment next to the api key in pinecone console
env = demoConfig.pinecone_env
index_name = demoConfig.pinecone_index_name


pinecone.init(api_key=api_key, environment=env)
#print(pinecone.whoami())


# check if index already exists (it shouldn't if this is first time)
if index_name not in pinecone.list_indexes():
    # if does not exist, create index
    pinecone.create_index(
        index_name,
        dimension=384,  # dimensionality of text-embedding-ada-002
        metric='cosine',
    )
# connect to index
index = pinecone.GRPCIndex(index_name)
# view index stats
print(index.describe_index_stats())


query = "How does Lanchain integrate with Ray?"

# create the query vector
xq = model.encode(query).tolist()

# now query
xc = index.query(xq, top_k=5, include_metadata=True)
print(xc)