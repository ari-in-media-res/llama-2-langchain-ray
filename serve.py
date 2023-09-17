import ray
import os
from starlette.requests import Request
from ray import serve
from typing import List, Optional, Any
import langchain
from langchain.llms.utils import enforce_stop_tokens
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.mapreduce import MapReduceChain
from transformers import pipeline as hf_pipeline
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
import torch
import time
import pinecone
from langchain.llms import HuggingFacePipeline
import torch
import time
from local_embeddings import LocalHuggingFaceEmbeddings
from llama_local_pipelines import LlamaPipeline
import demoConfig
from langchain.cache import InMemoryCache, RedisSemanticCache
from transformers import BitsAndBytesConfig
from fastapi import FastAPI

token = demoConfig.hf_token

    
prompt_template = """<s>[INST] <<SYS>>
    Answer the following query based on the CONTEXT
    given. If you do not know the answer and the CONTEXT doesn't
    contain the answer truthfully say "I don't know".

    CONTEXT:
    {context}

    QUESTION:
    {question} 

    <</SYS>>
    [/INST]
    """

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


@serve.deployment(ray_actor_options={"num_gpus": 1, "num_cpus": 16})
    
class LlamaRAGDeployment:
    def __init__(self):
        # Enable LLM Cache for Langchain QA Chain
        langchain.llm_cache = InMemoryCache() 

        # WandbTracer.init({"project": "retrieval_demo"})
        
        # add Pinecone API key from app.pinecone.io
        pinecone_api_key = demoConfig.pinecone_api_key
        # set Pinecone environment - find next to API key in console
        env = demoConfig.pinecone_env
        index_name = demoConfig.pinecone_index_name

        pinecone.init(api_key=pinecone_api_key, environment=env)

        # Load the data from Pinecone. No change from Part 1
        st = time.time()
        self.embeddings = LocalHuggingFaceEmbeddings("all-MiniLM-L6-v2")
        text_field = "text"


        index = pinecone.Index(index_name)

        self.vectorstore = Pinecone(
            index, self.embeddings.embed_query, text_field
        )

        et = time.time() - st

        print(f"Loading Pinecone database took {et} seconds.")
        st = time.time()
        
        global token
        
      
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from bitsandbytes.
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, # sets the computational type which might be different than the input time
            bnb_4bit_use_double_quant=True, # nested quantization where the quantization constants from the first quantization are quantized again.

        )

        self.llm = LlamaPipeline.from_model(
            model="meta-llama/Llama-2-13b-chat-hf",
            task="text-generation",
            token = token,
            model_kwargs={
                "device_map": "auto", 
                "quantization_config": bnb_config
            }
        )
        et = time.time() - st
        print(f"Loading HF model took {et} seconds.")

        self.chain = load_qa_chain(llm=self.llm, chain_type="stuff", prompt=PROMPT)

    def qa(self, query):
        
        st = time.time()
        context = self.vectorstore.similarity_search(query, k=3)
        print(f"Results from db are: {context}")
        et = time.time() - st

        result = self.chain({"input_documents": context, "question": query})

        print(f"Result is: {result}")
        print(f"Vector Retrieval took: {et} seconds.")
        return result["output_text"]

    
    async def __call__(self, request: Request) -> List[str]:
        return self.qa(request.query_params["query"])

deployment = LlamaRAGDeployment.bind()
