import os
import time
from typing import Any, List, Optional
import ray
import torch
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.utils import enforce_stop_tokens
from langchain.prompts import PromptTemplate
from ray import serve
from starlette.requests import Request
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                          AutoTokenizer)
from transformers import pipeline as hf_pipeline
# from wandb.integration.langchain import WandbTracer

from local_embeddings import LocalHuggingFaceEmbeddings
import demoConfig

model = "meta-llama/Llama-2-7b-chat-hf"
# HuggingFace token: https://huggingface.co/settings/tokens
token = demoConfig.hf_token
        
class LlamaPipeline(HuggingFacePipeline):
    """A Llama-2-13b Chat Pipeline with Retrieval Augmented Generation
    """
    global token
    

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.pipeline(
            prompt, 
            temperature=0.1, # Randomness of the model 
            max_new_tokens=512, # Model output of the tokens
            do_sample=True, 
            return_full_text = True, # Langchain expects full text
            top_k=10, # sampling technique: diversifies the text generation by randomly selecting among the k most probable tokens
            top_p=0.7, # nucleus sampling: dynamically forms a nucleus of tokens based on cumulative probability
            repetition_penalty=1.1, # prevent repitition of output    
            
        )
        if self.pipeline.task == "text-generation":
            # Text generation return includes the starter text.
            print(f"Response is: {response}")
            text = response[0]["generated_text"][len(prompt) :]
        else:
            raise ValueError(f"Got invalid task {self.pipeline.task}. ")
        # text = enforce_stop_tokens(text, [50278, 50279, 50277, 1, 0])
        return text

    @classmethod
    def from_model(
        cls,
        model: str,   # model_id to model for llama-2
        task: str,
        token: token,
        device: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ):
        """Construct the pipeline object from model and task."""

        pipeline = hf_pipeline(
            model=model,
            task=task,
            token=token,
            device=device,
            model_kwargs=model_kwargs,
        )
        return cls(
            pipeline=pipeline,
            model_id=model,
            model_kwargs=model_kwargs,
            **kwargs,
        )
