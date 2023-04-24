
import torch
from langchain.llms.base import LLM
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex, PromptHelper
from llama_index import LLMPredictor, ServiceContext
from transformers import pipeline
from typing import Optional, List, Mapping, Any

class CustomLLM(LLM):
    model_name = "google/flan-t5-base"
    pipeline = pipeline("text2text-generation", model=model_name, device=0, model_kwargs={"torch_dtype":torch.bfloat16})
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt_length = len(prompt)
        response = self.pipeline(prompt, max_length=1024)[0]["generated_text"]

        # only return newly generated tokens
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"

# define our LLM
llm_predictor = LLMPredictor(llm=CustomLLM())
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit = 512)

documents = SimpleDirectoryReader('data1').load_data()
index = GPTListIndex.from_documents(documents, service_context=service_context)
print(index.query("What is the position of modi"))

#executing to remove caches 
torch.cuda.empty_cache()
