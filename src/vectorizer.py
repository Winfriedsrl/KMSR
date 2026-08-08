import os

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import torch

from config import load_env


class Vectorizer:
    AVAILABLE_MODELS = {
        "ita-legal-bert": {
            "backend": "sentence-transformers",
            "id": "dlicari/distil-ita-legal-bert",
            "dims": 768,
            "max_input_length": 512,
        },
        "text-embedding-3-large": {
            "backend": "openai",
            "id": "text-embedding-3-large",
            "dims": 3072,
            "max_input_length": 8192,
        },
    }

    def __init__(self, model_name: str):
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError("Model not available")

        self.model_name = model_name
        info = self.AVAILABLE_MODELS[model_name]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if info["backend"] == "sentence-transformers":
            self.model = SentenceTransformer(info["id"], device=device)
        elif info["backend"] == "openai":
            load_env()
            self.model = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        else:
            raise ValueError("Unsupported backend")

    def get_embedding(self, text):
        info = self.AVAILABLE_MODELS[self.model_name]
        if info["backend"] == "sentence-transformers":
            return self.model.encode(text).tolist()
        if info["backend"] == "openai":
            response = self.model.embeddings.create(
                model=info["id"],
                input=text,
                encoding_format="float",
            )
            return response.data[0].embedding

        raise ValueError("Unsupported backend")

    def get_embeddings(self, texts):
        info = self.AVAILABLE_MODELS[self.model_name]
        if info["backend"] == "sentence-transformers":
            return self.model.encode(texts).tolist()
        if info["backend"] == "openai":
            response = self.model.embeddings.create(
                model=info["id"],
                input=texts,
                encoding_format="float",
            )
            return [item.embedding for item in response.data]

        raise ValueError("Unsupported backend")
