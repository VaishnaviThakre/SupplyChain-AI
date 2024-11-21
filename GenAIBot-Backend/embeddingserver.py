import json
from typing import List
from langchain.embeddings.base import Embeddings
import requests

class OpenEmbedding(Embeddings):
    """Interface for embedding models"""
    def __init__(self, server_url: str):
        self.server_url = server_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embedd = []
        for text in texts:
            dict_to_send = {
                "query": text
            }
            try:
                response = requests.post(
                    self.server_url,
                    json=dict_to_send,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                embedd.append(response.json()["embedding"])
            except Exception as e:
                print(f"Error embedding document: {str(e)}")
                raise
        return embedd

    def embed_query(self, text: str) -> List[float]:
        dict_to_send = {
            "query": text
        }
        try:
            response = requests.post(
                self.server_url,
                json=dict_to_send,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            print(f"Error embedding query: {str(e)}")
            raise