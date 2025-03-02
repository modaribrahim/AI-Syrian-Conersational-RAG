from ..config.constants import ALLOWED_WORDS
from ..config.settings import settings  # Import settings properly
import numpy as np
from ..utils.utils import bcolors
import re
from abc import ABC, abstractmethod
from ..models.schemas import DenseRetriever, State


class RetrieveFaiss(DenseRetriever):
    def __init__(self, embedding_model, index, docs):
        super().__init__(embedding_model, index, docs)

    def retrieve(self, state: State):
        print(bcolors.OKBLUE + "Retrieving.." + bcolors.ENDC)

        message = state["messages"][-1].content
        translation = state["translation"]
        query = translation if translation else message

        query_embedding = np.array([self.embedding_model.embed_query(query)])

        _, indices = self.index.search(query_embedding, k=settings.top_k)

        docs = [self.docs[i] for i in indices[0] if 0 <= i < len(self.docs)]

        return {"documents": docs}
