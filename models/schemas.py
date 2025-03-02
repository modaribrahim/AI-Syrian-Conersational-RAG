from abc import ABC , abstractmethod
import operator
from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field
from typing import Literal , Optional
from ..config.prompts import prompts
from typing import Annotated , TypedDict
from ..utils.utils import AddMessages

###################### Colors Model ######################

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    

###################### State Model ######################

class State(TypedDict):

    no_rag: bool
    summary: str
    tavily_docs: str
    duckduck_docs: str
    documents: list
    question: str
    translation: str
    search: str
    messages: Annotated[list[AnyMessage],AddMessages]
    generation: str

###################### Response Models ######################

class IntentClassifierResponseModel(BaseModel):
    answer: str = Field(
        description = prompts.intent_classifier_class_prompt
    )
    translation: str = Field(
        description = prompts.intent_classifier_translation_prompt
    )

class GraderResponseModel(BaseModel):
    """Binary score for relevance check on retrieved historical documents."""
    binary_score: Literal["yes", "no"] = Field(
        description = prompts.grader_prompt
    )

###################### Abstract Classes ######################

class IntentClassifier(ABC):

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def classify(self, state: State):
        pass

class DenseRetriever(ABC):

    def __init__(self, embedding_model, index, docs):
        self.embedding_model = embedding_model
        self.index = index
        self.docs = docs

    @abstractmethod
    def retrieve(self, state: State):
        pass

class Grader(ABC):
    
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def grade(self, state: State):
        pass

class Search(ABC):
    
    def __init__(self,model=None):
        self.model = model
        
    
    @abstractmethod
    def search(self,state: State):
        pass

class Generator(ABC):

    def __init__(self, model):
        self.model = model
        
    @abstractmethod
    def generate(self,state: State):
        pass

class summarizer(ABC):

    def __init__(self , model):
        self.model = model
        
    @abstractmethod
    def summarize(self, state: State):
        pass
    
