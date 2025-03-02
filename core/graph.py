import pickle
import faiss
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from ..config.settings import settings
from ..models.schemas import State
from langgraph.graph import StateGraph, END 
from ..services.generator import RAGGenerator
from ..services.grader import LLMGrader
from ..services.intent_classifier import LLMIntentClassifier
from ..services.retriever import RetrieveFaiss
from ..services.search import TavilySearch, DuckDuckGoSearch
from ..services.summarizer import SimpleSummarizer
from .routers import route_grader , route_generator , route_classifier

load_dotenv()

model = ChatGroq(
    model_name=settings.model_name,
    temperature=0.3,
    api_key=settings.groq_api_key
)

embeddings = HuggingFaceEmbeddings(
    model_name=settings.embedding_model,
    model_kwargs={'device': settings.device},
    encode_kwargs={'normalize_embeddings': settings.normalize_embeddings}
)

index = faiss.read_index(settings.index)
with open(settings.docs, "rb") as f:
    docs = pickle.load(f)

intent_classifier = LLMIntentClassifier(model)
retriever = RetrieveFaiss(embeddings, index, docs)
grader = LLMGrader(model)  
tavily_search = TavilySearch()
duckduckgo_search = DuckDuckGoSearch()
generator = RAGGenerator(model)
summarizer = SimpleSummarizer(model)

def build_graph():
    graph = StateGraph(State)

    graph.add_node("intent_classifier", intent_classifier.classify)
    graph.add_node("retriever", retriever.retrieve)
    graph.add_node("grader", grader.grade)
    graph.add_node("tavily_search", tavily_search.search)
    graph.add_node("duckduckgo_search", duckduckgo_search.search)
    graph.add_node("generator", generator.generate)
    graph.add_node("summarizer", summarizer.summarize)

    graph.add_conditional_edges(
        "intent_classifier",
        route_classifier
    )
    graph.add_edge("retriever", "grader")
    graph.add_conditional_edges(
        "grader",
        route_grader
    )
    graph.add_conditional_edges(
        'generator',
        route_generator
    )
    graph.add_edge("tavily_search", "generator")
    graph.add_edge("duckduckgo_search", "generator")
    graph.add_edge("summarizer", END)

    graph.set_entry_point("intent_classifier")

    app = graph.compile()

    return app