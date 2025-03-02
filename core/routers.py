from langgraph.graph import MessagesState 
from langgraph.graph import END 
from ..models.schemas import State
from ..utils.utils import bcolors



def route_grader(state:State) -> str:
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    search = state["search"]

    if search == "Yes":
        return ['tavily_search','duckduckgo_search']
    else:
        return "generator"
    

def route_generator(state: State) -> str:
    print(bcolors.OKBLUE + f'Check condition... {len(state["messages"])}' + bcolors.ENDC)
    return 'summarizer' if len(state["messages"]) >= 2 else END

def route_classifier(state: State) -> str:

    return "generator" if state.get("no_rag", False) else "retriever",
