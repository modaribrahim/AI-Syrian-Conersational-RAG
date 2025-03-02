from ..models.schemas import Generator , bcolors
from ..config.prompts import prompts
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState 
from langchain.schema import Document  
from langchain_core.messages import HumanMessage
from ..models.schemas import State

class RAGGenerator(Generator):

    def __init__(self, model):
        super().__init__(model)


    def generate(self, state: State):
   
        print(bcolors.OKBLUE + 'Generating...' + bcolors.ENDC)
        question = state.get("question","")
        documents = state.get("documents","")
        print(documents)
        summary = state.get("summary", "")
        messages = ''
        if summary:
            
            system_message = f"Summary of conversation earlier: {summary}\n New messages: "

            messages = system_message + ' '.join([message.content for message in state["messages"]])
        
        else:
            messages = ' '.join([message.content for message in state["messages"]])

        if isinstance(documents,list):
            documents.extend(Document(page_content=state.get('tavily_docs','')))
            documents.extend(Document(page_content=state.get('duckduck_docs','')))
        generation = self.model.invoke([SystemMessage(content=prompts.generation_prompt.format(question=question,documents=documents,memory= messages))]).content
        return {
            "documents": [],
            "question": question,
            "messages": [SystemMessage(content=generation)],
            "generation": generation
        }    