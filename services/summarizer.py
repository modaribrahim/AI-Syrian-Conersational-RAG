from ..models.schemas import summarizer , State
from ..utils.utils import bcolors
from langchain_core.messages import HumanMessage, RemoveMessage # type: ignore

class SimpleSummarizer(summarizer):

    def __init__(self, model):
        super().__init__(model)
    
    def summarize(self,state: State):

        print(bcolors.OKBLUE + 'Summarizing...' + bcolors.ENDC)
        summary = state.get("summary", "")
        if summary:
            
            summary_message = (
                f"\nThis is summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )
            
        else:
            summary_message = "\nCreate a summary of the conversation above:"

        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = self.model.invoke(messages)
        
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]
        return {"summary": response.content, "messages": delete_messages}