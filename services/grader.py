from ..models.schemas import Grader, State, bcolors, GraderResponseModel
from ..config.prompts import prompts
from langchain_core.messages import SystemMessage


class LLMGrader(Grader):
    def __init__(self,model):
        super().__init__(model)

    def grade(self, state: State):
        print(bcolors.OKBLUE + "Grading..." + bcolors.ENDC)
        question = state.get("question", "")
        documents = state.get("documents", []) 
        filtered_docs = []
        search = "No"

        for d in documents:
            print(bcolors.WARNING + "Grading the document content: \n" + str(d.page_content) + bcolors.ENDC)

            self.structured_llm_grader = self.model.with_structured_output(GraderResponseModel)

            grade_prompt = prompts.grader_prompt.format(question=question, documents=d.page_content)

            result = self.structured_llm_grader.invoke([SystemMessage(content=grade_prompt)])
            score = result.binary_score 

            if score == "yes":  
                filtered_docs.append(d)
            else:
                print(bcolors.FAIL + "Not related" + bcolors.ENDC)
                search = "Yes"

        return {
            "documents": filtered_docs,
            "question": question,
            "search": search,
        }
