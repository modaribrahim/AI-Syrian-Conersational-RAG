from ..config.constants import ALLOWED_WORDS
from langchain_core.messages import SystemMessage
import re
from ..models.schemas import IntentClassifier, IntentClassifierResponseModel, State ,bcolors
from ..config.prompts import prompts


class LLMIntentClassifier(IntentClassifier):
    def __init__(self, model):
        super().__init__(model)

    def classify(self, state: State):

        print(bcolors.OKBLUE + "Running the intent classifier.." + bcolors.ENDC)

        question = state.get("question", "")
        summary = state.get("summary", "")

        allowed_pattern = ALLOWED_WORDS  

        if re.fullmatch(rf"\s*({allowed_pattern})(\s+({allowed_pattern}))*\s*", question.lower()):
            no_rag = True
            translation = ""
        else:
            rag_prompt = prompts.intent_classifier_prompt.format(question=question)
            structured_model = self.model.with_structured_output(IntentClassifierResponseModel)
            structured_response = structured_model.invoke([SystemMessage(content=rag_prompt)])

            no_rag = structured_response.answer.lower() != "yes"
            translation = structured_response.translation
        print(bcolors.OKGREEN + f'NORAG: {no_rag}' + bcolors.ENDC)
        return {
            "question": question,
            "no_rag": no_rag,
            "translation": translation,
        }
