import os
import pandas as pd
import re
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv
import time
from tqdm import tqdm
from utils.utils import remove_think_content, bcolors  # Updated import
from config.prompts import prompts
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

DATA_DIR = "/home/modar/Desktop/AI_chatbot/Q&A_data/need_rag/"
OUTPUT_DIR = "/home/modar/Desktop/AI_chatbot/Q&A_data/need_rag_refined/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PARAPHRASE_PROMPT = prompts.PARAPHRASE_PROMPT

model = ChatGroq(model_name="llama3-8b-8192", temperature=0.5)

def refine_questions(file_path, output_path, think=False):
    df = pd.read_csv(file_path)
    questions = "\n".join([f"{i+1}- {q}" for i, q in enumerate(df['question'].tolist())])
    input_prompt = PARAPHRASE_PROMPT.format(questions=questions)
    result = model.invoke([SystemMessage(content=input_prompt)]).content
    if think:
        result = remove_think_content(result)
    pattern = r"\d+-\s*"
    refined_questions = re.split(pattern, result)
    refined_questions = [q.strip() for q in refined_questions if q.strip()]
    refined_df = pd.DataFrame({'question': refined_questions, 'label': ['rag'] * len(refined_questions)})
    refined_df.to_csv(output_path, index=False)
    time.sleep(1)

if __name__ == "__main__":
    for file in tqdm(os.listdir(DATA_DIR)):
        if file.endswith(".csv"):
            input_path = os.path.join(DATA_DIR, file)
            output_path = os.path.join(OUTPUT_DIR, file)
            print(f"Refining: {file}")
            refine_questions(input_path, output_path, think=True)
    print(bcolors.OKGREEN + "Refinement complete!" + bcolors.ENDC)