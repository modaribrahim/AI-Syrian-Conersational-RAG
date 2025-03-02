import os
import sys
import pytesseract
from pdf2image import convert_from_path
from tqdm import tqdm
from dotenv import load_dotenv
import re
import pandas as pd
import warnings
import time
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pickle
from config.settings import settings
from config.prompts import prompts
from utils.utils import clean_pdf_text, bcolors, remove_think_content  

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

general_QA_prompt = prompts.general_QA_prompt
general_QA_prompt_rephrase = prompts.general_QA_prompt_rephrase
QA_genration_prompt = prompts.QA_genration_prompt

print(bcolors.WARNING + 'Converting the pdf file to md...\n' + bcolors.ENDC)
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

def generate_dataset(rag=True, think=False, rephrase=False):
    model = ChatGroq(model_name=settings.model_name, temperature=0.3) 

    if rag:
        books_path = '/home/modar/Desktop/'
        books = ["ancient-syria.pdf", "History_of_syria.pdf"]
        text_name = 'Final_text.txt'
        if not os.path.exists(books_path + text_name):
            images1 = convert_from_path(f'{books_path}{books[0]}', first_page=26, last_page=338)
            print('f1')
            images2 = convert_from_path(f'{books_path}{books[1]}', first_page=24, last_page=300)
            print('f2')
            images3 = convert_from_path(f'{books_path}{books[1]}', first_page=301, last_page=450)
            print('f3')
            images4 = convert_from_path(f'{books_path}{books[1]}', first_page=451, last_page=730)
            print('f4')
            images = images1 + images2 + images3 + images4
            text = ''
            for img in tqdm(images):
                text += pytesseract.image_to_string(img)
            clean_text = clean_pdf_text(text)
            with open(books_path + text_name, "w") as text_file:
                text_file.write(clean_text)
        else:
            with open(books_path + text_name, "r") as text_file:
                clean_text = text_file.read()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=128,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        text_chunks = text_splitter.split_text(clean_text)
        final_chunks = [Document(page_content=text) if isinstance(text, str) else text for text in text_chunks]

        n = int(len(final_chunks) / 18)
        for i in tqdm(range(n)):
            input_prompt = QA_genration_prompt.format(text=final_chunks[i * 18 + 89])
            result = model.invoke([SystemMessage(content=input_prompt)]).content
            if think:
                result = remove_think_content(result)
            pattern = r"\d+\-\s*"
            data = re.split(pattern, result)
            data = [q.strip() for q in data if q.strip()]
            labels = ['rag' for _ in range(len(data))]
            file = pd.DataFrame({'question': data, 'label': labels})
            file.to_csv(f'/home/modar/Desktop/AI_chatbot/Q&A_data/need_rag/data_{i + 89}.csv')
            time.sleep(2)
    else:
        n = 63
        if not rephrase:
            for i in tqdm(range(n)):
                input_prompt = general_QA_prompt
                result = model.invoke([SystemMessage(content=input_prompt)]).content
                if think:
                    result = remove_think_content(result)
                pattern = r"\d+\-\s*"
                data = re.split(pattern, result)
                data = [q.strip() for q in data if q.strip()]
                labels = ['no_rag' for _ in range(len(data))]
                file = pd.DataFrame({'question': data, 'label': labels})
                file.to_csv(f'/home/modar/Desktop/AI_chatbot/Q&A_data/no_need_rag/data_{i}.csv')
                time.sleep(2.5)
        else:
            for i, file in tqdm(enumerate(os.listdir('/home/modar/Desktop/AI_chatbot/Q&A_data/no_need_rag'))):
                df = pd.read_csv('/home/modar/Desktop/AI_chatbot/Q&A_data/no_need_rag/' + file)
                questions = "\n".join([f"{i+1}- {q}" for i, q in enumerate(df['question'].tolist())])
                input_prompt = general_QA_prompt_rephrase.format(data=questions)
                result = model.invoke([SystemMessage(content=input_prompt)]).content
                if think:
                    result = remove_think_content(result)
                pattern = r"\d+\-\s*"
                data = re.split(pattern, result)
                data = [q.strip() for q in data if q.strip()]
                labels = ['no_rag' for _ in range(len(data))]
                file = pd.DataFrame({'question': data, 'label': labels})
                file.to_csv(f'/home/modar/Desktop/AI_chatbot/Q&A_data/no_need_rag/data_{n + i}.csv')
                time.sleep(2.5)

if __name__ == "__main__":
    generate_dataset(rag=False, think=True, rephrase=True)