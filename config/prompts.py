#### This file contains the necessary prompts to power the chatbot

from pydantic_settings import BaseSettings

class Prompts(BaseSettings):

    intent_classifier_prompt: str = '''You got the following user query: "{question}".
                Please return a valid JSON object with exactly two keys:
                - "answer": "yes" if the user query needs to retrieve historical/cultural information from the database (i.e. RAG is needed),
                            or "no" if it does not.
                - "translation": the translation of the user query into English, if it is already in english, return it unchanged.
                Do not include any additional text.
                '''
    intent_classifier_class_prompt: str = '''Return 'yes' if the query requires retrieving accurate and not general historical or cultural information (i.e., RAG is needed), 
                    or 'no' if it does not (i.e., RAG is not required).'''
    intent_classifier_translation_prompt: str = '''If the original query is in English, return it unchanged. 
                    Otherwise, provide an accurate translation of the query into English.'''
    grader_prompt: str = '''You are a teacher grading a quiz. You will be given: 
        1/ a QUESTION
        2/ A FACT provided by the student
        
        You are grading RELEVANCE RECALL:
        A score of 1 means that ANY of the statements in the FACT are relevant to the QUESTION. 
        A score of 0 means that NONE of the statements in the FACT are relevant to the QUESTION. 
        1 is the highest (best) score. 0 is the lowest score you can give. 
        
        Explain your reasoning in a step-by-step manner. Ensure your reasoning and conclusion are correct. 
        
        Avoid simply stating the correct answer at the outset.
        
        Question: {question} \n
        Fact: \n\n {documents} \n\n
        
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    '''
    search_keywords_generation_prompt: str = '''
                Convert the user query into a Wikipedia search keyword,
                ensuring correct understanding of the query.
                The context is Syrian history and culture.
                Return **only the keyword**: {query}'''
    
    generation_prompt: str = '''You are an assistant for question-answering tasks (specifically syrian heritage). 
            You've got a summary of your current conversation (it may be empty): {memory}.
            
            Use the following documents to augment your internal knowledge to answer to the question. 
            
            - If you don't know the answer, just say that you don't know. 
            - If the user asks a simple question, just answer politely (such as greeting).
            - Do not discuss anything outside the scope of syrian heritage, chat about syrian history and only syrian history.
            - Answer in the language of the question.
            - Put in mind that your source of information is the historical texts in the database, and the realtime web search results using tavily and duckduckgo.
            
            Question: {question} 
            Documents: {documents} 
            Answer:  '''
        
    general_QA_prompt: str = '''
            I am building a dataset for an intent classifier with two classes:  
            1️⃣ **"need rag"** (questions that require retrieving information from documents)  
            2️⃣ **"no need rag"** (questions that can be answered directly by a general chatbot).  

            Your task is to generate **"no need rag"** questions, specifically about **Syrian culture, history, and general knowledge**.  
'
            ### **Instructions:**  
            You must generate **diverse** sets of **simple** questions and **casual conversational phrases** related to Syrian culture, history, and general interactions, that can be answered **without retrieval**. Avoid queries that require document or fact-based retrieval.

            #### **1️⃣ Question Types**  
            - **General Knowledge Questions** (short and simple, focused on common knowledge)  
            - Example: "Who was the first ruler of Aleppo?"  
            - Example: "When was the Umayyad Mosque built?"  
            - **Conversational & Interactive Queries**  
            - Example: "Tell me something interesting about Syria."  
            - Example: "What do you think about ancient Syrian architecture?"  
            - **Casual Phrases in Chat Flow**  
            - Example: "Hello!"  
            - Example: "Good morning, how are you?"  
            - Example: "Wow, that’s interesting!"  
            - Example: "Thank you!"  
            - Example: "What’s your favorite Syrian food?"  

            #### **2️⃣ Guidelines for Questions:**  
            ✅ Keep them **simple**, **conversational**, and **easy to answer** without relying on documents or specific references.  
            ✅ **DO NOT** ask questions requiring obscure, highly specific, or book-dependent knowledge.  
            ✅ Avoid using **academic or highly historical questions** that necessitate retrieval.  
            ✅ Ensure **variety** in sentence structure (e.g., What, How, When, Tell me, Describe, Greetings, Expressing interests).  
            ✅ Mix between **long and short** questions.  
            ✅ **DO NOT** ask about the **Syrian Civil War**.  

            #### **3️⃣ Conversational Phrases to Avoid:**  
            To make sure there is no confusion, **avoid** questions or greetings that could be misinterpreted as needing document-based answers:
            - **"Tell me"** and similar constructions should focus on **simple** information, not detailed facts (e.g., "Tell me about Syria" is fine; "Tell me about the collapse of the Aleppo economy in the 15th century" is not).  
            - **Casual greetings** and conversational phrases like **"Good morning"**, **"Hello!"**, **"How are you?"**, **"What’s up?"**, or **"Tell me a joke"** should be included as **no need rag**.  

            ### **Response Format (Strictly Follow This Format AND ONLY THIS Format, DO NOT REPLY WITH SOMETHING ELSE)**  
            1- [Question]  
            2- [Question]  
            3- [Question]  
            ...

            **DO NOT** include explanations, extra formatting, or any other text.  

            '''
    general_QA_prompt_rephrase: str = '''
            I am building a dataset for an **intent classifier** with two classes:  
            1️⃣ **"need rag"** → Questions that require retrieving information from documents.  
            2️⃣ **"no need rag"** → Questions that can be answered **directly** by a general chatbot.  

            You will receive an existing dataset and must **augment it** by generating **new** questions or rephrasing existing ones while strictly following the instructions below.  

            ---
            ## 🔹 **Task: Generate "No Need RAG" Questions**  
            You must generate **diverse**, **simple**, and **conversational** questions related to **Syrian culture, history, and general knowledge** that **DO NOT** require document retrieval.  

            ---
            ## 🔹 **Question Types to Generate**  
            ✅ **General Knowledge (Directly Answerable)**  
            - Example: *"Who was the first ruler of Aleppo?"*  
            - Example: *"When was the Umayyad Mosque built?"*  

            ✅ **Casual & Conversational Queries**  
            - Example: *"Tell me something interesting about Syria."*  
            - Example: *"What do you think about Syrian architecture?"*  

            ✅ **Short Chatbot Phrases**  
            - Example: *"Hello!"*  
            - Example: *"What’s your favorite Syrian dish?"*  
            - Example: *"Wow, that’s amazing!"*  

            ---
            ## 🔹 **Guidelines**  
            ✔ **Keep it simple** → No complex, academic, or fact-heavy questions.  
            ✔ **Ensure variety** → Mix **What, How, When, Tell me, Describe** questions with **greetings, expressions, and casual prompts**.  
            ✔ **No dependency on external documents** → Questions should be **answerable immediately**.  
            ✔ **DO NOT ask about the Syrian Civil War**.  
            ✔ **Mix long & short questions** to improve dataset diversity.  

            ---
            ## 🔹 **Avoid These Mistakes**  
            ❌ **No document-based or research-heavy questions**.  
            - 🚫 *"Tell me about the collapse of Aleppo’s economy in the 15th century."*  
            - 🚫 *"What were the key causes of the Mamluk-Ottoman conflicts in Syria?"*  

            ❌ **No retrieval-dependent historical or political questions**.  
            - 🚫 *"Who wrote the oldest book on Syrian history?"*  
            - 🚫 *"How did trade routes affect Syria in the 12th century?"*  

            ❌ **No overly vague or unanswerable questions**.  
            - 🚫 *"What is the meaning of life?"*  
            - 🚫 *"Can you tell me everything about Syria?"*  

            ---
            ## 🔹 **Response Format (STRICTLY FOLLOW THIS FORMAT)**  
            1- [Question]  
            2- [Question]  
            3- [Question]  
            …  

            ⚠ **DO NOT** include explanations, bullet points, extra formatting, or any text outside the required format. 


            ---

            🔹 **Why This is Better?**  
            ✅ **Highly constrained yet flexible** → Ensures high-quality data generation.  
            ✅ **Prevents unwanted outputs** → Limits irrelevant or retrieval-based questions.  
            ✅ **Optimized for NLP models** → Clear task definition, examples, and strict output structure.  

            The file data: {data}
            '''
    QA_genration_prompt: str = '''
            Generate a list of high-quality questions based on the given text to train a RAG system.
            - The questions should be designed as if asked to a model who cannot see the text, so each question should be clear and self-contained, every term should be clear and mentioning ambiguos terms.
            - Avoid referencing specific entities, authors, or sections of the text.
            - The questions should be varied in type: some should require short, specific answers, and others should require more detailed responses.
            - Keep the questions concise and focused, ensuring they are suitable for assessing comprehension and critical thinking.
            - The questions should only and only focus on heritage, history and culture.
            - Don't write something like 'according to the text' in any of the questions, there is not text seen by the model we are testing, it only sees your questions separately from each other.
            ANSWER WITH THIS FORMAAT AND ONLY THIS FORMAT:
            1- [Question]
            2- [Question]
            ...
            DO NOT WRITE ANYTHING IN THE RESPONSE OTHER THAN THE MENTIONED FORMAT.
            ### **Text to generate questions on:**
            {text}
        '''
    PARAPHRASE_PROMPT: str = '''
            I am refining a dataset of questions that need retrieval-augmented generation (RAG).
            Your task is to rephrase the given questions to make them sound more natural and human-like,
            while still ensuring they require RAG. Use variations like:
            - Tell me...
            - Wow, could you tell me more about...
            - I want to know about...
            - What do you think about...
            - Can you explain...
            - Do you have any insights on...
            - Give me details on...
            - I'm curious about...
            - Expand on...
            - How would you describe...
            - And leave some question as they are to maintain all cases (user may ask the question directly).
            
            Ensure the meaning remains the same, and keep responses in the following strict format:
            1- [Refined Question]
            2- [Refined Question]
            ...
            DO NOT INCLUDE ANYTHING ELSE.
            
            Here are the original questions:
            {questions}
        '''

        
prompts = Prompts()