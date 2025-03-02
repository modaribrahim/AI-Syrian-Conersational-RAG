from pydantic_settings import BaseSettings
from pydantic import Field
from .constants import MODELS , EMBEDDING_MODELS

class Settings(BaseSettings):
    model_name: str = MODELS['regular'][0]
    embedding_model: str = EMBEDDING_MODELS[1]
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    tavily_api_key: str = Field(..., env="TAVILY_API_KEY")
    top_k: int = 2
    normalize_embeddings: bool = True
    device: str = "cpu"
    index: str = "/home/modar/Desktop/AI_Agent_final_version/data_scripts/vector_db_generation/my_vector_db.index"
    docs: str =  "/home/modar/Desktop/AI_Agent_final_version/data_scripts/vector_db_generation/my_vector_db.pkl"
    books_path: str = "/home/modar/Desktop/"
    books: list = ["ancient-syria.pdf", "History_of_syria.pdf"]

    class Config:
        env_file = "/home/modar/Desktop/AI_Agent_final_version/.env"
        env_file_encoding = "utf-8"

settings = Settings()
