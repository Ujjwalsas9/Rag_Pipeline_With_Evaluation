import os
from dotenv import load_dotenv

def load_config():
    load_dotenv('UAIS_NEW.env')
    config = {
        'azure_openai_endpoint': os.environ.get("MODEL_ENDPOINT"),
        'openai_api_version': os.environ.get("API_VERSION"),
        'embeddings_deployment_name': os.environ.get("EMBEDDINGS_MODEL_NAME"),
        'chat_deployment_name': os.environ.get("CHAT_MODEL_NAME"),
        'subscription_key': os.environ.get("AZURE_OPENAI_API_KEY"),
        'persist_directory': "vector_store_data/tk.db"
    }
    if not all(config.values()):
        raise ValueError("Missing environment variables in UAIS_NEW.env")
    return config