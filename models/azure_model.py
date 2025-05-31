import openai
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM

class AzureChatModelWrapper(DeepEvalBaseLLM):
    """Wrapper for AzureChatOpenAI to make it compatible with DeepEval."""
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        return (await self.model.ainvoke(prompt)).content

    def get_model_name(self):
        return "azure-gpt4o-mini"

def initialize_models(config):
    """Initialize Azure OpenAI embeddings and chat models."""
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=config['azure_openai_endpoint'],
        azure_deployment=config['embeddings_deployment_name'],
        openai_api_version=config['openai_api_version'],
        model=config['embeddings_deployment_name'],
        api_key=config['subscription_key']
    )

    chat_client = openai.AzureOpenAI(
        azure_endpoint=config['azure_openai_endpoint'],
        api_version=config['openai_api_version'],
        azure_deployment=config['chat_deployment_name']
    )

    chat_model = AzureChatOpenAI(
        openai_api_version=config['openai_api_version'],
        azure_deployment=config['chat_deployment_name'],
        azure_endpoint=config['azure_openai_endpoint']
    )

    wrapped_model = AzureChatModelWrapper(chat_model)
    return embeddings, chat_client, chat_model, wrapped_model