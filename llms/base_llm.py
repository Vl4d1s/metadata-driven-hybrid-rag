from langchain_openai import ChatOpenAI

def get_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.0) -> ChatOpenAI:
    """
    Returns a ChatOpenAI instance with the specified model name and temperature.

    Args:
        model_name (str): The name of the OpenAI model to use.
        temperature (float): The temperature setting for the model. Defaults to 0.0.

    Returns:
        ChatOpenAI: An instance of the ChatOpenAI class configured with the specified parameters.
    """
    return ChatOpenAI(model=model_name, temperature=temperature)