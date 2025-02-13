import os
import json
import logging
import time
from tqdm import tqdm
from openai import AzureOpenAI
from openai.types.chat import ChatCompletion
from dotenv import load_dotenv
load_dotenv()

endpoint = os.getenv("ENDPOINT_URL", "https://preetam.openai.azure.com/")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

client = AzureOpenAI(  
    azure_endpoint=endpoint,  
    api_key=subscription_key,  
    api_version="2024-05-01-preview",
)

def invoke_gpt_endpoint(prompt, temperature, max_tokens, opeani_api_keys, engine="gpt-4o-mini",
                        max_retries: int = 10, backoff_factor: float = 5.0):
    for attempt in range(max_retries):
        try:
            #Prepare the chat prompt 
            chat_prompt = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
            # Include speech result if speech is enabled
            messages = chat_prompt
            # Generate the completion  
            completion: ChatCompletion = client.chat.completions.create(  
                model=engine,
                messages=messages,
                max_tokens=max_tokens,  
                temperature=temperature,  
                top_p=0.95,  
                frequency_penalty=0,  
                presence_penalty=0,
                stop=None,  
                stream=False
            )
            return completion
        except Exception as e:
            # If this was the last attempt, re-raise the error.
            if attempt == max_retries - 1:
                raise

            # Otherwise, back off exponentially before retrying.
            sleep_time = backoff_factor ** attempt
            time.sleep(sleep_time)