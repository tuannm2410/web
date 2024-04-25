import os
import openai
from openai import OpenAI

key = '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' 
client = openai.OpenAI(api_key=key)

def create_text(prompt):
    completion = client.chat.completions.create( # Change the method name
    model = 'gpt-3.5-turbo',
    messages = [ # Change the prompt parameter to messages parameter
        {'role': 'user', 'content': prompt}
    ],
    temperature = 0  
    )
    return completion.choices[0].message.content





