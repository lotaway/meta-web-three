import os
from openai import OpenAI

def start():
    client = OpenAI.chat.completions.create(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    response = client.responses.create(
        model="gpt-4o",
        instructions="You are a coding assistant that talks like a pirate.",
        input="How do I check if a Python object is an instance of a class?",
    )
    for i in range(3):
        response = client.chat(

        )