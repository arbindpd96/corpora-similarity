import os

from openai import OpenAI

client = OpenAI(api_key= os.environ.get('OPENAI_API_KEY'))

def infer_gpt_4(system_prompt, user_prompt, temperature):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            model="gpt-4-1106-preview",
            temperature=temperature
        )

        return chat_completion.choices[0].message.content
    except Exception as e:
        return e
