from openai import OpenAI

client = OpenAI(api_key="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")

def analyze_text(prompt, model="gpt-4-turbo", temperature=0.2):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()
