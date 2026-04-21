class LLMClient:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def generate(self, prompt: str, max_tokens=20, temp=0.0):
        return self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=temp,
            max_output_tokens=max_tokens,
        ).output_text.strip()