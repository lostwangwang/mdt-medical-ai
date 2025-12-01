from openai import OpenAI


class LLMClient:
    def __init__(self, model_name, api_key, api_base):
        self.client = None
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base

    def init_client(self):
        self.client = OpenAI(
            api_key=self.api_key,  # 你的 Agent Token
            base_url=self.api_base,
        )

    def ask_model(self, role: str, content: str, prompt: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": role,
                        "content": content,
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            response_content = response.choices[0].message.content.strip()
        except Exception as e:
            response_content = f"[Error] {e}"

        return response_content
