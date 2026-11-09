from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown

client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
)

response = client.responses.create(
    model="qwen/qwen3-4b-2507",
    input="What's the planet earth made of?",
    temperature=1.0,
)

content = response.output_text
Console().print(Markdown(content))