# 1-token-counting-with-anthropic.py
import os
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

MODEL = "claude-sonnet-4-6"

def count_tokens(text: str, system_prompt: str = "") -> dict:
    messages = [{"role": "user", "content": text}]
    kwargs = {"model": MODEL, "messages": messages}
    if system_prompt:
        kwargs["system"] = system_prompt

    response = client.messages.count_tokens(**kwargs)

    return {
        "input_tokens": response.input_tokens,
        "char_count":   len(text),
        "ratio":        round(len(text) / response.input_tokens, 2) if response.input_tokens else 0,
    }


samples = [
    "The purchase order workflow requires approval from the finance department.",
    "Vendor: Cisco Systems\nAmount: $45,000\nDescription: Network switch replacement",
    """
    SELECT po.id, po.vendor, po.amount, d.name as department
    FROM purchase_orders po
    JOIN departments d ON po.department_id = d.id
    WHERE po.status = 'PENDING' AND po.amount > 10000
    ORDER BY po.created_at DESC;
    """,
]

for sample in samples:
    result = count_tokens(sample)
    print(f"Tokens: {result['input_tokens']:4d} | "
          f"Chars: {result['char_count']:5d} | "
          f"Ratio: {result['ratio']} | "
          f"Preview: {sample[:50].strip()!r}")
