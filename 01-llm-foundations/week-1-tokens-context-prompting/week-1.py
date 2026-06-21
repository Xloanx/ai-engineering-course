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
# 2-basic-classification-with-anthropic.py
import os
import time
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a document classifier for an enterprise procurement system.

Classify each purchase order into exactly ONE of these categories:
- CAPEX: Capital expenditure. Assets with lifespan > 3 years. Hardware, infrastructure, property.
- OPEX: Operational expenditure. Recurring costs. SaaS, consumables, services, maintenance.
- EMERGENCY: Urgent operational necessity. Business continuity at risk if delayed.
- REJECTED: ANY of vendor, amount, or description is absent from the text. This rule
  applies regardless of what other information is present, even if the category would
  otherwise be obvious. Check for REJECTED first, before applying any other rule.

Rules:
- Respond with ONLY the category label. No explanation, no punctuation.
- When in doubt between CAPEX and OPEX, apply the >3 year lifespan rule strictly."""


def classify_document(document: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=10,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": document}]
    )
    return response.content[0].text.strip()


def classify_with_few_shot(document: str) -> str:
    messages = [
        {"role": "user", "content": "Vendor: Microsoft. Amount: $12,000/year. Description: Microsoft 365 licenses for 100 users."},
        {"role": "assistant", "content": "OPEX"},
        {"role": "user", "content": "Vendor: Caterpillar. Amount: $2,300,000. Description: Construction excavator for site development."},
        {"role": "assistant", "content": "CAPEX"},
        {"role": "user", "content": document}
    ]

    response = client.messages.create(
        model=MODEL,
        max_tokens=10,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=messages
    )
    return response.content[0].text.strip()


test_docs = [
    ("Clear OPEX",      "Vendor: AWS. Amount: $2,400/month. Description: Cloud compute for CI/CD pipeline."),
    ("Clear CAPEX",     "Vendor: Dell. Amount: $180,000. Description: Server rack for new data centre."),
    ("Missing vendor",  "Amount: $500. Description: Office supplies."),
    ("Ambiguous",       "Vendor: Adobe. Amount: $8,000. Description: Adobe Creative Suite perpetual license."),
]

print(f"{'Case':<20} {'No few-shot':<15} {'With few-shot':<15} {'Match?'}")
print("-" * 60)
for label, doc in test_docs:
    basic   = classify_document(doc)
    fewshot = classify_with_few_shot(doc)
    match   = "✓" if basic == fewshot else "← DIVERGES"
    print(f"{label:<20} {basic:<15} {fewshot:<15} {match}")
# 3-structured-outputs-with-anthropic.py
import os
import json
import re
import anthropic
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

MODEL = "claude-sonnet-4-6"


class PurchaseOrderClassification(BaseModel):
    category:              Literal["CAPEX", "OPEX", "EMERGENCY", "REJECTED"]
    confidence:             Literal["high", "medium", "low"]
    reasoning:              str = Field(description="One sentence justification")
    requires_cfo_approval:  bool = Field(description="True if amount exceeds $50,000")
    amount_extracted:       float | None = Field(default=None)


STRUCTURED_SYSTEM_PROMPT = """You are a document classifier for an enterprise procurement system.

Return a JSON object with exactly these fields:
{
  "category": "CAPEX" | "OPEX" | "EMERGENCY" | "REJECTED",
  "confidence": "high" | "medium" | "low",
  "reasoning": "<one sentence>",
  "requires_cfo_approval": true | false,
  "amount_extracted": <number or null>
}

Classification rules:
- REJECTED: ANY of vendor, amount, or description is absent. Check this first.
- CAPEX: Assets with lifespan > 3 years.
- OPEX: Recurring costs, SaaS, services.
- EMERGENCY: Business continuity at risk.
- requires_cfo_approval: true if amount exceeds $50,000.

Return ONLY the JSON object. No markdown fences, no explanation outside the JSON."""


def extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract valid JSON from:\n{text}")


def classify_structured(document: str) -> PurchaseOrderClassification:
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        temperature=0,
        system=STRUCTURED_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": document}]
    )
    raw = response.content[0].text
    data = extract_json(raw)
    return PurchaseOrderClassification(**data)


test_cases = [
    ("Clear OPEX",           "Vendor: AWS. Amount: $2,400/month. Description: Cloud compute for CI/CD pipeline."),
    ("Clear CAPEX",          "Vendor: Dell. Amount: $180,000. Description: Server rack for new data centre expansion."),
    ("CFO approval trigger", "Vendor: Cisco. Amount: $75,000. Description: Network switches for Building A."),
    ("Missing vendor",       "Amount: $500. Description: Office supplies."),
    ("Ambiguous",            "Vendor: Adobe. Amount: $8,000. Description: Adobe Creative Suite perpetual license."),
]

for label, doc in test_cases:
    result = classify_structured(doc)
    print(f"\n[{label}]")
    print(f"  category:          {result.category}")
    print(f"  confidence:        {result.confidence}")
    print(f"  requires_cfo:      {result.requires_cfo_approval}")
    print(f"  amount_extracted:  {result.amount_extracted}")
    print(f"  reasoning:         {result.reasoning}")
