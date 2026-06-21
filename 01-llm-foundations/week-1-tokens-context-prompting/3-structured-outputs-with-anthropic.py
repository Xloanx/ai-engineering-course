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
