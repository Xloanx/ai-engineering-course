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
