"""Pure parsing/scrubbing helpers, designed to run inside `datasets.map`/`filter`.

`parse()` returns a flat dict of curated columns (never raises, never returns
None) so it composes with Arrow's map; `is_valid()` is the companion filter.
"""

import json
import re

MIN_CHARS = 600
MIN_PRICE = 0.5
MAX_PRICE = 999.49
MAX_TEXT_EACH = 3000
MAX_TEXT_TOTAL = 4000

REMOVALS = [
    "Part Number",
    "Best Sellers Rank",
    "Batteries Included?",
    "Batteries Required?",
    "Item model number",
]

# A run of >=7 chars that mixes letters and digits -> almost always a part number.
_CODE_RE = re.compile(r"\b(?=[A-Z0-9]{7,}\b)(?=.*[A-Z])(?=.*\d)[A-Z0-9]+\b")
_WS_RE = re.compile(r"\s+")

# Multiplier to convert each supported unit to pounds.
WEIGHT_UNITS = {
    "pounds": 1.0,
    "ounces": 1 / 16,
    "grams": 1 / 453.592,
    "milligrams": 1 / 453592,
    "kilograms": 1 / 0.453592,
}


def simplify(text_list) -> str:
    """Collapse all whitespace to single spaces and cap the length."""
    return _WS_RE.sub(" ", str(text_list)).strip()[:MAX_TEXT_EACH]


def scrub(title, description, features, details) -> str:
    """Build one cleansed string with part numbers and noise fields removed."""
    for remove in REMOVALS:
        details.pop(remove, None)
    parts = [title]
    if description:
        parts.append(simplify(description))
    if features:
        parts.append(simplify(features))
    if details:
        parts.append(json.dumps(details))
    result = "\n".join(parts) + "\n"
    return _CODE_RE.sub("", result).strip()[:MAX_TEXT_TOTAL]


def get_weight(details) -> float:
    """Weight in pounds, or 0.0 if absent/unparseable."""
    weight_str = details.get("Item Weight")
    if not weight_str:
        return 0.0
    try:
        parts = weight_str.split(" ")
        amount = float(parts[0])
        unit = parts[1].lower()
        if unit in WEIGHT_UNITS:
            return amount * WEIGHT_UNITS[unit]
        if unit == "hundredths" and parts[2].lower() == "pounds":
            return amount / 100
    except (ValueError, IndexError):
        pass
    return 0.0


def parse(datapoint, category) -> dict:
    """Map a raw datapoint to curated columns. Unusable rows get price=NaN /
    full="" so that `is_valid` can drop them in a following `.filter`."""
    try:
        price = float(datapoint["price"])
    except (ValueError, TypeError):
        price = float("nan")
    details = json.loads(datapoint["details"] or "{}")
    return {
        "title": datapoint["title"],
        "category": category,
        "price": price,
        "full": scrub(
            datapoint["title"],
            datapoint["description"],
            datapoint["features"],
            details,
        ),
        "weight": get_weight(details),
    }


def is_valid(row) -> bool:
    """Keep items priced in range with enough descriptive text. NaN prices
    fail the comparison and are dropped."""
    return MIN_PRICE <= row["price"] <= MAX_PRICE and len(row["full"]) >= MIN_CHARS
