import re
import json
from guardrails import Guard, OnFailAction, Validator, register_validator
from guardrails.validators import PassResult, FailResult

# 1. PII Detector Validator
@register_validator(name="pii-detector", data_type="string")
class PIIDetector(Validator):
    """
    Detects and redacts Personally Identifiable Information (PII).
    """
    PII_PATTERNS = {
        "EMAIL":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE":       r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",
        "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }

    def validate(self, value: str, metadata: dict):
        redacted_text = value
        found_pii = []

        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, value)
            for match in matches:
                redacted_text = redacted_text.replace(match, f"[{pii_type}_REDACTED]")
                found_pii.append((pii_type, match))

        if found_pii:
            return FailResult(
                error_message=f"Detected {len(found_pii)} PII items",
                fix_value=redacted_text
            )
        return PassResult()

# 2. JSON Formatter Validator
@register_validator(name="json-formatter", data_type="string")
class JSONFormatter(Validator):
    """
    Validates and auto-repairs malformed JSON strings.
    """
    @staticmethod
    def _repair(text: str) -> str:
        text = text.strip()
        # Remove markdown fences
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$',          '', text)
        text = text.strip()
        # Single quotes → double quotes
        # Note: This is naive and might break if strings contain single quotes
        # But for this lab it's usually enough.
        text = text.replace("'", '"')
        # Remove trailing commas
        text = re.sub(r',\s*([}\]])', r'\1', text)
        return text

    def validate(self, value: str, metadata: dict):
        try:
            parsed = json.loads(value)
            return PassResult(value_override=json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            pass

        # Try repair
        try:
            repaired_text = self._repair(value)
            parsed = json.loads(repaired_text)
            return PassResult(value_override=json.dumps(parsed, indent=2))
        except json.JSONDecodeError as e:
            fallback = json.dumps({"error": "unrecoverable_json", "raw": value})
            return FailResult(
                error_message=f"Invalid JSON after repair attempt: {e}",
                fix_value=fallback
            )

# 3. PII Guard demo
def demo_pii_guard():
    print("\n" + "=" * 55)
    print("  PII Detection Demo")
    print("=" * 55)

    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Email",       "Contact John at john.doe@example.com for details."),
        ("Phone",       "Call our support line at (555) 867-5309."),
        ("SSN",         "Patient SSN is 123-45-6789 on file."),
        ("Credit Card", "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII",   "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean",       "No sensitive information in this text."),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        print(f"\n[{label}]")
        print(f"  Input:  {text}")
        print(f"  Output: {result.validated_output}")

# 4. JSON Guard demo
def demo_json_guard():
    print("\n" + "=" * 55)
    print("  JSON Formatting Demo")
    print("=" * 55)

    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Valid JSON",        '{"name": "Alice", "age": 30}'),
        ("Markdown fences",   '```json\n{"name": "Bob"}\n```'),
        ("Single quotes",     "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma",    '{"key": "value",}'),
        ("Truly invalid",     "This is not JSON at all: ??? {]"),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        status = "✅ Pass" if result.validation_passed else "❌ Repaired/Fail"
        print(f"\n[{label}] {status}")
        print(f"  Input:  {text[:60]}")
        print(f"  Output: {str(result.validated_output)[:60]}")

# 5. Main
def main():
    print("=" * 55)
    print("  Step 4: Guardrails AI Validators")
    print("=" * 55)

    demo_pii_guard()
    demo_json_guard()

    print("\n✅ Step 4 complete!")

if __name__ == "__main__":
    main()
