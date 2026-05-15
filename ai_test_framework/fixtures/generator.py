from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from ai_test_framework.tests.consistency import SurfaceFramingConsistencyTest, FramingVariant


def generate_variants(
    template: str,
    substitutions: list[dict],
    expected: str,
    label_key: Optional[str] = None,
) -> list[FramingVariant]:
    """Generate FramingVariant instances from a prompt template and substitution dicts.

    Template uses str.format_map() syntax — single braces: {name}, {thing}, etc.

    Label resolution order for each substitution dict:
      1. "label" key present in the dict
      2. Value of `label_key` in the dict (if label_key is specified)
      3. f"variant_{i}" fallback

    Example:
        variants = generate_variants(
            template="{name} has 3 {thing}s. How many {thing}s? Reply with just the number.",
            substitutions=[
                {"label": "alice",   "name": "Alice",  "thing": "apple"},
                {"label": "carlos",  "name": "Carlos", "thing": "orange"},
                {"label": "library", "name": "A library", "thing": "book"},
            ],
            expected="3",
        )
    """
    variants: list[FramingVariant] = []
    for i, subs in enumerate(substitutions):
        try:
            prompt = template.format_map(subs)
        except KeyError as exc:
            raise ValueError(
                f"Template key {exc} missing from substitution dict at index {i}: {subs}"
            ) from exc

        if "label" in subs:
            label = str(subs["label"])
        elif label_key and label_key in subs:
            label = str(subs[label_key])
        else:
            label = f"variant_{i}"

        variants.append(FramingVariant(label=label, prompt=prompt, expected=expected))
    return variants


@dataclass
class VariantTemplate:
    """Combines a prompt template with substitution dicts to build a SurfaceFramingConsistencyTest.

    Use when building fixture libraries programmatically — avoids copy-pasting
    prompt text across variants that differ only in names, objects, or context.

    Example:
        template = VariantTemplate(
            prompt_template=(
                "{name} has {start} {thing}s. Someone gives them {give} more. "
                "They use {use}. How many {thing}s does {name} have? "
                "Reply with just the number."
            ),
            substitutions=[
                {"label": "alice_apples",   "name": "Alice",   "thing": "apple",  "start": 3, "give": 2, "use": 1},
                {"label": "carlos_oranges", "name": "Carlos",  "thing": "orange", "start": 3, "give": 2, "use": 1},
                {"label": "shelf_books",    "name": "A shelf", "thing": "book",   "start": 3, "give": 2, "use": 1},
            ],
            expected_answer="4",
            extraction_method="numeric",
        )
        test = template.build(name="give_and_take")
        suite.add(test)
    """
    prompt_template: str
    substitutions: list[dict]
    expected_answer: str
    extraction_method: str
    label_key: Optional[str] = None

    def variants(self) -> list[FramingVariant]:
        return generate_variants(
            template=self.prompt_template,
            substitutions=self.substitutions,
            expected=self.expected_answer,
            label_key=self.label_key,
        )

    def build(self, name: Optional[str] = None) -> SurfaceFramingConsistencyTest:
        return SurfaceFramingConsistencyTest(
            variants=self.variants(),
            extraction_method=self.extraction_method,
            expected_answer=self.expected_answer,
            name=name,
        )

    def add_substitution(self, subs: dict) -> "VariantTemplate":
        """Return a new VariantTemplate with one more substitution appended."""
        return VariantTemplate(
            prompt_template=self.prompt_template,
            substitutions=self.substitutions + [subs],
            expected_answer=self.expected_answer,
            extraction_method=self.extraction_method,
            label_key=self.label_key,
        )
