from ai_test_framework.fixtures.logic_problems import FIXTURES as _LOGIC_FIXTURES
from ai_test_framework.fixtures.causal_fixtures import CAUSAL_FIXTURES
from ai_test_framework.fixtures.generator import generate_variants, VariantTemplate

FIXTURES = {**_LOGIC_FIXTURES, **CAUSAL_FIXTURES}

__all__ = ["FIXTURES", "CAUSAL_FIXTURES", "generate_variants", "VariantTemplate"]
