import pytest
from ai_test_framework.core.extractors import extract


class TestExtractNumeric:
    def test_integer_in_sentence(self):
        assert extract("The answer is 42.", "numeric") == "42"

    def test_decimal_normalised(self):
        assert extract("Result: 4.0", "numeric") == "4"

    def test_negative_number(self):
        assert extract("Temperature is -3 degrees", "numeric") == "-3"

    def test_first_number_returned(self):
        assert extract("2 cats and 5 dogs", "numeric") == "2"

    def test_no_number_returns_none(self):
        assert extract("no numbers here", "numeric") is None

    def test_decimal_preserved(self):
        assert extract("score: 3.14", "numeric") == "3.14"


class TestExtractYN:
    def test_plain_yes(self):
        assert extract("yes", "yn") == "yes"

    def test_plain_no(self):
        assert extract("no", "yn") == "no"

    def test_yes_in_sentence(self):
        assert extract("Yes, that is correct.", "yn") == "yes"

    def test_no_in_sentence(self):
        assert extract("No, it is not.", "yn") == "no"

    def test_case_insensitive(self):
        assert extract("YES", "yn") == "yes"

    def test_neither_returns_none(self):
        assert extract("maybe so", "yn") is None


class TestExtractLetter:
    def test_single_letter_a(self):
        assert extract("A", "letter") == "A"

    def test_letter_in_sentence(self):
        assert extract("The answer is B.", "letter") == "B"

    def test_lowercase_normalised(self):
        assert extract("answer: c", "letter") == "C"

    def test_no_letter_returns_none(self):
        assert extract("none here", "letter") is None

    def test_first_letter_returned(self):
        assert extract("A or B", "letter") == "A"


class TestExtractExact:
    def test_strips_and_lowercases(self):
        assert extract("  Hello World  ", "exact") == "hello world"

    def test_empty_string(self):
        assert extract("", "exact") == ""


class TestExtractDispatch:
    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            extract("text", "unknown_method")
