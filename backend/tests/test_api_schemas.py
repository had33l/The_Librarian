import pytest
from pydantic import ValidationError

from src.api.schemas import AskRequest


def test_question_is_trimmed():
    request = AskRequest(
        question="  What is a Python decorator?  "
    )

    assert request.question == "What is a Python decorator?"


def test_blank_question_is_rejected():
    with pytest.raises(ValidationError):
        AskRequest(question="   ")


def test_too_short_question_is_rejected():
    with pytest.raises(ValidationError):
        AskRequest(question="Hi")


def test_valid_question_is_accepted():
    request = AskRequest(
        question="What is a Python decorator?"
    )

    assert request.question == "What is a Python decorator?"