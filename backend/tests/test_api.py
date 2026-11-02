
"""Tests for The Librarian API request validation."""

import pytest

from pydantic import ValidationError

from src.api.schemas import AskRequest


def test_valid_question():

    request = AskRequest(
        question="What is a Python decorator?"
    )

    assert request.question == "What is a Python decorator?"


def test_question_is_trimmed():

    request = AskRequest(
        question="  What is Python?  "
    )

    assert request.question == "What is Python?"


def test_empty_question_is_rejected():

    with pytest.raises(ValidationError):
        AskRequest(question="")


def test_whitespace_question_is_rejected():

    with pytest.raises(ValidationError):
        AskRequest(question="   ")


def test_question_too_short_is_rejected():

    with pytest.raises(ValidationError):
        AskRequest(question="Hi")


def test_question_too_long_is_rejected():

    with pytest.raises(ValidationError):
        AskRequest(question="A" * 2001)


def test_question_within_limit_is_accepted():

    request = AskRequest(
        question="A" * 2000
    )

    assert len(request.question) == 2000
