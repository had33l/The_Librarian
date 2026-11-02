from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The question asked about Python documentation."
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Question cannot be empty.")

        return value


class SourceItem(BaseModel):
    citation: str
    chunk_id: str | None = None
    title: str
    url: str


class TimingInfo(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    timing: TimingInfo
    token_usage: TokenUsage


class HealthResponse(BaseModel):
    status: str
    service: str
    chunks_loaded: int
    generation_model: str