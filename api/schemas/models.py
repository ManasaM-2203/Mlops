from pydantic import BaseModel


class NERRequest(BaseModel):
    text: str


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0


class NERResponse(BaseModel):
    tokens: list[str]
    tags: list[str]
    entities: list[Entity]


class TranslateRequest(BaseModel):
    text: str
    target_lang: str = "es"
    backend: str = "google"
    manual_entities: list[Entity] = []
    transliterate: bool = True


class TranslateResponse(BaseModel):
    source_text: str
    translated_text: str
    entities: list[Entity]
    target_lang: str


class StreamMessage(BaseModel):
    type: str
    data: dict
