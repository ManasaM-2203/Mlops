from api.services.translation.base import Translator
from api.services.translation.google import GoogleTranslator
from api.services.translation.marian import MarianTranslator
from api.services.translation.mymemory import MyMemoryTranslator

_BACKENDS = {
    "google": GoogleTranslator,
    "marian": MarianTranslator,
    "mymemory": MyMemoryTranslator,
}


def get_translator(backend: str = "google") -> Translator:
    cls = _BACKENDS.get(backend)
    if cls is None:
        raise ValueError(f"unknown translation backend: {backend}. available: {list(_BACKENDS.keys())}")
    return cls()
