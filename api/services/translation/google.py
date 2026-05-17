from deep_translator import GoogleTranslator as _GoogleTranslator
from api.services.translation.base import Translator


class GoogleTranslator(Translator):
    @property
    def name(self) -> str:
        return "google"

    def translate(self, text: str, target_lang: str) -> str:
        if not text.strip():
            return text
        return _GoogleTranslator(source="auto", target=target_lang).translate(text)
