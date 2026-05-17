from deep_translator import MyMemoryTranslator as _MyMemoryTranslator
from api.services.translation.base import Translator

# MyMemory Hub (via deep-translator) often requires full language names
# and does not support "auto" source detection in this version.
ISO_TO_FULL = {
    "en": "english",
    "hi": "hindi",
    "hi-IN": "hindi",
    "es": "spanish",
    "fr": "french",
    "ta": "tamil",
    "te": "telugu",
    "kn": "kannada",
    "ml": "malayalam",
    "gu": "gujarati",
    "bn": "bengali",
    "pa": "punjabi",
}


class MyMemoryTranslator(Translator):
    @property
    def name(self) -> str:
        return "mymemory"

    def translate(self, text: str, target_lang: str) -> str:
        if not text.strip():
            return text

        # Default to "english" source and resolve target language full name
        full_target = ISO_TO_FULL.get(target_lang, target_lang)

        try:
            return _MyMemoryTranslator(source="english", target=full_target).translate(text)
        except Exception as e:
            print(f"[mymemory] translation failed: {e}")
            raise RuntimeWarning(f"MyMemory translation failed: {e}")
