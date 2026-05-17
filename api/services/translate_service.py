import re
from typing import TYPE_CHECKING
from api.services.translation.factory import get_translator
from api.services.translation.transliterate_util import transliterate_text, clean_hindi_artifacts

if TYPE_CHECKING:
    from api.services.ner_service import NERService


class TranslateService:
    def __init__(self, ner: "NERService", backend: str = "google"):
        self.ner = ner
        self.translator = get_translator(backend)

    def translate(self, text: str, target_lang: str, backend: str | None = None, manual_entities: list[dict] = None, transliterate: bool = True) -> dict:
        text = clean_hindi_artifacts(text)
        tokens, tags, confidences, cleaned_text = self.ner.predict(text)
        entities = self.ner.extract_entities(tokens, tags, text=cleaned_text, confidences=confidences, manual_entities=manual_entities)

        translator = self.translator
        if backend and backend != self.translator.name:
            from api.services.translation.factory import get_translator
            translator = get_translator(backend)

        masked_text, placeholders = self._mask_entities(cleaned_text, entities)
        translated = translator.translate(masked_text, target_lang)
        final_text = self._restore_entities(translated, placeholders, target_lang, transliterate)

        # Post-process to remove any leaked Latin artifacts from the final Hindi text
        final_text = clean_hindi_artifacts(final_text)

        return {
            "source_text": cleaned_text,
            "translated_text": final_text,
            "entities": entities,
            "target_lang": target_lang,
        }

    def _mask_entities(self, text: str, entities: list[dict]) -> tuple[str, dict]:
        placeholders = {}
        sorted_entities = sorted(entities, key=lambda e: e["start"], reverse=True)

        for ent in sorted_entities:
            placeholder = f"__ENT{len(placeholders)}__"
            placeholders[placeholder] = ent["text"]
            text = text[:ent["start"]] + placeholder + text[ent["end"]:]

        return text, placeholders

    def _restore_entities(self, text: str, placeholders: dict, target_lang: str, transliterate: bool) -> str:
        def replace_match(match):
            # Extract the ID from the mangled placeholder (e.g., "__ ENT 0 __" -> 0)
            id_match = re.search(r"\d+", match.group(0))
            if not id_match:
                return match.group(0)
                
            ent_id = id_match.group()
            placeholder_key = f"__ENT{ent_id}__"

            original = placeholders.get(placeholder_key)
            if not original:
                return match.group(0)
                
            replacement = original
            if transliterate:
                replacement = transliterate_text(original, target_lang)
            return replacement

        # Regex to find markers like __ENT0__, __ ENT 0 __, __ent 0__, etc.
        pattern = r"__\s*ENT\s*\d+\s*__"
        return re.sub(pattern, replace_match, text, flags=re.IGNORECASE)
