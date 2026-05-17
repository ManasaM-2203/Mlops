from transformers import MarianMTModel, MarianTokenizer
from api.services.translation.base import Translator

_model_cache = {}


class MarianTranslator(Translator):
    @property
    def name(self) -> str:
        return "marian"

    def translate(self, text: str, target_lang: str) -> str:
        if not text.strip():
            return text

        model_name = f"Helsinki-NLP/opus-mt-en-{target_lang}"
        if model_name not in _model_cache:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            _model_cache[model_name] = (tokenizer, model)

        tokenizer, model = _model_cache[model_name]
        tokens = tokenizer(text, return_tensors="pt", truncation=True)
        translated = model.generate(**tokens)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
