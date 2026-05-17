import os

from api.services.ner_service import NERService
from api.services.translate_service import TranslateService

_ner_service: NERService | None = None
_translate_service: TranslateService | None = None


def init_services():
    global _ner_service, _translate_service
    _ner_service = NERService()
    backend = os.environ.get("TRANSLATION_BACKEND", "google")
    _translate_service = TranslateService(_ner_service, backend=backend)


def get_ner_service() -> NERService:
    return _ner_service


def get_translate_service() -> TranslateService:
    return _translate_service
