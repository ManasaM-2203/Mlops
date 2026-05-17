from fastapi import APIRouter, Depends

from api.schemas.models import TranslateRequest, TranslateResponse, Entity
from api.deps import get_translate_service

router = APIRouter(prefix="/translate", tags=["translate"])


@router.post("", response_model=TranslateResponse)
def translate(req: TranslateRequest, svc=Depends(get_translate_service)):
    manual_entities = [e.model_dump() for e in req.manual_entities]
    result = svc.translate(req.text, req.target_lang, backend=req.backend, manual_entities=manual_entities, transliterate=req.transliterate)
    return TranslateResponse(
        source_text=result["source_text"],
        translated_text=result["translated_text"],
        entities=[Entity(**e) for e in result["entities"]],
        target_lang=result["target_lang"],
    )
