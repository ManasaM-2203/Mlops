from fastapi import APIRouter, Depends

from api.schemas.models import NERRequest, NERResponse, Entity
from api.deps import get_ner_service

router = APIRouter(prefix="/ner", tags=["ner"])


@router.post("", response_model=NERResponse)
def predict_ner(req: NERRequest, ner=Depends(get_ner_service)):
    tokens, tags, confidences, text = ner.predict(req.text)
    entities = ner.extract_entities(tokens, tags, text=text, confidences=confidences)
    return NERResponse(
        tokens=tokens,
        tags=tags,
        entities=[Entity(**e) for e in entities],
    )
