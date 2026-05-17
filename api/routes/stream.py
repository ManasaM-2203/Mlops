from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

_entity_store: list[dict] = []


@router.websocket("/ws/translate")
async def ws_translate(ws: WebSocket):
    await ws.accept()
    from api.deps import _ner_service, _translate_service
    ner = _ner_service
    svc = _translate_service

    try:
        while True:
            data = await ws.receive_json()
            text = data.get("text", "")
            target_lang = data.get("target_lang", "es")

            tokens, tags = ner.predict(text)
            entities = ner.extract_entities(tokens, tags)

            for ent in entities:
                if not any(e["text"] == ent["text"] and e["label"] == ent["label"] for e in _entity_store):
                    _entity_store.append(ent)

            result = svc.translate(text, target_lang)

            await ws.send_json({
                "type": "ner",
                "tokens": tokens,
                "tags": tags,
                "entities": entities,
            })
            await ws.send_json({
                "type": "translation",
                "translated_text": result["translated_text"],
                "entities": result["entities"],
            })
    except WebSocketDisconnect:
        pass


@router.get("/entities")
def get_collected_entities():
    return _entity_store


@router.delete("/entities")
def clear_entities():
    _entity_store.clear()
    return {"status": "cleared"}
