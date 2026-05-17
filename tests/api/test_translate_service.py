from api.services.translate_service import TranslateService


class FakeNER:
    def predict(self, text):
        tokens = text.split()
        tags = ["O"] * len(tokens)
        for i, t in enumerate(tokens):
            if t[0].isupper() and t not in ("The", "A", "In", "On", "At", "Is"):
                tags[i] = "B-per"
        confidences = [1.0] * len(tokens)
        return tokens, tags, confidences, text

    def extract_entities(self, tokens, tags, text=None, confidences=None, manual_entities=None):
        entities = []
        for i, (t, tag) in enumerate(zip(tokens, tags)):
            if tag != "O":
                start = sum(len(tokens[j]) + 1 for j in range(i))
                entities.append({
                    "text": t,
                    "label": tag[2:],
                    "start": start,
                    "end": start + len(t),
                })
        return entities


class FakeTranslator:
    name = "fake"

    def translate(self, text, target_lang):
        return text.upper()


def _make_service():
    svc = TranslateService.__new__(TranslateService)
    svc.ner = FakeNER()
    svc.translator = FakeTranslator()
    return svc


def test_entity_masking():
    svc = _make_service()
    result = svc.translate("John lives in Paris", "es")
    assert "John" in result["translated_text"]
    assert "Paris" in result["translated_text"]
    assert len(result["entities"]) == 2


def test_entities_preserved_across_translation():
    svc = _make_service()
    result = svc.translate("Obama met Merkel in Berlin", "de")
    for entity in result["entities"]:
        assert entity["text"] in result["translated_text"]


def test_no_entities_passthrough():
    svc = _make_service()
    result = svc.translate("the cat sat on a mat", "fr")
    assert result["translated_text"] == "THE CAT SAT ON A MAT"
    assert len(result["entities"]) == 0


def test_empty_input():
    svc = _make_service()
    result = svc.translate("", "es")
    assert result["translated_text"] == ""
    assert len(result["entities"]) == 0


def test_response_structure():
    svc = _make_service()
    result = svc.translate("Visit London", "fr")
    assert "source_text" in result
    assert "translated_text" in result
    assert "entities" in result
    assert "target_lang" in result
    assert result["target_lang"] == "fr"
    assert result["source_text"] == "Visit London"
