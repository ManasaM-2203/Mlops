import json
import re
from pipeline.config import ROOT


class LexiconService:
    def __init__(self):
        self.entities = {}
        self.load_gazetteer()

    def load_gazetteer(self):
        path = ROOT / "data" / "gazetteer.json"
        if path.exists():
            with open(path, "r") as f:
                self.entities = json.load(f)
        else:
            self.entities = {"org": [], "geo": [], "per": []}

    def find_matches(self, text: str) -> list[dict]:
        matches = []
        for label, names in self.entities.items():
            for name in names:
                # Use regex to find partial or full name matches, case-insensitive
                # Ensures we match "India" even in "In India"
                pattern = r"\b" + re.escape(name) + r"\b"
                for m in re.finditer(pattern, text, re.IGNORECASE):
                    matches.append({
                        "text": m.group(),
                        "label": label,
                        "start": m.start(),
                        "end": m.end(),
                        "confidence": 1.0  # Dictionary matches are high confidence
                    })
        return matches
