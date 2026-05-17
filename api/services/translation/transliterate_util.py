import re
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


# Map our target_lang codes to indic-transliteration schemes
SCRIPT_MAP = {
    "hi": sanscript.DEVANAGARI,
    "ta": sanscript.TAMIL,
    "te": sanscript.TELUGU,
    "kn": sanscript.KANNADA,
    "ml": sanscript.MALAYALAM,
    "gu": sanscript.GUJARATI,
    "bn": sanscript.BENGALI,
    "pa": sanscript.GURMUKHI,
}

# Custom transliterations for common entities (English -> Target Local)
# These override the default phonetic (ITRANS) transliteration.
CUSTOM_TRANSLITERATIONS = {
    "hi": {
        "google": "गूगल",
        "microsoft": "माइक्रोसॉफ्ट",
        "amazon": "अमेज़न",
        "apple": "एप्पल",
        "india": "इंडिया",
        "california": "कैलिफोर्निया",
        "bengaluru": "बेंगलुरु",
        "hyderabad": "हैदराबाद",
        "facebook": "फेसबुक",
        "meta": "मेटा",
    }
}


def clean_hindi_artifacts(text: str) -> str:
    """
    Strips English alphabet characters from words containing Devanagari
    characters to fix transliteration artifacts.
    """
    # Check if text contains Devanagari characters
    if not re.search(r'[\u0900-\u097F]', text):
        return text

    # Use a regex to find words (sequences of non-whitespace characters)
    # This preserves the exact whitespace of the original string
    def replace_word(match):
        word = match.group(0)
        # If word has Hindi characters, remove English letters
        if re.search(r'[\u0900-\u097F]', word):
            cleaned = re.sub(r'[A-Za-z]', '', word)
            return cleaned if cleaned else word
        return word

    return re.sub(r'\S+', replace_word, text)


def transliterate_text(text: str, target_lang: str) -> str:
    """
    Transliterates English text to the target script if supported.
    Uses CUSTOM_TRANSLITERATIONS if available to ensure natural spelling
    for common entities, otherwise falls back to phonetic (ITRANS) conversion.
    """
    # Check custom mapping first (case-insensitive)
    custom_map = CUSTOM_TRANSLITERATIONS.get(target_lang, {})
    if text.lower() in custom_map:
        return custom_map[text.lower()]

    script = SCRIPT_MAP.get(target_lang)
    if not script:
        return text

    # Fallback to phonetic ITRANS transliteration
    return transliterate(text.lower(), sanscript.ITRANS, script)
