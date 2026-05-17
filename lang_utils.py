_CODE_TO_NAME: dict[str, str] = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
    "bn": "Bengali",
    "uk": "Ukrainian",
    "he": "Hebrew",
    "no": "Norwegian",
    "sr": "Serbian",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "et": "Estonian",
    "sw": "Swahili",
    "tl": "Tagalog",
    "auto": "Auto-detect",
}

_NAME_SET = {v.lower() for v in _CODE_TO_NAME.values()}


def get_language_name(code: str | None) -> str:
    if code is None:
        return "Auto-detect"
    lower = code.strip().lower()
    if lower in _NAME_SET:
        for name in _CODE_TO_NAME.values():
            if name.lower() == lower:
                return name
    return _CODE_TO_NAME.get(lower, code)


def to_qwen_language(lang: str | None) -> str | None:
    if lang is None:
        return None
    lower = lang.strip().lower()
    if lower in _NAME_SET:
        for name in _CODE_TO_NAME.values():
            if name.lower() == lower:
                return name
    return _CODE_TO_NAME.get(lower, lang)
