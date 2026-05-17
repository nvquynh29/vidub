ASR_ENGINES: dict[str, type] = {}
TRANSLATE_ENGINES: dict[str, type] = {}
TTS_ENGINES: dict[str, type] = {}


def register_asr(name: str):
    def wrapper(cls):
        ASR_ENGINES[name] = cls
        return cls
    return wrapper


def register_translate(name: str):
    def wrapper(cls):
        TRANSLATE_ENGINES[name] = cls
        return cls
    return wrapper


def register_tts(name: str):
    def wrapper(cls):
        TTS_ENGINES[name] = cls
        return cls
    return wrapper
