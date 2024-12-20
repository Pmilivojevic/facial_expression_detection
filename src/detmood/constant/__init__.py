from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")
SCHEMA_FILE_PATH = Path("schema.yaml")

MOOD_DICT = {
    'sad': 0,
    'happy': 1,
    'surprised': 2,
    'anger': 3,
    'fear': 4,
    'disgust': 5,
    'neutral': 6
}

MOOD_DICT_BENCHMARK = {
    1: 2,
    2: 4,
    3: 5,
    4: 1,
    5: 0,
    6: 3,
    7: 6
}