"""Client wrapper for Watson NLP emotion prediction."""

import json
from typing import Any

import requests


URL = (
    "https://sn-watson-emotion.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
HEADERS = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
REQUEST_TIMEOUT = 3
EMPTY_RESULT = {
    "anger": None,
    "disgust": None,
    "fear": None,
    "joy": None,
    "sadness": None,
    "dominant_emotion": None,
}


def _empty_result() -> dict[str, float | str | None]:
    """Return a fresh invalid-input response."""
    return EMPTY_RESULT.copy()


def emotion_detector(text_to_analyze: str) -> dict[str, float | str | None]:
    """Detect emotions in customer feedback text using Watson NLP."""
    if not text_to_analyze or not text_to_analyze.strip():
        return _empty_result()

    payload = {"raw_document": {"text": text_to_analyze}}
    try:
        response = requests.post(URL, json=payload, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    except requests.RequestException:
        return _empty_result()

    if response.status_code == 400:
        return _empty_result()

    try:
        response.raise_for_status()
    except requests.RequestException:
        return _empty_result()

    formatted_response: dict[str, Any] = json.loads(response.text)
    emotions = formatted_response["emotionPredictions"][0]["emotion"]
    dominant_emotion = max(emotions, key=emotions.get)

    return {
        "anger": emotions["anger"],
        "disgust": emotions["disgust"],
        "fear": emotions["fear"],
        "joy": emotions["joy"],
        "sadness": emotions["sadness"],
        "dominant_emotion": dominant_emotion,
    }
