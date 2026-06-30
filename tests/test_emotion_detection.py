"""Unit tests for emotion detection."""

import json
import unittest
from unittest.mock import Mock, patch

import requests

from EmotionDetection import emotion_detector
from EmotionDetection.emotion_detection import EMPTY_RESULT


def watson_response(emotions):
    """Build a mock Watson response payload."""
    response = Mock()
    response.status_code = 200
    response.text = json.dumps({"emotionPredictions": [{"emotion": emotions}]})
    response.raise_for_status = Mock()
    return response


class TestEmotionDetector(unittest.TestCase):
    """Validate the formatted output from the detector wrapper."""

    def assert_dominant_emotion(self, text_to_analyze, emotions, expected_emotion):
        """Assert that the strongest emotion is returned as dominant."""
        with patch(
            "EmotionDetection.emotion_detection.requests.post",
            return_value=watson_response(emotions),
        ):
            result = emotion_detector(text_to_analyze)

        self.assertEqual(result["dominant_emotion"], expected_emotion)

    def test_joy(self):
        """Joy is dominant for happy feedback."""
        self.assert_dominant_emotion(
            "I am glad this happened.",
            {"anger": 0.02, "disgust": 0.01, "fear": 0.03, "joy": 0.91, "sadness": 0.03},
            "joy",
        )

    def test_anger(self):
        """Anger is dominant for angry feedback."""
        self.assert_dominant_emotion(
            "I am really mad about this.",
            {"anger": 0.82, "disgust": 0.04, "fear": 0.06, "joy": 0.01, "sadness": 0.07},
            "anger",
        )

    def test_disgust(self):
        """Disgust is dominant for disgusted feedback."""
        self.assert_dominant_emotion(
            "I feel disgusted just hearing about this.",
            {"anger": 0.08, "disgust": 0.79, "fear": 0.04, "joy": 0.01, "sadness": 0.08},
            "disgust",
        )

    def test_sadness(self):
        """Sadness is dominant for sad feedback."""
        self.assert_dominant_emotion(
            "I am so sad about this.",
            {"anger": 0.03, "disgust": 0.02, "fear": 0.04, "joy": 0.01, "sadness": 0.9},
            "sadness",
        )

    def test_fear(self):
        """Fear is dominant for fearful feedback."""
        self.assert_dominant_emotion(
            "I am really afraid that this will happen.",
            {"anger": 0.04, "disgust": 0.02, "fear": 0.87, "joy": 0.01, "sadness": 0.06},
            "fear",
        )

    def test_empty_text(self):
        """Empty text returns the invalid-input result."""
        result = emotion_detector("")

        self.assertEqual(result, EMPTY_RESULT)

    def test_service_error(self):
        """External service errors return the invalid-input result."""
        with patch(
            "EmotionDetection.emotion_detection.requests.post",
            side_effect=requests.RequestException("service unavailable"),
        ):
            result = emotion_detector("I am glad this happened.")

        self.assertEqual(result, EMPTY_RESULT)


if __name__ == "__main__":
    unittest.main()
