"""Unit tests for the EmotionDetection package."""

import json
import unittest
from unittest.mock import Mock, patch

from EmotionDetection import emotion_detector


def watson_response(emotions):
    """Create a mock Watson response containing emotion scores."""
    response = Mock()
    response.status_code = 200
    response.text = json.dumps({"emotionPredictions": [{"emotion": emotions}]})
    response.raise_for_status = Mock()
    return response


class TestEmotionDetection(unittest.TestCase):
    """Test that the detector returns the expected dominant emotions."""

    def assert_dominant_emotion(self, text_to_analyze, emotions, expected_emotion):
        """Validate the dominant emotion for a statement."""
        with patch(
            "EmotionDetection.emotion_detection.requests.post",
            return_value=watson_response(emotions),
        ):
            result = emotion_detector(text_to_analyze)

        self.assertEqual(result["dominant_emotion"], expected_emotion)

    def test_joy(self):
        """Test joy detection."""
        self.assert_dominant_emotion(
            "I am glad this happened",
            {"anger": 0.02, "disgust": 0.01, "fear": 0.03, "joy": 0.91, "sadness": 0.03},
            "joy",
        )

    def test_anger(self):
        """Test anger detection."""
        self.assert_dominant_emotion(
            "I am really mad about this",
            {"anger": 0.82, "disgust": 0.04, "fear": 0.06, "joy": 0.01, "sadness": 0.07},
            "anger",
        )

    def test_disgust(self):
        """Test disgust detection."""
        self.assert_dominant_emotion(
            "I feel disgusted just hearing about this",
            {"anger": 0.08, "disgust": 0.79, "fear": 0.04, "joy": 0.01, "sadness": 0.08},
            "disgust",
        )

    def test_sadness(self):
        """Test sadness detection."""
        self.assert_dominant_emotion(
            "I am so sad about this",
            {"anger": 0.03, "disgust": 0.02, "fear": 0.04, "joy": 0.01, "sadness": 0.9},
            "sadness",
        )

    def test_fear(self):
        """Test fear detection."""
        self.assert_dominant_emotion(
            "I am really afraid that this will happen",
            {"anger": 0.04, "disgust": 0.02, "fear": 0.87, "joy": 0.01, "sadness": 0.06},
            "fear",
        )


if __name__ == "__main__":
    unittest.main()
