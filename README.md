# Emotion Detector

Flask web application that sends customer feedback text to Watson NLP and returns
scores for anger, disgust, fear, joy, sadness, plus the dominant emotion.

## Run locally

```bash
python3 -m pip install -r requirements.txt
python3 server.py
```

Open `http://127.0.0.1:5000`.

## Test and lint

```bash
python3 -m unittest discover -s tests
python3 -m pylint server.py EmotionDetection tests
```
