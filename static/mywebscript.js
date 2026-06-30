function runEmotionAnalysis() {
    const textToAnalyze = document.getElementById("textToAnalyze").value;
    const request = new XMLHttpRequest();

    request.onreadystatechange = function updateResponse() {
        if (this.readyState === 4 && this.status === 200) {
            document.getElementById("system_response").innerHTML = request.responseText;
        }
    };

    request.open(
        "GET",
        `emotionDetector?textToAnalyze=${encodeURIComponent(textToAnalyze)}`,
        true
    );
    request.send();
}
