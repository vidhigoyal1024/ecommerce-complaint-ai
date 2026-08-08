"""
End-to-end pipeline:

    Customer Complaint (text + optional photo)
          |
     Preprocessing
          |
     CNN / RNN / LSTM  ---------------->  Image Authenticity CNN (ELA-based)
          |                                        |
      Model Fusion                          Real / Fake + confidence
          |                                        |
  Category + Severity + Confidence  <---------------
          |
       LLM Agent  (retrieve policy -> reason, factoring in image verdict)
          |
   Final Recommendation (Refund / Replacement / Escalation)

Usage:
    python -m src.pipeline "My package is three weeks late and support won't respond."
"""
import sys
import os
import argparse
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from src.predict import ComplaintClassifier
from src.llm_agent import analyze_with_llm

_classifier = None
_image_checker = None
_image_checker_error = None


def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = ComplaintClassifier()
    return _classifier


def get_image_checker():
    """Lazily loads the image authenticity CNN. Returns None if it hasn't been trained yet."""
    global _image_checker, _image_checker_error
    if _image_checker is None and _image_checker_error is None:
        try:
            from src.image_predict import ImageAuthenticityChecker
            _image_checker = ImageAuthenticityChecker()
        except FileNotFoundError as e:
            _image_checker_error = str(e)
    return _image_checker


def run_pipeline(complaint_text: str, image: Image.Image = None) -> dict:
    clf = get_classifier()
    prediction = clf.predict(complaint_text)

    image_authenticity = None
    image_check_note = None
    if image is not None:
        checker = get_image_checker()
        if checker is not None:
            image_authenticity = checker.predict(image)
        else:
            image_check_note = _image_checker_error

    llm_result = analyze_with_llm(
        complaint_text=complaint_text,
        category=prediction["category"],
        severity=prediction["severity"],
        confidence=prediction["category_confidence"],
        image_authenticity=image_authenticity,
    )

    result = {
        "complaint": complaint_text,
        "classification": {
            "category": prediction["category"],
            "severity": prediction["severity"],
            "category_confidence": prediction["category_confidence"],
            "severity_confidence": prediction["severity_confidence"],
        },
        "image_authenticity": image_authenticity,
        "recommendation": {
            "action": llm_result["action"],
            "reasoning": llm_result["reasoning"],
            "policy_used": llm_result["policy_used"],
            "source": llm_result["source"],
        },
    }
    if image_check_note:
        result["image_check_note"] = image_check_note
    return result


def main():
    parser = argparse.ArgumentParser(description="Run the complaint review pipeline on a single complaint.")
    parser.add_argument("complaint", type=str, help="The customer complaint text.")
    args = parser.parse_args()

    result = run_pipeline(args.complaint)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
