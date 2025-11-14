"""
Example runner for the Smart Grading Engine.

Prepare:
- Put a sample scanned script image at 'samples/student1.jpg' (or change path).
- Provide model_answers and rubric as below (you can generate these from teacher content).
"""

from src.engine import evaluate_script
import json
import os

# Example: model answers keyed by question_id
model_answers = {
    1: "Binary search works by repeatedly dividing the search interval in half. Start with the middle element and compare with target. If target is smaller, search left half; otherwise right half. Time complexity O(log n).",
    2: "A linked list is a linear data structure where elements are not stored at contiguous memory locations and each node points to the next node using a pointer.",
}

# Example rubric structure
rubric = {
    "questions": [
        {
            "question_id": 1,
            "max_marks": 5,
            "expected_keywords": ["binary search", "middle", "time complexity", "O(log n)"],
            "penalties": {
                "length_penalty": {"min_words": 10, "deduct_per_missing_word": 0.2}
            },
            "bonus": {
                "example": 0.5
            }
        },
        {
            "question_id": 2,
            "max_marks": 4,
            "expected_keywords": ["linked list", "pointer", "contiguous"],
            "penalties": {},
            "bonus": {}
        }
    ]
}

if __name__ == "__main__":
    # Path to student scanned image (change to your file)
    image_path = os.path.join("samples", "student1.jpg")
    # For this demo, if sample doesn't exist, we'll create a synthetic text image using PIL
    if not os.path.exists(image_path):
        from PIL import Image, ImageDraw, ImageFont
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        txt = ("Q1. Binary search works by repeatedly dividing the search interval in half. "
               "Start with the middle element and compare with target. Time complexity O(log n).\n\n"
               "Q2. A linked list is a linear data structure where elements are not stored at contiguous memory locations "
               "and nodes point to next node using pointers.")
        # Create simple image
        img = Image.new('RGB', (1200, 400), color='white')
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
        d.text((10,10), txt, fill=(0,0,0), font=font)
        img.save(image_path)
        print(f"Created synthetic sample at {image_path}")

    result = evaluate_script(image_path, model_answers, rubric)
    print(json.dumps(result, indent=2))
