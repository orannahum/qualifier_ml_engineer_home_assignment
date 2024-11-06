import sys
from transformers import pipeline

# Load the text classification pipeline with the specified model
pipe = pipeline("text-classification", model="oranne55/qualifier-model3-finetune-pretrained-transformer")

def classify_prompt(prompt):
    """Classify the input prompt and return the label and confidence score."""
    if not isinstance(prompt, str) or prompt.strip() == "":
        return {"error": "Invalid input: Please provide a non-empty text prompt."}

    try:
        # Run inference
        result = pipe(prompt)
        
        # Extract the label and score (confidence)
        label = result[0]['label']
        confidence = result[0]['score']
        
        return {"classification": label, "confidence": confidence}
    except Exception as e:
        return {"error": f"An error occurred during classification: {str(e)}"}

def main():
    # Check if prompt is provided as a command-line argument
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])  # Combine all arguments into a single prompt
    else:
        print("Usage: python inference.py \"<text_prompt>\"")
        return

    # Run inference
    result = classify_prompt(prompt)
    print("Result:", result)

if __name__ == "__main__":
    main()
