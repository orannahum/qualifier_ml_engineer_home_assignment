import sys
from transformers import pipeline
import torch
import warnings
import time  # Import time module to measure latency
warnings.filterwarnings('ignore')

# Check if GPU is available and set the device accordingly
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, else fallback to CPU

# Load the text classification pipeline once at the beginning to avoid reloading it
pipe = pipeline("text-classification", model="oranne55/qualifier-model3-finetune-pretrained-transformer", device=device)

def classify_prompt(prompt):
    """Classify the input prompt and return the label, confidence score, and latency."""
    
    # Check for invalid input and skip unnecessary checks
    if not isinstance(prompt, str) or prompt.strip().isdigit() or len(prompt) == 0:
        return {"error": "Invalid input: Please provide a non-empty, non-numeric text prompt."}
    
    # Check if the length of the prompt is less than 20,000 characters
    if len(prompt) > 20000:
        return {"error": "Invalid input: The text prompt exceeds the 20,000 character limit."}

    try:
        # Start time measurement before inference
        start_time = time.time()
        
        # Perform inference
        result = pipe(prompt)
        
        # End time measurement after inference
        end_time = time.time()
        
        # Calculate the latency
        latency = end_time - start_time
        
        # Extract the label and score (confidence)
        label = result[0]['label']
        confidence = result[0]['score']
        
        # Return the classification result along with latency
        return {"classification": label, "confidence": confidence, "latency": latency}
    
    except Exception as e:
        return {"error": f"An error occurred during classification: {str(e)}"}

def main():
    # Check if prompt is provided as a command-line argument
    if len(sys.argv) > 1:
        prompt = sys.argv[1]  # Directly use the first argument as prompt
    else:
        print("Usage: python inference.py \"<text_prompt>\"")
        return

    # Run inference and print the result
    result = classify_prompt(prompt)
    print("Result:", result)

if __name__ == "__main__":
    main()
