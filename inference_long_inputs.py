from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import sys
import time  # Import time module to measure latency
from preprocessing import preprocess_text

# Model and tokenizer paths
model_path = 'oranne55/qualifier-model4-finetune-pretrained-transformer-for-long-inputs'
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Check if MPS is available or fallback to CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

def predict_long_text_with_preprocess(text, model, tokenizer):
    # Input validation
    if not isinstance(text, str) or text.strip().isdigit() or len(text) == 0:
        return {"error": "Invalid input: Please provide a non-empty, non-numeric text."}

    # Preprocess and tokenize the text
    start_time = time.time()

    text = preprocess_text(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"].squeeze()

    # Split into chunks of 512 tokens with 100 tokens overlap
    chunk_size = 512
    overlap = 100
    
    # Create overlapping chunks
    chunks = []
    for i in range(0, len(input_ids), chunk_size - overlap):
        chunk = input_ids[i:i + chunk_size]
        if len(chunk) == chunk_size:  # Ensure the chunk is of correct size
            chunks.append(chunk)

    # Flag to track if any chunk is classified as "jailbreak"
    contains_jailbreak = False

    # Start time measurement for latency calculation
    for chunk in chunks:
        chunk = chunk.unsqueeze(0).to(device)  # Move chunk to the correct device (MPS or CPU)

        # Predict on the chunk
        with torch.no_grad():
            outputs = model(chunk)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()

            # Check if this chunk is classified as "jailbreak"
            if model.config.id2label[prediction] == "jailbreak":
                contains_jailbreak = True
                break  # Stop further checks if "jailbreak" is detected

    # End time measurement after prediction
    end_time = time.time()

    # Calculate latency
    latency = end_time - start_time

    # Final decision based on whether any chunk was classified as "jailbreak"
    final_prediction = "jailbreak" if contains_jailbreak else "benign"
    
    # Return the classification result and latency
    return {"classification": final_prediction, "latency": latency}

def main():
    # Check if prompt is provided as a command-line argument
    if len(sys.argv) > 1:
        prompt = sys.argv[1]  # Directly use the first argument as prompt
    else:
        print("Usage: python inference_long_inputs.py \"<text_prompt>\"")
        return

    # Run inference and print the result
    result = predict_long_text_with_preprocess(prompt, model, tokenizer)
    print("Result:", result)

if __name__ == "__main__":
    main()
