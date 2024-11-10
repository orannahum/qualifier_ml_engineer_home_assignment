import sys
from transformers import pipeline
import torch
import warnings
import time  # Import time module to measure latency
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio

# Initialize FastAPI app
app = FastAPI()

# Suppress warnings for a cleaner log output
warnings.filterwarnings('ignore')

# Check if GPU is available and set the device accordingly
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, else fallback to CPU

# Load the text classification pipeline once at the beginning to avoid reloading it
pipe = pipeline("text-classification", model="oranne55/qualifier-model3-finetune-pretrained-transformer", device=device)

class PromptRequest(BaseModel):
    prompt: str

@app.post("/classify")
async def classify_prompt(request: PromptRequest):
    """Classify the input prompt and return the label, confidence score, and latency."""
    prompt = request.prompt

    # Check for invalid input and skip unnecessary checks
    if not isinstance(prompt, str) or prompt.strip().isdigit() or len(prompt) == 0:
        raise HTTPException(status_code=400, detail="Invalid input: Please provide a non-empty, non-numeric text prompt.")
    
    # Check if the length of the prompt is less than 20,000 characters
    if len(prompt) > 20000:
        raise HTTPException(status_code=400, detail="Invalid input: The text prompt exceeds the 20,000 character limit.")
    
    try:
        # Start time measurement before inference
        start_time = time.time()

        # Perform inference asynchronously (FastAPI automatically handles async)
        result = await asyncio.to_thread(pipe, prompt)

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
        raise HTTPException(status_code=500, detail=f"An error occurred during classification: {str(e)}")

@app.get("/")
async def read_root():
    """Simple root endpoint to check if the app is running."""
    return {"message": "Welcome to the text classification API!"}
