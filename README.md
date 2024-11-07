![Alt text](image_bengin_or_jailbreak.png)


# 0.Create Environments for each part

### 0.1. Create "qualifier-env" and install requirements.txt
#### for all notebooks except 3-custom_neural_network.ipynb 
```bash
conda create --name qualifier-env python=3.10

conda activate qualifier-env

pip install -r requirements.txt
```

### 0.2. Create "tf-qualifier-env" and install requirements-tf.txt(for tensorflow)


#### just for 3-custom_neural_network.ipynb


```bash
conda create --name tf-qualifier-env python=3.10

conda activate tf-qualifier-env

pip install -r requirements-tf.txt
```

### 0.3. Create "inference-qualifier-env" and install requirements-tf.txt(for inference)
#### just for inference.py 
```bash
conda create --name inference-qualifier-env python=3.10

conda activate inference-qualifier-env

pip install -r requirements-inference.txt
```

# 1. Notebooks
### 1.1. 1-EDA.ipynb -> env:"qualifier-env"
#### Process of exploring and understanding the data using statistical and visualization techniques, such as analyzing word distributions, identifying important features, and examining the structure of the data

### 1.2. 2-traditional_ML_approaches.ipynb (Task 1 + Task 2) -> env:"qualifier-env"
#### Creating and training of various ML models, such as Logistic Regression, Random Forest, Gradient Boosting, and more. Comparison between training on metadata features alone versus training with metadata features + TFIDF. Training models using boosting techniques such as XGBoost and CatBoost.Performing hyperparameter tuning to select the best model, and generating metrics and a confusion matrix.Finally, saving, uploading, and evaluating the best model and preprocess pipeline.

### 1.3. 3-custom_Neural_Network.ipynb (Task 1 + Task 2) 
#### Creating and training Custom Neural Network model. first shuffles and splits the dataset into training (80%) and validation (20%) sets. Then tokenizes and pads the text data, while encoding labels for classification. A neural network model is defined and trained using the training set, with tuning hyperparameters(layer parmas) like embedding size, dense units, and dropout rate tested for optimal performance. The best model is selected based on validation accuracy. After training, the model is evaluated on a test set. A confusion matrix is plotted to visualize the model's performance. The model is used for predictions, providing a label and confidence score for each input text. Finally, Save the pipeline(that include the preprocess and model) and the model, if we want retrain so we uploaded just the model with previous pipeline.

### 1.4. 4-fine_tune_pretrained_transformer.ipynb (Task 1 + Task 2 + Task 3) -> env:"qualifier-env"
#### This code demonstrates the process of fine tune training a pretrained transformer model. It loads and preprocesses a dataset, renames columns, and applies a label mapping to convert categorical labels into numerical ones. The text is tokenized using the BERT tokenizer, and the dataset is split into training and validation sets. A pretrained BERT model is initialized and trained with the specified hyperparameters on our data(fine tune). After training, the model is evaluated using metrics like F1 score, accuracy, recall, and precision. Predictions are made on the test set, and a confusion matrix is visualized. The model is further fine-tuned by combining the training and validation datasets for additional training. and check by metrics and tests. The model saved localy and load as pipeline. finally the model upload to my Hungging Face Hub -> https://huggingface.co/oranne55/qualifier-model3-finetune-pretrained-transformer

### 1.5. 5-test_my_hf_model.ipynb -> env:"qualifier-env"
#### Loading and check my model from Hungging Face.

# 2. inference
### 2.1. infernce.py -> env:"inference-qualifier-env" (Task 3)
#### This Python script performs text classification using a pre-trained Hugging Face model.

#### * It checks if a GPU is available and selects it for processing if possible.
#### * It validates the input to ensure it's a non-empty string and doesn't exceed 20,000 characters.
#### * It measures the time (latency) taken to classify the input text.
#### * The script returns the classification label, confidence score, and latency.
#### * It accepts input through command-line arguments and prints the result.

#### To run it :
```bash
python inference.py "Your text to classify here"
```
#### example of result: 
{'classification': 'benign', 'confidence': 0.9997623562812805, 'latency': 0.46270108222961426}

### 2.2. Deploy FastAPI Service Using Docker Image (Task 3-faster)
#### This service is a text classification API built with FastAPI. It uses a pre-trained transformer model to classify input text prompts, returning the predicted label, confidence score, and the latency of the inference process. The API is accessible via HTTP requests, and the service is exposed on port 8000 inside the Docker container, which is mapped to port 8000 on the host machine.

#### The image store in: https://hub.docker.com/repository/docker/oranne5/text-classification-qualifier-api/general

#### To pull the image and run it:
```bash
docker run -p 8000:8000 oranne5/text-classification-qualifier-api:v1
```

#### Open another terminal and post prompt and get response:
```bash
curl -X 'POST' \
  'http://localhost:8000/classify' \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Your text to classify here"
  }'
```

#### example of result(same example from 2.2.)
{"classification":"benign","confidence":0.9997623562812805,"latency":0.09252357482910156}


# 3. Folders
### 3.1. datasets/
#### Contain test.csv and train.csv for simple loading


### 3.2. examples/
#### Conatin notebooks from web that hendle with jackhhao/jailbreak-classification dataset

#### 3.2.1. example1.ipynb - notebook that fine tune on jackhhao/jailbreak-classification dataset from: https://github.com/jackhhao/llm-warden/blob/main/src/train.py

#### 3.2.2. jailbreak-classifier_model_test.ipynb - notebook that check the model of jackhhao/jailbreak-classifier from: https://huggingface.co/jackhhao/jailbreak-classifier

### 3.3. fast_api_service/
#### conatin all the files for build the image: text-classification-qualifier-api(Dockerfile, main.py, requirements.txt)



