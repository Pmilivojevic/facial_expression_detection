# Facial Expression Detection

A project that uses deep learning to classify emotions based on facial expressions, leveraging the **EfficientNet** model from PyTorch. It identifies seven emotions: **happiness**, **sadness**, **fear**, **surprise**, **anger**, **disgust**, and **neutral expression**.

---

## Features

- Emotion classification into seven categories.
- Data pipelines for ingestion, validation, transformation, and training.
- Visualization of training progress and results.
- Easy-to-modify hyperparameter configuration in `params.yaml`.

---

## Requirements

- Python 3.x
- Libraries listed in `requirements.txt`

---

## Setup

### Clone the Repository
```bash
git clone https://github.com/Pmilivojevic/facial_expression_detection.git
cd facial_expression_detection
```

### Create a Virtual Environment
```bash
virtualenv env
source env/bin/activate
```

---

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Project
1. Execute the main script:

```bash
python main.py
```

2. Upon running, the project generates a structured output in the artifact folder, containing results from the pipelines.

---

## Outputs
- Statistics: Training/validation loss plots and confusion matrices in the *`evaluation_results`* folder.
- Pipelines:
    - Data Ingestion
    - Data Validation
    - Data Transformation
    - Model Training
    - Model Evaluation

---

## License
This project is licensed under the MIT License. See the **`LICENSE`** file for details.