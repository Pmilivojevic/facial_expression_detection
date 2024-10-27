# facial_expression_detection

This is a project for detecting people's emotions based on their facial expression.

This project uses the EfficientNet model of the PyTorch library for this purpose. EfficientNet model uses dataset https://drive.google.com/file/d/184K8sLVcL_fq76pA2o8Bg4BthNLcwUel/view for model training so it can distinguish 7 different emotion (happiness, sadness, fear, surprise, anger, disgust and the neutral expression).

In order to use this project, you need to follow the next steps!

# How to run?
### STEPS:

Clone the repository

```bash
Project repo: https://github.com/Pmilivojevic/facial_expression_detection.git
```

### STEP 01- Create a virtual environment after opening the repository

```bash
virtualenv env
```

### STEP 02- install the requirements
```bash
pip Install -r requirements.txt
```

### STEP 03- Run file "main.py"
```bash
python main.py
```

Running process will create folder structure:

    - artifact

        - data_ingestion

        - data_validation

        - data_transformation

        - model_trainer

        - model_validation

Project execution consists of five stages, five pipelines for downloading the data, validating the data, transforming the data, training the model, and validating the model. Results of every pipeline execution are put in a corresponding folder. The model validation part is missing. Because of the  limited time, I didn't get to implement that phase.
