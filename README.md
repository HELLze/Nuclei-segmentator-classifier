# Histo_AI
Repository for master thesis project.
Required Python packages specified in requirements.txt, script was written using Python 2.7 version.

## Run lymphocyte classifier
`prediction pipeline` folder contains pre-trained autoencoder for nuclei segmentation and consecutive cell classifier to detect lymphocytes.

The script will classify WSI .svs images stored in `slidepath` directory, which has to be specified in `prediction pipeline/Lymphocyte_prediction_workflow.py` line 205.
Predictions will be stored in `prediction pipeline/mormin_predictions` folder, which will be created at the beginning of prediction workflow.

To execute the code, run the following line in `prediction pipeline` directory:
```python
python Lymphocyte_prediction_workflow.py
```

## Train models
Segmentation autoencoder training image augmentation and model training scripts are in `autoencoder_training` directory;
classifier model was trained using script in `classifier_training` folder.

Training data is not supplied in this project.
