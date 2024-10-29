
# Diseased Thyroid Tissue Classification in OCT Images Using Deep Learning: Towards Surgical Decision Support

This repository contains code for the classification of thyroid tissue in optical coherence tomography (OCT) images using deep learning. This work explores the use of 2D and 3D deep learning models to automatically distinguish between normal and diseased thyroid tissue, providing potential real-time support in surgical settings.

[Journal](https://doi.org/10.1002/jbio.202200227) | [Cite](#reference)

**Abstract**
Intraoperative guidance tools for thyroid surgery based on optical coherence tomography (OCT) could aid distinguish between normal and diseased tissue. However, OCT images are difficult to interpret, thus, real-time automatic analysis could support the clinical decision-making. In this study, several deep learning models were investigated for thyroid disease classification on 2D and 3D OCT data obtained from ex vivo specimens of 22 patients undergoing surgery and diagnosed with several thyroid pathologies. Additionally, two open-access datasets were used to evaluate the custom models. On the thyroid dataset, the best performance was achieved by the 3D vision transformer model with a Matthew's correlation coefficient (MCC) of 0.79 (accuracy = 0.90) for the normal-versus-abnormal classification. On the open-access datasets, the custom models achieved the best performance (MCC > 0.88, accuracy > 0.96). Results obtained for the normal-versus-abnormal classification suggest OCT, complemented with deep learning-based analysis, as a tool for real-time automatic diseased tissue identification in thyroid surgery.

**Key highlights:**
- Real-time analysis potential for surgical support.
- Models achieve high accuracy on both thyroid and open-access OCT datasets.
- Best performance with a 3D vision transformer model on normal-versus-abnormal classification.
## Table of Contents
- [Requirements](#requirements)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Reference](#reference)
- [License](#license)

## Requirements
This project was developed in Python. Ensure you have the following libraries installed:

```bash
pip install requirements.txt
```
## Usage

### 1. Dataset Preparation
- Use create_dataset.py for 2D datasets and create_dataset_3D.py for 3D datasets. These scripts will structure the dataset for model training and validation, converting OCT data (saved in .nii format) into .tfrecords.

To create a 2D OCT dataset that can be used for training by this framework, run the create_dataset.py script. This can be done by:

Example command:
```bash
python3 create_dataset.py -dt PATH_raw_OCT_volume -ds PATH_to_destination -s PATH_to_the_csv_file -ss 1.4 2.0 -r 0.07
```
This will create both the anisotropic and isotropic 2D OCT dataset saved as .nii files as well as TRF records in the PATH_to_destination folder. The process can be lengthy depending on the number of raw OCT volumes to process

### 2. Training Configuration
- Use `configure_training.py` to set up training parameters, such as model type, batch size, learning rate, and training epochs. Use the command below to see al the settings:
```bash
python configure_training.py --help
```

(EXAMPLE) Run the following command to get the configuration file for the LightOCT model and the normal-vs-diseased classification task (assuming that the repository is saved in PATH_working_folder).
```bash
python configure_training.py -wd PATH_working_folder -df PATH_to_destination/2D_isotropic_TFR -tts PATH_to_destination/2D_isotropic_TFR/train_test_split.json -mc LightOCT -mn TEST_LightOCT -ct c1
```
This will create a config.json file saved in PATH_working_folder/trained_models/TEST_LightOCT containing the information about the model and its configuration as well as the training, validation and testing file names used during model training and evaluation.

### 3. Model Training
- Run `run_training.py` for 2D models or `run_training_3D.py` for 3D models to initiate the training process. The run_training scripts use the information available in the config.json file to create the model as well as the data-generators used to load and process the data during training. Use the command below to see al the settings:
```bash
python run_training.py --help
```

(EXAMPLE) Run the following command to train the LightOCT model for normal-vs-diseased classification.
```bash
python run_training.py -cf PATH_working_folder/trained_models/TEST_LightOCT/config.json -e 250 -p 250 
```
During model training, the logs of the training will be displayed in the terminal as well as saved in the PATH_working_folder/trained_models/TEST_LightOCT folder, where for every trained fold, the best model, the last model and training curves are saved.

### 4. Model Evaluation
- Evaluate trained models using `test_model.py` for 2D data or `test_model_3D.py` for 3D data. This will produce evaluation metrics such as accuracy, Matthew's correlation coefficient (MCC), and confusion matrices. Use the command below to see al the settings:
```bash
python test_model.py --help
```
(EXAMPLE) 
```bash
python test_model.py --model_path PATH_to_trained_model --data_path PATH_dataset_folder
```
This will generate test summary files, plot the confusion matrix of the model ensemble and well as the ROC and PP curve.

### 5. Plotting Results
Use the following scripts for analysis and visualization:
- `print_model_performance.py`: Outputs and summarizes model performance metrics.
- `visualize_dataset.py` and `visualize_activation_maps.py`: Inspect dataset samples and model activation maps for model interpretability.

## Code Structure

The repository is organized as follows:

- **Data Preparation**:
  - `create_dataset.py` and `create_dataset_3D.py`: Scripts to prepare 2D and 3D OCT datasets for training and testing.
- **Model Training and Evaluation**:
  - `run_training.py` and `run_training_3D.py`: Scripts for training 2D and 3D models on OCT data.
  - `test_model.py` and `test_model_3D.py`: Scripts to evaluate models on test data.
  - `resume_training.py`: Allows resumption of model training.
- **Utilities and Visualization**:
  - `models_tf.py` and `models_3D_tf.py`: Contain the definitions of the models.
  - `utilities.py` and `utilities_models_tf.py`: Utility functions for data handling, model evaluation, and performance metrics.
  - `visualize_dataset.py` and `visualize_activation_maps.py`: Tools for dataset inspection and model activation map visualization.
  - `print_model_performance.py`: Summarizes and outputs model performance metrics.

Additionally, an example of a bach script that uses the above Python scripts that runs training configuration, model training and testing is also provided (`LightOCT.sh`)

## Citation
If you use this work, please cite:

```bibtex
@article{tampu_2023_diseased,
author = {Tampu, Iulian Emil and Eklund, Anders and Johansson, Kenth and Gimm, Oliver and Haj-Hosseini, Neda},
title = {Diseased thyroid tissue classification in OCT images using deep learning: Towards surgical decision support},
journal = {Journal of Biophotonics},
volume = {16},
number = {2},
pages = {e202200227},
keywords = {convolutional neural networks, optical coherence tomography, surgical guidance, thyroid, tissue classification, vision transformers},
doi = {https://doi.org/10.1002/jbio.202200227},
year = {2023}
}
```

## License
This work is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/).
