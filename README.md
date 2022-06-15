# Deep learning-based classification of OCT images: application on thyroid diseases

This repository contains the framework to configure, train and evaluate several deep learning model for the classification of thyroid diseases using OCT data.

This code was used to obtain the results reported in [reference to the publication].

## Python environment setup

Information about how to set up a python environment that runs the code of the project.

## Steps 
The following will describe the steps for training one of the available models (LightOCT) on 2D thyroid data for one of the classification tasks (binary classification). The main steps are:
- [Dataset preparation](###Dataset-preparation)
- [Training configuration](###Training-configuration)
- [Model training](###Model-training)
- [Model evaluation](###Model-evaluation)
- [Plotting results](###Plotting-results)

### Dataset preparation
Assuming that the OCT volumes saved as .nii files are saved in *PATH_raw_OCT_volume* and that the .csv file describing the classification of every volume is available (see dataset example). 

To create a 2D OCT dataset that can be used for training by this framework, run the **create_datase.py** script. This can be done by

```bash
python3 create_dataset.py -dt PATH_raw_OCT_volume -ds PATH_to_destination -s PATH_to_the_csv_file -ss 1.4 2.0 -r 0.07
```

This will create both the anisotropi and isotropic 2D OCT dataset saved as .nii files as well as TRF records in the *PATH_to_destination* folder. The process can be lengthy depending on the number of raw OCT volumes to process

### Training configuration
This is done by running the configure_taining.py script which creates a config.json file read by the run_training.py script. The training gonfiguration consists in, among other, specifying the model one wants to use and the classification task the model should be trained for. See all the available settings by running

```bash
python3 configure_training.py --help
```

For this example, run the following comand to get the configuration file for the LightOCT model and the normal-vs-diseased classification task (assuming that the repository is saved in *PATH_working_folder*)

```bash
python3 configure_training.py -wd PATH_working_folder -df PATH_to_destination/2D_isotropic_TFR -tts PATH_to_destination/2D_isotropic_TFR/train_test_split.json -mc LightOCT -mn TEST_LightOCT -ct c1
```

This will create a config.json file saved in PATH_working_folder/trained_models/TEST_LightOCT containing the infromation about the model and its configuration as well as the training, validation and testing file names used during model training and evaluation. 

### Model training
The run_training.py script uses the information available in the config.json file to create the model as well as the data-generators used to load and process the data during training. In the context of this example, run the following comand to train the LightOCT model for normal-vs-diseased classification. 
```bash
python3 run_training.py -cf PATH_working_folder/trained_models/TEST_LightOCT/config.json -e 250 -p 250 
```

During model training, the logs of the training will be displayed in the terminal as well as saved in the PATH_working_folder/trained_models/TEST_LightOCT folder, where for every trained fold, the best model, the last model and training curves are saved.

For all the available settings run
```bash
python3 run_training.py --help
```
### Model evaluation
Ones the model has been trained it can be evaluated by running the test_model.py script. This uses the config.json file created previously to load the testing images.

```bash
python3 -u test_model.py -m PATH_working_folder/trained_models/TEST_LightOCT -d PATH_to_destination/2D_isotropic_TFR
```
This will generate test summary files, plot the confusion matrix of the model ensamble and well as the ROC and PP curve.

For all the available settings run 
```bash
python3 test_model.py --help
```

### Plotting results

(TO BE COMPLETED)

### Bash scripts
For convenience, bash scripts running training confuguration, model training and evaluation are also available. See train_LightOCT.sh script for example. One can use such file as an example for training different model configurations on different classification tasks.


