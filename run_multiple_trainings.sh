#!/bin/bash

# working_folder=/local/data1/iulta54/P3-THR_DL_refined
# dataset_folder=/local/data1/iulta54/Data/OCT/Thyroid_2019

working_folder=/flush/iulta54/Research/P3-THR_DL_refined
dataset_folder=/flush/iulta54/Research/Data/OCT/Thyroid

# make sure to have the right conda environment open when running the script

# ### PART 3 - test all the models

# ## LightOCT

python3 run_training_inLine_input_TFR.py -wd /flush/iulta54/Research/P3-THR_DL_refined -df /flush/iulta54/Research/Data/OCT/Thyroid/2D_classification_dataset_isotropic_TFR -mc LightOCT -mn LightOCT_isotropic_with_augmentation_kernel5_flatten_batch200 -b 200 -ct 1 -f 3 -l wcce -lr 0.001 -v 2

python3 run_training_inLine_input_TFR.py -wd /flush/iulta54/Research/P3-THR_DL_refined -df /flush/iulta54/Research/Data/OCT/Thyroid/2D_classification_dataset_isotropic_TFR -mc LightOCT -mn LightOCT_isotropic_with_augmentation_kernel5_flatten_batch400 -b 400 -ct 1 -f 3 -l wcce -lr 0.001 -v 2

python3 run_training_inLine_input_TFR.py -wd /flush/iulta54/Research/P3-THR_DL_refined -df /flush/iulta54/Research/Data/OCT/Thyroid/2D_classification_dataset_isotropic_TFR -mc LightOCT -mn LightOCT_isotropic_with_augmentation_kernel5_flatten_batch200_l0001 -b 200 -ct 1 -f 3 -l wcce -lr 0.0001 -v 2

python3 run_training_inLine_input_TFR.py -wd /flush/iulta54/Research/P3-THR_DL_refined -df /flush/iulta54/Research/Data/OCT/Thyroid/2D_classification_dataset_isotropic_TFR -mc LightOCT -mn LightOCT_isotropic_with_augmentation_kernel5_flatten_batch400_l0001 -b 400 -ct 1 -f 3 -l wcce -lr 0.0001 -v 2

# python3 run_training_inLine_input_TFR.py -wd $working_folder -df $dataset_folder/2D_classification_dataset_isotropic_TFR -mc LightOCT -mn LightOCT_c1 -ct 1 -f 3 -l wcce -lr 0.001 -is 200 200 -b 100 -augment True
#
# python3 run_training_inLine_input_TFR.py -wd $working_folder -df $dataset_folder/2D_classification_dataset_isotropic_TFR -mc LightOCT -mn LightOCT_c2 -ct 2 -f 3 -l wcce -lr 0.001 -is 200 200 -b 100 -augment True
#
# python3 run_training_inLine_input_TFR.py -wd $working_folder -df $dataset_folder/2D_classification_dataset_isotropic_TFR -mc LightOCT -mn LightOCT_c3 -ct 3 -f 3 -l wcce -lr 0.001 -is 200 200 -b 100 -augment True


# ## M2

# python3 run_training_inLine_input_TFR.py -wd $working_folder -df $dataset_folder/2D_classification_dataset_isotropic_TFR -mc M2 -mn M2_c1 -ct 1 -f 3 -l wcce -lr 0.001 -is 200 200 -b 100 -augment True
#
# python3 run_training_inLine_input_TFR.py -wd $working_folder -df $dataset_folder/2D_classification_dataset_isotropic_TFR -mc M2 -mn M2_c2 -ct 2 -f 3 -l wcce -lr 0.001 -is 200 200 -b 100 -augment True
#
# python3 run_training_inLine_input_TFR.py -wd $working_folder -df $dataset_folder/2D_classification_dataset_isotropic_TFR -mc M2 -mn M2_c3 -ct 3 -f 3 -l wcce -lr 0.001 -is 200 200 -b 100 -augment True

# ## M3

# python3 run_training_inLine_input_TFR.py -wd $working_folder -df $dataset_folder/2D_classification_dataset_isotropic_TFR -mc M3 -mn M3_c1 -ct 1 -f 3 -l wcce -lr 0.001 -is 200 200 -b 100 -augment True
#
# python3 run_training_inLine_input_TFR.py -wd $working_folder -df $dataset_folder/2D_classification_dataset_isotropic_TFR -mc M3 -mn M3_c2 -ct 2 -f 3 -l wcce -lr 0.001 -is 200 200 -b 100 -augment True
#
# python3 run_training_inLine_input_TFR.py -wd $working_folder -df $dataset_folder/2D_classification_dataset_isotropic_TFR -mc M3 -mn M3_c3 -ct 3 -f 3 -l wcce -lr 0.001 -is 200 200 -b 100 -augment True

# ## ResNet50

# python3 run_training_inLine_input_TFR.py -wd $working_folder -df $dataset_folder/2D_classification_dataset_isotropic_TFR -mc ResNet50 -mn ResNet50_c1 -ct 1 -f 3 -l wcce -lr 0.00001 -is 200 200 -b 50 -augment True
#
# python3 run_training_inLine_input_TFR.py -wd $working_folder -df $dataset_folder/2D_classification_dataset_isotropic_TFR -mc ResNet50 -mn ResNet50_c2 -ct 2 -f 3 -l wcce -lr 0.00001 -is 200 200 -b 50 -augment True
#
# python3 run_training_inLine_input_TFR.py -wd $working_folder -df $dataset_folder/2D_classification_dataset_isotropic_TFR -mc ResNet50 -mn ResNet50_c3 -ct 3 -f 3 -l wcce -lr 0.00001 -is 200 200 -b 50 -augment True

# ## VAE_original

# python3 run_training_inLine_input_TFR.py -wd $working_folder -df $dataset_folder/2D_classification_dataset_isotropic_TFR -mc VAE_original -mn VAE_original_c1 -ct 1 -f 3 -l wcce -lr 0.001 -is 200 200 -b 100 -augment True
#
# python3 run_training_inLine_input_TFR.py -wd $working_folder -df $dataset_folder/2D_classification_dataset_isotropic_TFR -mc VAE_original -mn VAE_original_c2 -ct 2 -f 3 -l wcce -lr 0.001 -is 200 200 -b 100 -augment True
#
# python3 run_training_inLine_input_TFR.py -wd $working_folder -df $dataset_folder/2D_classification_dataset_isotropic_TFR -mc VAE_original -mn VAE_original_c3 -ct 3 -f 3 -l wcce -lr 0.001 -is 200 200 -b 100 -augment True

# ## VAE1

# python3 run_training_inLine_input_TFR.py -wd $working_folder -df $dataset_folder/2D_classification_dataset_isotropic_TFR -mc VAE1 -mn VAE1_c1 -ct 1 -f 3 -l wcce -lr 0.001 -is 200 200 -b 100 -augment True
#
# python3 run_training_inLine_input_TFR.py -wd $working_folder -df $dataset_folder/2D_classification_dataset_isotropic_TFR -mc VAE1 -mn VAE1_c2 -ct 2 -f 3 -l wcce -lr 0.001 -is 200 200 -b 100 -augment True
#
# python3 run_training_inLine_input_TFR.py -wd $working_folder -df $dataset_folder/2D_classification_dataset_isotropic_TFR -mc VAE1 -mn VAE1_c3 -ct 3 -f 3 -l wcce -lr 0.001 -is 200 200 -b 100 -augment True





