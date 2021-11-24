#!/bin/bash

Help()
{
   # Display Help
   echo "Bash script to run multiple trainings"
   echo
   echo "Syntax: run_training [w|d|g]"
   echo "required inputs:"
   echo "w     Working folder (where the scripts are)"
   echo "d     Dataset folder (were the data is located)"
   echo "g     GPU number on which to run training"
   echo
}

while getopts w:hd:g: option; do
case "${option}" in
   h) # display Help
       Help
       exit;;
   w) working_folder=${OPTARG};;
   d) dataset_folder=${OPTARG};;
   g) gpu=${OPTARG};;

   \?) # incorrect option
         echo "Error: Invalid input"
         exit 1
esac
done

# set which GPU to work on
export CUDA_VISIBLE_DEVICES=$gpu

# make sure to have the right conda environment open when running the script
# activate conda environment
eval "$(conda shell.bash hook)"
conda activate P5

# go to the working folder
cd $working_folder

# create trained_log_file folder
if ! [ -d $working_folder/trained_models_log ]; then
   echo "Creating folder to save log."
   mkdir $working_folder/trained_models_log
fi

log_folder=$working_folder/trained_models_log


 ##########################################################################
 ################################ TESTING  ################################
 ##########################################################################

declare -a model_configuration=ViT_3D

declare -a classification_type=c13
declare -a custom_classification=True

# declare an array variable
declare -a normalization=BatchNorm
declare -a dropout_rate=0.3
declare -a loss=wcce
declare -a ids=weights
declare -a batchSize=4
declare -a lr=0.00001
declare -a nFolds=1

# specific to Vit
declare -a vit_patch_size=16
declare -a vit_projection_dim=32
declare -a vit_num_heads=4
declare -a vit_transformer_layers=2



save_model_name="$model_configuration"_"$classification_type"_lr"$lr"_pts"$vit_patch_size"_prjd"$vit_projection_dim"_batch"$batchSize"
python3 -u configure_training.py -wd $working_folder -df $dataset_folder/2D_isotropic_TFR -3d True -tts $dataset_folder/2D_isotropic_TFR/train_test_split.json -mc $model_configuration -norm $normalization -dr $dropout_rate -mn $save_model_name -b $batchSize -ct $classification_type -cct $custom_classification -f $nFolds -l $loss -lr $lr -is 200 200 -vit_ps $vit_patch_size -vit_pd $vit_projection_dim -vit_nh $vit_num_heads -vit_mhu 2048 1024 -vit_tl $vit_transformer_layers -v 2 -db False |& tee $log_folder/$save_model_name.log

python3 -u run_training_3D.py -cf $working_folder/trained_models/$save_model_name/config.json -db False -e 250 -p 100 |& tee -a $log_folder/$save_model_name.log

# test models (best and last)
python3 -u test_model_3D.py -m $working_folder/trained_models/$save_model_name -d $dataset_folder -mv best |& tee -a $log_folder/$save_model_name.log
python3 -u test_model_3D.py -m $working_folder/trained_models/$save_model_name -d $dataset_folder -mv last |& tee -a $log_folder/$save_model_name.log
