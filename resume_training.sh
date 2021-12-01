#!/bin/bash

Help()
{
   # Display Help
   echo "Bash script to resume training of a specific model."
   echo
   echo "Syntax: run_training [w|d|g]"
   echo "required inputs:"
   echo "w     Working folder (where the scripts are)"
   echo "d     Dataset folder (were the data is located)"
   echo "m     Folder where the model to resume training is located."
   echo "f     Fold of the model to resume training."
   echo "g     GPU number on which to run training."
   echo
}

while getopts w:hd:g:m:f: option; do
case "${option}" in
   h) # display Help
       Help
       exit;;
   w) working_folder=${OPTARG};;
   d) dataset_folder=${OPTARG};;
   g) gpu=${OPTARG};;
   m) model=${OPTARG};;
   f) fold=${OPTARG};;

   \?) # incorrect option
         echo "Error: Invalid input"
         exit 1
esac
done


# make sure to have the right conda environment open when running the script
# activate conda environment
eval "$(conda shell.bash hook)"
conda activate P5

# work on GPU 0
export CUDA_VISIBLE_DEVICES=$gpu

# go to the working folder
cd $working_folder

# create trained_log_file folder
if ! [ -d $working_folder/resume_training_models_log ]; then
   echo "Creating folder to save log."
   mkdir $working_folder/resume_training_models_log
fi

log_folder=$working_folder/resume_training_models_log

# ########################## DECLARE VARIABLES

declare -a overwrite=False
declare -a epocs=10
declare -a patience=10
declare -a model_version=best

# run training

python3 -u resume_training.py -m $model -mv $model_version -f $fold -d $dataset_folder -e $epocs -p $patience -cf None -r $overwrite |& tee -a $log_folder/$save_model_name.log








