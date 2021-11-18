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


# make sure to have the right conda environment open when running the script
# activate conda environment
eval "$(conda shell.bash hook)"
conda activate P5

# work on GPU 0
export CUDA_VISIBLE_DEVICES=$gpu

# go to the working folder
cd $working_folder

# create trained_log_file folder
if ! [ -d $working_folder/trained_models_log ]; then
   echo "Creating folder to save log."
   mkdir $working_folder/trained_models_log
fi

log_folder=$working_folder/trained_models_log


#  # ############################################################################
#  # ################################ TESTING M4 ################################
#  # ############################################################################
#  # Using a 'easy' classification type (enlarged vs shrunk/depleted) and testing
#  # different learning rates (0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01)
#  # different normalization (BatchNorm and GroupNorm)
#  # different dropout_rate (0.2, 0.3 and 0.4)
#  # loss type (wcce and cce) - using alwasy wcce but with and without weights
#
# classification_type=c6
#
# model_configuration=M4
#
# ## declare an array variable
# declare -a normValues=("BatchNorm" "GroupNorm")
# declare -a dropoutValues=(0.2 0.3 0.4)
# declare -a lrValues=(0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01)
# declare -a lossValues=("wcce" "cce")
# declare -a idsValues=("weights" "none")
# declare -a batchSizeValues=(16 64 128)
#
# for normalization in "${normValues[@]}"
# do
#     for dropout_rate in "${dropoutValues[@]}"
#     do
#         for lr in "${lrValues[@]}"
#         do
#             for ids in "${idsValues[@]}"
#             do
#                 for batchSize in "${batchSizeValues[@]}"
#                 do
#                     save_model_name="$model_configuration"_"$classification_type"_"$normalization"_dr"$dropout_rate"_lr"$lr"_wcce_"$ids"_batch"$batchSize"
#                     python3 -u configure_training.py -wd $working_folder -df $dataset_folder/2D_isotropic_TFR -tts $dataset_folder/2D_isotropic_TFR/train_test_split.json -mc $model_configuration -norm $normalization -dr $dropout_rate -mn $save_model_name -b $batchSize -ct $classification_type -cct True -f 2 -l wcce -lr $lr -ks 5 5 -is 200 200 -ids $ids -v 2 -db False |& tee $log_folder/$save_model_name.log
#
#                     python3 -u run_training.py -cf $working_folder/trained_models/$save_model_name/config.json -db False |& tee -a $log_folder/$save_model_name.log
#
#                 done
#             done
#         done
#     done
# done


#  # ############################################################################
#  # ################################ TESTING M4 ################################
#  # ############################################################################
#
# classification_type=c1
#
# model_configuration=M4
#
# declare -a normValues=("BatchNorm")
# declare -a dropoutValues=(0.2 )
# declare -a lrValues=( 0.00001 )
# declare -a lossValues=("wcce")
# declare -a idsValues=("weights")
# declare -a batchSizeValues=( 64 )
#
# for normalization in "${normValues[@]}"
# do
#     for dropout_rate in "${dropoutValues[@]}"
#     do
#         for lr in "${lrValues[@]}"
#         do
#             for ids in "${idsValues[@]}"
#             do
#                 for batchSize in "${batchSizeValues[@]}"
#                 do
#                     save_model_name="$model_configuration"_"$classification_type"_"$normalization"_dr"$dropout_rate"_lr"$lr"_wcce_"$ids"_batch"$batchSize"
#                     python3 -u configure_training.py -wd $working_folder -df $dataset_folder/2D_isotropic_TFR -tts $dataset_folder/2D_isotropic_TFR/train_test_split.json -mc $model_configuration -norm $normalization -dr $dropout_rate -mn $save_model_name -b $batchSize -ct $classification_type -cct False -f 2 -l wcce -lr $lr -ks 5 5 -is 200 200 -ids $ids -v 2 -db False |& tee $log_folder/$save_model_name.log
#
#                     python3 -u run_training.py -cf $working_folder/trained_models/$save_model_name/config.json -db False |& tee -a $log_folder/$save_model_name.log
#
#                 done
#             done
#         done
#     done
# done



 ###########################################################################
 ############################### TESTING VAE_ORIGINAL ################################
 ###########################################################################

classification_type=c6

model_configuration=VAE_DEBUG

declare an array variable
declare -a normalization=BatchNorm
declare -a dropout_rate=0.2
declare -a lr=0.0001
declare -a loss=wcce
declare -a ids=weights
declare -a batchSize=16


save_model_name="$model_configuration"_"$classification_type"_"$normalization"_dr"$dropout_rate"_lr"$lr"_wcce_"$ids"_batch"$batchSize"_withoutIntermediatActivation
# python3 -u configure_training.py -wd $working_folder -df $dataset_folder/2D_isotropic_TFR -tts $dataset_folder/2D_isotropic_TFR/train_test_split.json -mc $model_configuration -norm $normalization -dr $dropout_rate -mn $save_model_name -b $batchSize -ct $classification_type -cct True -f 2 -l $loss -lr $lr -ks 5 5 -is 200 200 -ids $ids -v 2 -db True |& tee $log_folder/$save_model_name.log

python3 -u run_training.py -cf $working_folder/trained_models/$save_model_name/config.json -db False |& tee -a $log_folder/$save_model_name.log







