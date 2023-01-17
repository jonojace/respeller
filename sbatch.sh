#!/bin/sh

############################################################
##### Run python script as slurm job:                  #####
##### Create temporary sbatch job script               #####
##### Fill it with slurm options                       #####
##### Then fill it with the script to run and its args #####
##### Then submit it to slurm cluster                  #####
############################################################

# =====================
# How to call this script
# =====================
# call this script as:
#    ../sbatch.sh 2 python train_apc.py --experiment_name test --batch_size 32
#    ../sbatch.sh 2 python train_tacotron.py --hp_file hparams.py
#    ../sbatch.sh 2 python gen_tacotron.py --hp_file hparams.py
# Where 2 is the number of gpus to request

# note you can specify a gpu, this will ensure that the job is only submitted to nodes with that GPU
#    ../sbatch.sh 4 2080 python train_tacotron.py --hp_file hparams.py
#    will request 4 2080s

# =====================
# Exclude particular nodes
# =====================

# WARNING WARNING WARNING using an exclude list may fail! debug this with INF support
exclude_list=""
#exclude_list=arnold
#exclude_list=duflo
exclude_list=arnold,duflo

# Only include particular nodes
include_list=""
#include_list=arnold

# =====================
# check if specific gpu was supplied
# =====================

#jobname=$1

#gpu_num=$2
gpu_num=1

#gpu_type="gtx2080ti:"
gpu_type=""

#if [ $3 = "2080" ]; then
#    gpu_type="gtx2080ti:"
#    cmd_to_run_on_cluster=${@:4} #skip jobname, gpu_num, gpu_type
#else
#    gpu_type=""
#    cmd_to_run_on_cluster=${@:3} #skip jobname, gpu_num
##    cmd_to_run_on_cluster=${@} #use all supplied arguments i.e. "python train_tacotron.py --hp_file hparams.py"
#fi

cmd_to_run_on_cluster=${@} #use all supplied arguments i.e. "python train_tacotron.py --hp_file hparams.py"


# =====================
# Create directory for logging files if it doesn't exist
# =====================
LOG_DIR="./logs_slurm"
if [ ! -d $LOG_DIR ]; then
  mkdir -p $LOG_DIR;
fi

# =====================
# rsyncing of data to scratch disk
# =====================
# instead of needing to do this, use:
# onallnodes /home/s1785140/fairseq/examples/speech_audio_corrector/copy_data_to_scratch.sh
# to copy data to all nodes
# then use a feature_manifest train.tsv that points to the scratch disk

## determine which node we are on
#NODENAME=$(echo $HOSTNAME | cut -d. -f1)
#echo we are on node: $NODENAME
#if [ $NODENAME = "arnold" ]; then
#    SCRATCH_DISK_NAME=scratch_fast
#else
#    SCRATCH_DISK_NAME=scratch
#fi
#
#
## edit path to audio in manifests
## remove previous ones
#rm train.tsv
#rm dev.tsv
#rm test.tsv
## add new scratch disk to
#sed "s/\/home\/s1785140\/data\/LJSpeech-1.1\/feature_manifest\//\/disk\/${SCRATCH_DISK_NAME}\/s1785140\//g" train_original.tsv > train.tsv
#sed "s/\/home\/s1785140\/data\/LJSpeech-1.1\/feature_manifest\//\/disk\/${SCRATCH_DISK_NAME}\/s1785140\//g" dev_original.tsv > dev.tsv
#sed "s/\/home\/s1785140\/data\/LJSpeech-1.1\/feature_manifest\//\/disk\/${SCRATCH_DISK_NAME}\/s1785140\//g" test_original.tsv > test.tsv
## move audio data
#mkdir -p /disk/${SCRATCH_DISK_NAME}/s1785140
#rsync -avu /home/s1785140/data/LJSpeech-1.1/feature_manifest/logmelspec80.zip /disk/${SCRATCH_DISK_NAME}/s1785140
#ls /disk/${SCRATCH_DISK_NAME}/s1785140

# =====================

# setup sbatch params
# =====================
nodes=1
gpus=${gpu_type}${gpu_num} #note if gpu_type is empty string then it is just gpu_num, which is fine
cpus_per_gpu=1
cpus=$(( cpus_per_gpu*gpu_num ))
#cpus=1 #might need to use only 1 cpu if node does not have many cores
tasks=1
part=ILCC_GPU,CDT_GPU
# part=ILCC_GPU,CDT_GPU,M_AND_I_GPU
time=7-00:00:00
mem=16G
#mem=16G
mail_user=s1785140@sms.ed.ac.uk
mail_type=BEGIN,END,FAIL,INVALID_DEPEND,REQUEUE,STAGE_OUT # same as ALL
#mail_type=END,FAIL,INVALID_DEPEND,REQUEUE,STAGE_OUT

# =====================
# create the sbatch file
# =====================
#shebang
echo '#!/bin/bash' > temp_slurm_job.sh
echo ""
#job params

#echo "#SBATCH --job-name=${jobname}" >> temp_slurm_job.sh
echo "#SBATCH --nodes=${nodes}" >> temp_slurm_job.sh
echo "#SBATCH --gres=gpu:${gpus}" >> temp_slurm_job.sh
echo "#SBATCH --cpus-per-task=${cpus}" >> temp_slurm_job.sh
#echo "#SBATCH --mem=${mem}" >> temp_slurm_job.sh  # warning this requests unreserved mem, if not found you will
# not be able to get a node. if you don't use this flag, you will get a default setting for mem
# and your jobs will get killed if they exceed it.
#echo "#SBATCH --ntasks=${tasks}" >> temp_slurm_job.sh
echo "#SBATCH --time=${time}" >> temp_slurm_job.sh
echo "#SBATCH --partition=${part}" >> temp_slurm_job.sh
echo "#SBATCH --mail-user=${mail_user}" >> temp_slurm_job.sh
echo "#SBATCH --mail-type=${mail_type}" >> temp_slurm_job.sh
echo "#SBATCH --output=${LOG_DIR}/%j" >> temp_slurm_job.sh #Note! remember to make this directory if it doesn't exist

echo ""


#####################################################################
# Include or exclude nodes

# exclude list
if [ -z "$exclude_list" ]
then
      echo "exclude_list is empty"
else
      echo "exclude_list is NOT empty, excluding nodes:${exclude_list}"
      echo "#SBATCH --exclude=${exclude_list}" >> temp_slurm_job.sh
fi

# include list
if [ -z "$include_list" ]
then
      echo "include_list is empty"
else
      echo "include_list is NOT empty, including nodes:${include_list}"
      echo "#SBATCH --nodelist=${include_list}" >> temp_slurm_job.sh
fi

echo "" >> temp_slurm_job.sh

echo "#### Initialise wandb ####" >> temp_slurm_job.sh
echo "export WANDB_API_KEY=afb11ca4f70d9fc0dfab01f24f1dece4c707cd36" >> temp_slurm_job.sh
echo "wandb login" >> temp_slurm_job.sh

echo "" >> temp_slurm_job.sh

echo "#### create the command to be run on the cluster ####" >> temp_slurm_job.sh
echo "cmd_to_run_on_cluster=\"${cmd_to_run_on_cluster}\"" >> temp_slurm_job.sh

echo "" >> temp_slurm_job.sh

#echo "#### handle copying of data and feature manifest manipulation based on what scratch disk is available ####" >> temp_slurm_job.sh
    # if scratch_fast doesnt have write permissions or is not found, changing feature_manifest to feature_manifest_standardscratch;
    # dynamically change back to scratch if scratch_fast doesn't exist or doesn't have write permissions
    # i.e. in the training command, replace feature_manifest with feature_manifest_standardscratch
#echo 'scratch_disk=$(bash examples/speech_audio_corrector/bash_scripts/check_scratchdisks.sh)' >> temp_slurm_job.sh
#echo 'echo scratch disk is $scratch_disk' >> temp_slurm_job.sh
#echo 'cmd_to_run_on_cluster="${cmd_to_run_on_cluster/--new-logmelspec-dir None/--new-logmelspec-dir $scratch_disk}"' >> temp_slurm_job.sh
#echo "if [ ! -w "/disk/scratch_fast" ]; then" >> temp_slurm_job.sh
#    echo 'cmd_to_run_on_cluster="${cmd_to_run_on_cluster/feature_manifest/feature_manifest_standardscratch}"' >> temp_slurm_job.sh
#    echo "mkdir -p /disk/scratch/s1785140" >> temp_slurm_job.sh
#    echo "rsync -avu /home/s1785140/data/LJSpeech-1.1/feature_manifest/logmelspec80.zip /disk/scratch/s1785140" >> temp_slurm_job.sh
#    echo "else" >> temp_slurm_job.sh
#    echo "mkdir -p /disk/scratch_fast/s1785140" >> temp_slurm_job.sh
#    echo "rsync -avu /home/s1785140/data/LJSpeech-1.1/feature_manifest/logmelspec80.zip /disk/scratch_fast/s1785140" >> temp_slurm_job.sh
#echo "fi" >> temp_slurm_job.sh

#rsyncing of data to scratch disk (need to do in job script since scratch data is isolated within job script)
#echo "SCRATCH_DISK=scratch_fast" >> temp_slurm_job.sh
#SCRATCH_DISK=scratch_fast
#echo "mkdir -p /disk/scratch_fast/s1785140" >> temp_slurm_job.sh
#echo "rsync -avu /home/s1785140/data/LJSpeech-1.1/feature_manifest/logmelspec80.zip /disk/scratch_fast/s1785140" >> temp_slurm_job.sh
#
##echo "SCRATCH_DISK=scratch" >> temp_slurm_job.sh
#SCRATCH_DISK=scratch
#echo "mkdir -p /disk/${SCRATCH_DISK}/s1785140" >> temp_slurm_job.sh
#echo "rsync -avu /home/s1785140/data/LJSpeech-1.1/feature_manifest/logmelspec80.zip /disk/${SCRATCH_DISK}/s1785140" >> temp_slurm_job.sh
# echo "rsync -avu ${repo_home}/data $scratch_folder/" >> temp_slurm_job.sh #move preprocessed data from repo dir to the scratch disk
# echo 'if [ "$?" -eq "0" ]' >> temp_slurm_job.sh #'$?' holds result of last command, '0' is success
# echo 'then' >> temp_slurm_job.sh
# echo "  echo \"Rsync succeeded.\"; data_path_flag=\"--data_path ${scratch_folder}/data\"" >> temp_slurm_job.sh #load data from scratch disk
# echo 'else' >> temp_slurm_job.sh
# echo '  echo "Error while running rsync."; data_path_flag=""' >> temp_slurm_job.sh #load data over network
# echo 'fi' >> temp_slurm_job.sh

##pre experiment logging
#start_date=`date '+%d/%m/%Y %H:%M:%S'`
#echo "echo \"Job started: $start_date\"" >> temp_slurm_job.sh
#echo "start=`date +%s`" >> temp_slurm_job.sh

#actual command to be run on cluster
#echo "srun ${cmd_to_run_on_cluster} \${data_path_flag}" >> temp_slurm_job.sh
echo "" >> temp_slurm_job.sh
echo "#### the command that will be run ####" >> temp_slurm_job.sh
echo 'echo Running the following command on slurm: ${cmd_to_run_on_cluster}' >> temp_slurm_job.sh
#echo 'srun ${cmd_to_run_on_cluster}' >> NB temp_slurm_job.sh # is using srun causing jobs to fail to get onto cluster???
echo '${cmd_to_run_on_cluster}' >> temp_slurm_job.sh

##post experiment logging
#echo "echo \"Job started: $start_date\"" >> temp_slurm_job.sh
#echo "echo \"Job finished: $(date '+%d/%m/%Y %H:%M:%S')\"" >> temp_slurm_job.sh
#echo "end=`date +%s`" >> temp_slurm_job.sh
#echo "runtime=$((end-start))" >> temp_slurm_job.sh
#echo "echo \"Job took: $runtime seconds\"" >> temp_slurm_job.sh

# =====================
# submit this temporary sbatch script to the cluster
# =====================
#echo
#echo =================== temp_slurm_job.sh below ======================
#cat temp_slurm_job.sh # debug
chmod +x temp_slurm_job.sh
sbatch temp_slurm_job.sh # submit script to slurm
