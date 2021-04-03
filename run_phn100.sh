#!/bin/bash
#SBATCH --partition=ILCC_GPU                 # ILCC_GPU, CDT_GPU, ILCC_CPU, etc

#SBATCH --job-name=p_phone                  # Job name
#SBATCH --mail-type=END,FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=j.williams@ed.ac.uk      # Where to send mail
#SBATCH --ntasks=1                           # Run on a single machine
#SBATCH --gres=gpu:1                         # request N gpus
#SBATCH --cpus-per-task=1                    # require N cpus
#SBATCH --mem=14000                          # Job memory request
#SBATCH --time=02:00:00                      # Time limit hrs:min:sec
#SBATCH --output=/home/s1738075/taco_modified/logs/phn100_1024_phone.out
#SBATCH --error=/home/s1738075/taco_modified/logs/phn100_1024_phone.err

#echo "hello"

CUDA_VISIBLE_DEVICES=0
hostname
/usr/bin/nvidia-smi
. "/home/s1738075/miniconda3/etc/profile.d/conda.sh"
conda activate taco

DATASET=vqcodes
NAME=all_vctk_phn100_SP_phones_1024
N=1025
ORIG_DATA=/home/s1738075/special/L1_dat_files/sys5_phn100/phn100_648024/${NAME}

SCRATCH=/disk/scratch/s1738075
SCRATCH_DATA=/disk/scratch/s1738075/data
SCRATCH_CHECK=/disk/scratch/s1738075/checkpoints
SCRATCH_DATA_RAW=/disk/scratch/s1738075/data/${NAME}
SOURCE_DATA=/disk/scratch/s1738075/data/${NAME}_source
TARGET_DATA=/disk/scratch/s1738075/data/${NAME}_target
CHECKPOINTS=/home/s1738075/checkpoints/${NAME}
LIST=/home/s1738075/taco_modified/self_attention_tacotron/examples/codes
HPARAM_FILE=/home/s1738075/taco_modified/self_attention_tacotron/examples/codes/self-attention-tacotron.json
OUTPUT_DIR=/home/s1738075/taco_modified/prediction/${NAME}

export PYTHONPATH=/home/s1738075/taco_modified:/home/s1738075/taco_modified/tacotron2:/home/s1738075/taco_modified/self_attention_tacotron:/home/s1738075/taco_modified/multi_speaker_tacotron:/home/s1738075/miniconda3/envs/taco/lib/python3.6/site-packages:/home/s1738075/miniconda3/envs/taco/lib/python3.6/site-packages/mkl:$PYTHONPATH
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=0

mkdir $SCRATCH
mkdir $SCRATCH_DATA
mkdir $SCRATCH_CHECK
mkdir ${CHECKPOINTS}
#mkdir $SOURCE_DATA
#mkdir $TARGET_DATA
#rsync -ruva $ORIG_DATA $SCRATCH_DATA
#python preprocess_vqcodes.py --target-only $SCRATCH_DATA_RAW $TARGET_DATA vctk 0 $N
#rsync -ruva $TARGET_DATA /home/s1738075/data/
#python preprocess_vqcodes.py --source-only $SCRATCH_DATA_RAW $SOURCE_DATA vctk 0 $N
#rsync -ruva $SOURCE_DATA /home/s1738075/data/
#rsync -ruva $SCRATCH_DATA_RAW /home/s1738075/data/
#exit

rsync -ruva /home/s1738075/data/${NAME}_source $SCRATCH_DATA
rsync -ruva /home/s1738075/data/${NAME}_target $SCRATCH_DATA
python train.py --source-data-root=$SOURCE_DATA --target-data-root=$TARGET_DATA --selected-list-dir=$LIST --checkpoint-dir=$CHECKPOINTS --hparam-json-file=$HPARAM_FILE
