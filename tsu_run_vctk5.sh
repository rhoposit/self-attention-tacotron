#!/bin/sh
#$ -cwd
#$ -l q_node=1
#$ -l h_rt=01:00:00
#$ -N vctk5
#$ -o /gs/hs0/tgh-20IAA/jenn/taco_exp/log/vctk5.out
#$ -e /gs/hs0/tgh-20IAA/jenn/taco_exp/log/vctk5.err
#$ -m e
#$ -M j.williams@ed.ac.uk
#$ -v GPU_COMPUTE_MODE=0

# Command to run:
# qsub -g tgh-20IAA tsu_run_vctk5.sh

. /etc/profile.d/modules.sh
module load cuda/10.0.130
module load cudnn/7.4
hostname
/usr/bin/nvidia-smi
. "/gs/hs0/tgh-20IAA/jenn/miniconda3/etc/profile.d/conda.sh"
conda deactivate
conda activate taco

#python preprocess_vqcodes.py --source-only --hparam-json-file=/gs/hs0/tgh-20IAA/jenn/taco_exp/self_attention_tacotron/examples/codes/self-attention-tacotron.json /gs/hs0/tgh-20IAA/jenn/special/L1_dat_files/sys5/vctk_753011/all_vctk /gs/hs0/tgh-20IAA/jenn/taco_exp/data/vctk_source

#python preprocess_vqcodes.py --target-only --hparam-json-file=/gs/hs0/tgh-20IAA/jenn/taco_exp/self_attention_tacotron/examples/codes/self-attention-tacotron.json /gs/hs0/tgh-20IAA/jenn/special/L1_dat_files/sys5/vctk_753011/all_vctk /gs/hs0/tgh-20IAA/jenn/taco_exp/data/vctk_target0 0 170





#export PATH="/home/smg/v-j-williams/miniconda2/bin:$PATH"
export PYTHONPATH=/gs/hs0/tgh-20IAA/jenn/taco_exp:/gs/hs0/tgh-20IAA/jenn/taco_exp/tacotron2:/gs/hs0/tgh-20IAA/jenn/taco_exp/self_attention_tacotron:/gs/hs0/tgh-20IAA/jenn/taco_exp/multi_speaker_tacotron:/home/smg/v-j-williams/miniconda2/envs/taco/lib/python3.6/site-packages:/home/smg/v-j-williams/miniconda2/envs/taco/lib/python3.6/site-packages/mkl:$PYTHONPATH
export TF_FORCE_GPU_ALLOW_GROWTH=true

# TRAINING command to run
DATASET=vqcodes
SOURCE_DATA=/gs/hs0/tgh-20IAA/jenn/taco_exp/data/vctk_source
TARGET_DATA=/gs/hs0/tgh-20IAA/jenn/taco_exp/data/vctk_target0
CHECKPOINTS=/gs/hs0/tgh-20IAA/jenn/taco_exp/checkpoints/vctk5
VCTK_SELECTED_LIST=/gs/hs0/tgh-20IAA/jenn/taco_exp/self_attention_tacotron/examples/codes
HPARAM_FILE=/gs/hs0/tgh-20IAA/jenn/taco_exp/self_attention_tacotron/examples/codes/self-attention-tacotron.json

export CUDA_VISIBLE_DEVICES=0
python train.py --source-data-root=$SOURCE_DATA --target-data-root=$TARGET_DATA --selected-list-dir=$VCTK_SELECTED_LIST --checkpoint-dir=$CHECKPOINTS --hparam-json-file=$HPARAM_FILE

OUTPUT_DIR=/gs/hs0/tgh-20IAA/jenn/taco_exp/prediction/vctk5
python predict_code.py  --source-data-root=$SOURCE_DATA --target-data-root=$TARGET_DATA --selected-list-dir=$VCTK_SELECTED_LIST --checkpoint-dir=$CHECKPOINTS --hparam-json-file=$HPARAM_FILE --output-dir=$OUTPUT_DIR

python tsu_postprocess_vqcodes.py vctk5
