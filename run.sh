

#python preprocess_vqcodes.py --source-only --hparam-json-file=/home/smg/v-j-williams/workspace/external_modified/self_attention_tacotron/examples/codes/self-attention-tacotron.json /home/smg/v-j-williams/workspace/tsubame_work/special/Data_txt_files/sys5/vctk_753011/all /home/smg/v-j-williams/workspace/external_modified/data/sys5_txt_source

#export PATH="/home/smg/v-j-williams/miniconda2/bin:$PATH"
export PYTHONPATH=/home/smg/v-j-williams/workspace/external_modified:/home/smg/v-j-williams/workspace/external_modified/tacotron2:/home/smg/v-j-williams/workspace/external_modified/self_attention_tacotron:/home/smg/v-j-williams/workspace/external_modified/multi_speaker_tacotron:/home/smg/v-j-williams/miniconda2/envs/taco/lib/python3.6/site-packages:/home/smg/v-j-williams/miniconda2/envs/taco/lib/python3.6/site-packages/mkl:$PYTHONPATH
export TF_FORCE_GPU_ALLOW_GROWTH=true




# TRAINING command to run
DATASET=vqcodes
SOURCE_DATA=/home/smg/v-j-williams/workspace/external_modified/data/sys5_txt_source
TARGET_DATA=/home/smg/v-j-williams/workspace/external_modified/data/sys5_txt_target
CHECKPOINTS=/home/smg/v-j-williams/workspace/external_modified/sys5_checkpoints
VCTK_SELECTED_LIST=/home/smg/v-j-williams/workspace/external_modified/self_attention_tacotron/examples/codes
HPARAM_FILE=/home/smg/v-j-williams/workspace/external_modified/self_attention_tacotron/examples/codes/self-attention-tacotron.json

export CUDA_VISIBLE_DEVICES=0
python train.py --source-data-root=$SOURCE_DATA --target-data-root=$TARGET_DATA --selected-list-dir=$VCTK_SELECTED_LIST --checkpoint-dir=$CHECKPOINTS --hparam-json-file=$HPARAM_FILE

#--multi-gpus


# INFERENCE command to run
#python synthesize.py  --dataset=$DATASET --data-root=$DATA --checkpoint-dir=$CHECKPOINTS --postnet-checkpoint-dir=</path/to/postnet/model/dir> --hparams=$HPARAM
