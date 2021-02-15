

#python preprocess_vqcodes.py --source-only --hparams=phoneme=flite,flite_binary_path='/home/smg/cooper/installs/flite/bin/flite' /home/smg/v-j-williams/workspace/tsubame_work/special/L1_dat_files/sys5_lang/siwis_552024/all_siwis_161_selected /home/smg/v-j-williams/workspace/external_modified/data/siwis_source_selected siwis 1 161

#python preprocess_vqcodes.py --target-only --hparams=phoneme=flite,flite_binary_path='/home/smg/cooper/installs/flite/bin/flite' /home/smg/v-j-williams/workspace/tsubame_work/special/L1_dat_files/sys5_lang/siwis_552024/all_siwis_161_selected /home/smg/v-j-williams/workspace/external_modified/data/siwis_target_selected1 siwis 1 161

#export PATH="/home/smg/v-j-williams/miniconda2/bin:$PATH"
export PYTHONPATH=/home/smg/v-j-williams/workspace/external_modified:/home/smg/v-j-williams/workspace/external_modified/tacotron2:/home/smg/v-j-williams/workspace/external_modified/self_attention_tacotron:/home/smg/v-j-williams/workspace/external_modified/multi_speaker_tacotron:/home/smg/v-j-williams/miniconda2/envs/taco/lib/python3.6/site-packages:/home/smg/v-j-williams/miniconda2/envs/taco/lib/python3.6/site-packages/mkl:$PYTHONPATH
export TF_FORCE_GPU_ALLOW_GROWTH=true

# TRAINING command to run
DATASET=vqcodes
SOURCE_DATA=/home/smg/v-j-williams/workspace/external_modified/data/siwis_source_selected
TARGET_DATA=/home/smg/v-j-williams/workspace/external_modified/data/siwis_target_selected1
CHECKPOINTS=/home/smg/v-j-williams/workspace/external_modified/checkpoints/siwis_selected1
VCTK_SELECTED_LIST=/home/smg/v-j-williams/workspace/external_modified/self_attention_tacotron/examples/codes_siwis
HPARAM_FILE=/home/smg/v-j-williams/workspace/external_modified/self_attention_tacotron/examples/codes_siwis/self-attention-tacotron.json

export CUDA_VISIBLE_DEVICES=0
python train.py --source-data-root=$SOURCE_DATA --target-data-root=$TARGET_DATA --selected-list-dir=$VCTK_SELECTED_LIST --checkpoint-dir=$CHECKPOINTS --hparam-json-file=$HPARAM_FILE

OUTPUT_DIR=/home/smg/v-j-williams/workspace/external_modified/prediction/siwis_selected1
#python predict_code.py  --source-data-root=$SOURCE_DATA --target-data-root=$TARGET_DATA --selected-list-dir=$VCTK_SELECTED_LIST --checkpoint-dir=$CHECKPOINTS --hparam-json-file=$HPARAM_FILE --output-dir=$OUTPUT_DIR

#python postprocess_vqcodes.py siwis_selected1
