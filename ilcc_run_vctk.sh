

# rsync special files to scratch space
# generate the data on scratch space
# rsync the data to home

#rsync -ruva /home/s1738075/special/L1_dat_files/sys5/vctk_753011/all_vctk_170_selected/ /disk/scratch/s1738075/data/
rsync -ruva /home/s1738075/special/L1_dat_files/sys5/vctk_753011/vctk_170_small /disk/scratch/s1738075/data/
mkdir /disk/scratch/s1738075/data/vctk_source_small
mkdir /disk/scratch/s1738075/data/vctk_target_small
python preprocess_vqcodes.py --target-only --hparams=phoneme=flite,flite_binary_path='/home/s1738075/taco_modified/flite' /disk/scratch/s1738075/data/vctk_170_small /disk/scratch/s1738075/data/vctk_target_small vctk 0 171
python preprocess_vqcodes.py --source-only --hparams=phoneme=flite,flite_binary_path='/home/s1738075/taco_modified/flite' /disk/scratch/s1738075/vctk_170_small /disk/scratch/data/vctk_source_small vctk 0 171
rsync -ruva /disk/scratch/s1738075/data/ /home/s1738075/data/
exit


export PYTHONPATH=/home/s1738075/taco_modified:/home/s1738075/taco_modified/tacotron2:/home/s1738075/taco_modified/self_attention_tacotron:/home/s1738075/taco_modified/multi_speaker_tacotron:/home/s1738075/miniconda3/envs/taco/lib/python3.6/site-packages:/home/s1738075/miniconda3/envs/taco/lib/python3.6/site-packages/mkl:$PYTHONPATH
export TF_FORCE_GPU_ALLOW_GROWTH=true

# TRAINING command to run
DATASET=vqcodes
SOURCE_DATA=/disk/scratch/s1738075/data/vctk_source_small
TARGET_DATA=/disk/scratch/s1738075/data/vctk_target_small
CHECKPOINTS=/disk/scratch/s1738075/checkpoints/vctk_small
VCTK_SELECTED_LIST=/home/s1738075/taco_modified/self_attention_tacotron/examples/codes
HPARAM_FILE=/home/s1738075/taco_modified/self_attention_tacotron/examples/codes/self-attention-tacotron.json

export CUDA_VISIBLE_DEVICES=0
python train.py --source-data-root=$SOURCE_DATA --target-data-root=$TARGET_DATA --selected-list-dir=$VCTK_SELECTED_LIST --checkpoint-dir=$CHECKPOINTS --hparam-json-file=$HPARAM_FILE

OUTPUT_DIR=/home/s1738075/taco_modified/prediction/vctk_selected0
#python predict_code.py  --source-data-root=$SOURCE_DATA --target-data-root=$TARGET_DATA --selected-list-dir=$VCTK_SELECTED_LIST --checkpoint-dir=$CHECKPOINTS --hparam-json-file=$HPARAM_FILE --output-dir=$OUTPUT_DIR

#python postprocess_vqcodes.py vctk_selected0
