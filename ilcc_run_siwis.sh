

# rsync special files to scratch space
# generate the data on scratch space
# rsync the data to home

DATASET=vqcodes
NAME=all_siwis_161
ORIG_DATA=/home/s1738075/special/L1_dat_files/sys5_lang/siwis_552024/${NAME} 

SCRATCH=/disk/scratch/s1738075
SCRATCH_DATA=/disk/scratch/s1738075/data
SCRATCH_CHECK=/disk/scratch/s1738075/checkpoints
SCRATCH_DATA_RAW=/disk/scratch/s1738075/data/${NAME}
SOURCE_DATA=/disk/scratch/s1738075/data/${NAME}_source
TARGET_DATA=/disk/scratch/s1738075/data/${NAME}_target
CHECKPOINTS=/disk/scratch/s1738075/checkpoints/${NAME}
LIST=/home/s1738075/taco_modified/self_attention_tacotron/examples/codes
HPARAM_FILE=/home/s1738075/taco_modified/self_attention_tacotron/examples/codes/self-attention-tacotron.json
OUTPUT_DIR=/home/s1738075/taco_modified/prediction/${NAME}


#mkdir $SCRATCH
#mkdir $SCRATCH_DATA
#mkdir $SCRATCH_CHECK
#mkdir ${CHECKPOINTS}
#mkdir $SOURCE_DATA
#mkdir $TARGET_DATA
#rsync -ruva $ORIG_DATA $SCRATCH_DATA

python preprocess_vqcodes.py --target-only --hparams=phoneme=flite,flite_binary_path='/home/s1738075/taco_modified/flite' $SCRATCH_DATA_RAW $TARGET_DATA siwis 0 161
rsync -ruva $TARGET_DATA /home/s1738075/data/

python preprocess_vqcodes.py --source-only --hparams=phoneme=flite,flite_binary_path='/home/s1738075/taco_modified/flite' $SCRATCH_DATA_RAW $SOURCE_DATA siwis 0 161
rsync -ruva $SOURCE_DATA /home/s1738075/data/

#rsync -ruva $SCRATCH_DATA_RAW /home/s1738075/data/
exit


export PYTHONPATH=/home/s1738075/taco_modified:/home/s1738075/taco_modified/tacotron2:/home/s1738075/taco_modified/self_attention_tacotron:/home/s1738075/taco_modified/multi_speaker_tacotron:/home/s1738075/miniconda3/envs/taco/lib/python3.6/site-packages:/home/s1738075/miniconda3/envs/taco/lib/python3.6/site-packages/mkl:$PYTHONPATH
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=0

python train.py --source-data-root=$SOURCE_DATA --target-data-root=$TARGET_DATA --selected-list-dir=$LIST --checkpoint-dir=$CHECKPOINTS --hparam-json-file=$HPARAM_FILE

#python predict_code.py  --source-data-root=$SOURCE_DATA --target-data-root=$TARGET_DATA --selected-list-dir=$LIST --checkpoint-dir=$CHECKPOINTS --hparam-json-file=$HPARAM_FILE --output-dir=$OUTPUT_DIR
#python postprocess_vqcodes.py siwis
