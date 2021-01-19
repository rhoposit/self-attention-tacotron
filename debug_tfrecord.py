import glob
import sys
import tensorflow as tf
#tf.enable_eager_execution()
from collections import namedtuple
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

class PredictionResult(namedtuple("PredictionResult",
        ["id", "key", "codes", "codes_length", "codes_width"])): pass




def parse_prediction_result(proto):
    features = {
        'id': tf.FixedLenFeature((), tf.int64),
        'key': tf.FixedLenFeature((), tf.string),
        'codes': tf.FixedLenFeature((), tf.string),
        'codes_length': tf.FixedLenFeature((), tf.int64),
        'codes_width': tf.FixedLenFeature((), tf.int64)
    }
    parsed_features = tf.parse_single_example(proto, features)
    return parsed_features

def decode_prediction_result(parsed):
    codes = tf.decode_raw(parsed['codes'], tf.float32)
    codes_length = parsed["codes_length"]
    codes_width = parsed["codes_width"]
    print(codes_length)
    print(codes)
    return PredictionResult(
        id=parsed['id'],
        key=parsed['key'],
        codes=tf.reshape(codes, shape=tf.stack([codes_length, codes_width], axis=0)),
        codes_length=codes_length,
        codes_width=codes_width,
    )




datadir = "/home/smg/v-j-williams/workspace/external_modified/data/sys5_txt_target"
tfiles = glob.glob(datadir+"/*.tfrecord")
#print(tfiles)




count = 0
sess = tf.InteractiveSession()
with sess.as_default():
    for record in tfiles:
        count += 1
        for example in tf.python_io.tf_record_iterator(record):
                r = 1
                features = parse_prediction_result(example)
                result = decode_prediction_result(features)
                fid = result.key.eval().decode('utf-8')
                
                a = np.array([366])
                silence = np.zeros((a.size, 512))
                silence[np.arange(a.size),a] = 1
                silence = np.float32(silence)
                print("* silence shape", silence.shape)
#                print("* silence", silence)
                paddings = [[r, r], [0, 0]]

                codes = result.codes
                preds = np.array(codes.eval()).astype(float)
                print("* codes", tf.shape(codes), codes)
                print("* preds", preds.shape)

                codes_silence = tf.pad(codes, paddings=paddings, mode="CONSTANT")
                preds_padded = np.array(codes_silence.eval()).astype(float)
                print("* codes silence", codes_silence)
#                print("* preds padded", preds_padded[1])
                
                target_length = result.codes_length + 2 * r
                padded_target_length = (target_length // r + 1) * r
                print("* code length", target_length.eval())
                print("* padded code length", padded_target_length.eval())

                def padding_function(t):
                    tail_padding = padded_target_length - target_length
                    print("* tail_padding", tail_padding.eval())
                    padding_shape = tf.sparse_tensor_to_dense(
                        tf.SparseTensor(indices=[(0, 1)], values=tf.expand_dims(tail_padding, axis=0), dense_shape=(2, 2)))
                    return lambda: tf.pad(t, paddings=padding_shape, mode="CONSTANT")
#                    return lambda: tf.pad(t, paddings=padding_shape, constant_values=hparams.silence_mel_level_db)

                zero64 = tf.cast(0, dtype=tf.int64)
                no_padding_condition = tf.equal(zero64, target_length % r)
                print("* no condition (T/F)", no_padding_condition.eval())
                      
                codes = tf.cond(no_padding_condition, lambda: codes_silence, padding_function(codes_silence))
                print("* codes", tf.shape(codes).eval())
                
                padded_target_length = tf.cond(no_padding_condition, lambda: target_length, lambda: padded_target_length)
                print("* padded target length", padded_target_length.eval())

                done = tf.concat([tf.zeros(padded_target_length // r - 1, dtype=tf.float32),
                              tf.ones(1, dtype=tf.float32)], axis=0)
                code_loss_mask = tf.ones(shape=padded_target_length, dtype=tf.float32)
                binary_loss_mask = tf.ones(shape=padded_target_length // r, dtype=tf.float32)
                print("* done", tf.shape(done).eval(), done.eval())
                print("* code_loss_mask", tf.shape(code_loss_mask).eval(), code_loss_mask.eval())
                print("* binary_loss_mask", tf.shape(binary_loss_mask).eval(), binary_loss_mask.eval())

                print(result.id)
                print(result.key)
                print(codes)
                print(result.codes_length)
                print(padded_target_length)
                print(done)
                print(code_loss_mask)
                print(binary_loss_mask)
                

                if count == 1:
                    sys.exit()
