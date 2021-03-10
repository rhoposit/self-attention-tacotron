import glob
import sys
import tensorflow as tf
#tf.enable_eager_execution()
from collections import namedtuple
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

class PredictionResult(namedtuple("PredictionResult",
    ["id", "key", "codes", "codes_length", "codes_width"])): pass

class Target(namedtuple("Target",
    ["id", "key", "codes",  "codes_length", "codes_width"])): pass

class Source(namedtuple("Source",
    ["id", "key", "source", "source_length", "speaker_id", "age", "gender", "text", "phone", "phone_length", "phone_txt"])): pass



def parse_prediction(proto):
    features = {
        'id': tf.FixedLenFeature((), tf.int64),
        'key': tf.FixedLenFeature((), tf.string),
        'codes': tf.FixedLenFeature((), tf.string),
        'codes_length': tf.FixedLenFeature((), tf.int64),
        'codes_width': tf.FixedLenFeature((), tf.int64)
    }
    parsed_features = tf.parse_single_example(proto, features)
    return parsed_features

def decode_prediction(parsed):
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

def parse_source(proto):
    features = {
        'id': tf.FixedLenFeature((), tf.int64),
        'key': tf.FixedLenFeature((), tf.string),
        'source': tf.FixedLenFeature((), tf.string),
        'source_length': tf.FixedLenFeature((), tf.int64),
        'speaker_id': tf.FixedLenFeature((), tf.int64),
        'age': tf.FixedLenFeature((), tf.int64),
        'gender': tf.FixedLenFeature((), tf.int64),
        'text': tf.FixedLenFeature((), tf.string),
        'phone': tf.FixedLenFeature((), tf.string),
        'phone_length': tf.FixedLenFeature((), tf.int64),
        'phone_txt': tf.FixedLenFeature((), tf.string),
    }
    parsed_features = tf.parse_single_example(proto, features)
    return parsed_features

def decode_source(parsed):
    source = tf.decode_raw(parsed['source'], tf.int64)
    phone = tf.decode_raw(parsed['phone'], tf.int64)
    return Source(
        id=parsed["id"],
        key=parsed["key"],
        source=source,
        source_length=parsed["source_length"],
        speaker_id=parsed["speaker_id"],
        age=parsed["age"],
        gender=parsed["gender"],
        text=parsed["text"],
        phone=phone,
        phone_length=parsed["phone_length"],
        phone_txt=parsed["phone_txt"])


def parse_target(proto):
    features = {
        'id': tf.FixedLenFeature((), tf.int64),
        'key': tf.FixedLenFeature((), tf.string),
        'codes': tf.FixedLenFeature((), tf.string),
        'codes_length': tf.FixedLenFeature((), tf.int64),
        'codes_width': tf.FixedLenFeature((), tf.int64),
    }
    parsed_features = tf.parse_single_example(proto, features)
    return parsed_features


def decode_target(parsed):
    codes_length = parsed["codes_length"]
    codes_width = parsed["codes_width"]
    codes = tf.decode_raw(parsed['codes'], tf.float32)
    return Target(
        id=parsed['id'],
        key=parsed['key'],
        codes=tf.reshape(codes, shape=tf.stack([codes_length, codes_width], axis=0)),
        codes_length=codes_length,
        codes_width=codes_width,
    )


#target_source = sys.argv[1]
#datadir = "/home/smg/v-j-williams/workspace/external_modified/data/vctk_target_selected0"
datadir = "/home/smg/v-j-williams/workspace/external_modified/data/vctk_source_selected"
tfiles = glob.glob(datadir+"/*.tfrecord")
#print(tfiles)




count = 0
sess = tf.InteractiveSession()
with sess.as_default():
    for record in tfiles:
        count += 1
        for example in tf.python_io.tf_record_iterator(record):
                r = 1
                features = parse_source(example)
                result = decode_source(features)
                fid = result.key.eval().decode('utf-8')
                print("source", result.source.eval())
                print("length", result.source_length.eval())
                print("speaker_id", result.speaker_id.eval())
                print("age", result.age.eval())
                print("gender", result.gender.eval())
                print("text", result.text.eval())
                print("phone", result.phone.eval())
                print("length", result.phone_length.eval())
                print("phone_txt", result.phone_txt.eval())

#                features = parse_target(example)
#                result = decode_target(features)
#                fid = result.key.eval().decode('utf-8')
#                print("fid", fid)
#                print("codes", result.codes.eval())
#                print("length", result.codes_length.eval())
#                print("width", result.codes_width.eval())

#                a = np.array([170])
#                silence = np.zeros((a.size, 171))
#                silence[np.arange(a.size),a] = 1
#                silence = np.float32(silence)
#                print("* silence shape", silence.shape)
#                paddings = [[r, r], [0, 0]]

#                codes = result.codes
#                preds = np.array(codes.eval()).astype(float)
#                print("* codes", tf.shape(codes), codes)
#                print("* preds", preds.shape)

#                codes_silence = tf.pad(codes, paddings=paddings, mode="CONSTANT")
#                preds_padded = np.array(codes_silence.eval()).astype(float)
#                print("* codes silence", codes_silence)
#                print("* preds padded", preds_padded[1])
                
#                padded_target_length = (target_length // r + 1) * r
#                print("* code length", target_length.eval())
#                print("* padded code length", padded_target_length.eval())

                def padding_function(t):
                    tail_padding = padded_target_length - target_length
                    print("* tail_padding", tail_padding.eval())
                    padding_shape = tf.sparse_tensor_to_dense(
                        tf.SparseTensor(indices=[(0, 1)], values=tf.expand_dims(tail_padding, axis=0), dense_shape=(2, 2)))
                    return lambda: tf.pad(t, paddings=padding_shape, mode="CONSTANT")
#                    return lambda: tf.pad(t, paddings=padding_shape, constant_values=hparams.silence_mel_level_db)

#                zero64 = tf.cast(0, dtype=tf.int32)
#                no_padding_condition = tf.equal(zero64, target_length % r)
#                print("* no condition (T/F)", no_padding_condition.eval())
                      
#                codes = tf.cond(no_padding_condition, lambda: codes_silence, padding_function(codes_silence))
#                print("* codes", tf.shape(codes).eval())
                
#                padded_target_length = tf.cond(no_padding_condition, lambda: target_length, lambda: padded_target_length)
#                print("* padded target length", padded_target_length.eval())



#                if count == 1:
#                    sys.exit()
