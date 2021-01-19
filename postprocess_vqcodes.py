import glob
import sys
import tensorflow as tf
#tf.enable_eager_execution()
from collections import namedtuple
import numpy as np


class PredictionResult(namedtuple("PredictionResult",
                                  ["id", "key", "codes", "codes_length", "ground_truth_codes","ground_truth_codes_length", "text", "source", "source_length"])): pass




def parse_prediction_result(proto):
    features = {
        'id': tf.FixedLenFeature((), tf.int64),
        'key': tf.FixedLenFeature((), tf.string),
        'codes': tf.FixedLenFeature((), tf.string),
        'codes_length': tf.FixedLenFeature((), tf.int64),
        'ground_truth_codes': tf.FixedLenFeature((), tf.string),
        'ground_truth_codes_length': tf.FixedLenFeature((), tf.int64),
        'text': tf.FixedLenFeature((), tf.string),
        'source': tf.FixedLenFeature((), tf.string),
        'source_length': tf.FixedLenFeature((), tf.int64)
    }
    parsed_features = tf.parse_single_example(proto, features)
    return parsed_features

def decode_prediction_result(parsed):
    source = tf.decode_raw(parsed['source'], tf.int64)
    codes = tf.decode_raw(parsed['codes'], tf.int32)
    codes_length = parsed["codes_length"]
    ground_truth_codes = tf.decode_raw(parsed['ground_truth_codes'], tf.int32)
    ground_truth_codes_length = parsed["ground_truth_codes_length"]
    return PredictionResult(
        id=parsed['id'],
        key=parsed['key'],
        codes=tf.reshape(codes, shape=tf.stack([codes_length], axis=0)),
        codes_length=codes_length,
        ground_truth_codes=tf.reshape(ground_truth_codes, shape=tf.stack([ground_truth_codes_length], axis=0)),
        ground_truth_codes_length=ground_truth_codes_length,
        text=parsed['text'],
        source=source,
        source_length=parsed['source_length'],
    )




datadir = "/gs/hs0/tgh-20IAA/jenn/taco_exp/prediction/"
tfiles = glob.glob(datadir+"/*.tfrecord")
print(tfiles)
outdir = "/gs/hs0/tgh-20IAA/jenn/taco_exp/synth"

sess = tf.InteractiveSession()
with sess.as_default():
    for record in tfiles:
        for example in tf.python_io.tf_record_iterator(record):
            features = parse_prediction_result(example)
            result = decode_prediction_result(features)
            fid = result.key.eval().decode('utf-8')
            truth = np.array(result.ground_truth_codes.eval()).astype(int)
            preds = np.array(result.codes.eval()).astype(int)
            text = result.text.eval().decode('utf-8')

            print(text)
            print(preds)
            print(truth)
            print(fid)

            # save these into a file, in a directory to synthesize from
            outfile = outdir+"/"+fid
            output = open(outfile+".txt", "w")
            output.write(text)
            output.close()
            np.savetxt(outfile+".preds.txt", preds, delimiter=',', fmt='%i')
            np.savetxt(outfile+".truth.txt", truth, delimiter=',', fmt='%i') 
