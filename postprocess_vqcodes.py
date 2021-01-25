import glob
import sys
import tensorflow as tf
from collections import namedtuple
import numpy as np
np.set_printoptions(threshold=sys.maxsize)


class PredictionResult(namedtuple("PredictionResult",
                                  ["id", "key", "codes", "codes_length", "codes_width", "ground_truth_codes","ground_truth_codes_length", "text", "source", "source_length"])): pass




def parse_prediction_result(proto):
    features = {
        'id': tf.FixedLenFeature((), tf.int64),
        'key': tf.FixedLenFeature((), tf.string),
        'codes': tf.FixedLenFeature((), tf.string),
        'codes_length': tf.FixedLenFeature((), tf.int64),
        'codes_width': tf.FixedLenFeature((), tf.int64),
        'ground_truth_codes': tf.FixedLenFeature((), tf.string),
        'ground_truth_codes_length': tf.FixedLenFeature((), tf.int64),
        'text': tf.FixedLenFeature((), tf.string),
        'source': tf.FixedLenFeature((), tf.string),
        'source_length': tf.FixedLenFeature((), tf.int64),
    }
    parsed_features = tf.parse_single_example(proto, features)
    return parsed_features

def decode_prediction_result(parsed):
    source = tf.decode_raw(parsed['source'], tf.int64)
    codes = tf.decode_raw(parsed['codes'], tf.int32)
    codes_length = parsed["codes_length"]
    codes_width = parsed["codes_width"]
    ground_truth_codes = tf.decode_raw(parsed['ground_truth_codes'], tf.int32)
    ground_truth_codes_length = parsed["ground_truth_codes_length"]
    return PredictionResult(
        id=parsed['id'],
        key=parsed['key'],
        codes=tf.reshape(codes, shape=tf.stack([codes_length, codes_width], axis=0)),
        codes_length=codes_length,
        codes_width=codes_width,
        ground_truth_codes=tf.reshape(ground_truth_codes, shape=tf.stack([ground_truth_codes_length, codes_width], axis=0)),
        ground_truth_codes_length=ground_truth_codes_length,
        text=parsed['text'],
        source=source,
        source_length=parsed['source_length'],
    )




datadir = "/home/smg/v-j-williams/workspace/external_modified/prediction/vctk/"
tfiles = glob.glob(datadir+"/*.tfrecord")
print(tfiles)
outdir = "/home/smg/v-j-williams/workspace/external_modified/synth/vctk/"


txtlist = []
predlist = []
truthlist = []
synthoutdir = "/home/smg/v-j-williams/workspace/tsubame_work/special/mt_lists/"

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
            codes_pred = np.argmax(preds, axis=1)
            codes_truth = np.argmax(truth, axis=1)            
            print(text)
            print(fid)

            # save these into a file, in a directory to synthesize from
            outfile = outdir+"/"+fid
            output = open(outfile+".txt", "w")
            output.write(text)
            output.close()
            codes_pred_list = list(codes_pred)
            codes_pred_list = [str(c) for c in codes_pred_list]
            outstring = " ".join(codes_pred_list)+"\n"
            output = open(outfile+".preds.txt", "w")
            output.write(outstring)
            output.close()
            codes_truth_list = list(codes_truth)
            codes_truth_list = [str(c) for c in codes_truth_list]
            outstring = " ".join(codes_truth_list)+"\n"
            output = open(outfile+".truth.txt", "w")
            output.write(outstring)
            output.close()
            
            txtlist.append(fid+".txt")
            predlist.append(" ".join(codes_pred_list))
            truthlist.append( " ".join(codes_truth_list))
            
output = open(synthoutdir+"tacotron.txt", "w")
output.write("\n".join(txtlist))
output.close()
output = open(synthoutdir+"tacotron.hypothesis.txt", "w")
output.write("\n".join(predlist))
output.close()
output = open(synthoutdir+"tacotron.true.txt", "w")
output.write("\n".join(truthlist))
output.close()
                
