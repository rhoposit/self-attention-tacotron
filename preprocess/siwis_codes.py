# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """

import os, sys
from collections import namedtuple
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from pyspark import RDD, StorageLevel
from utils.tfrecord import write_tfrecord, int64_feature, bytes_feature
from preprocess.cleaners import basic_cleaners
from preprocess.text import text_to_sequence


def write_preprocessed_target_data(_id: int, key: str, codes: np.ndarray, codes_length: int, filename: str):
    raw_codes = codes.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'id': int64_feature([_id]),
        'key': bytes_feature([key.encode('utf-8')]),
        'codes': bytes_feature([raw_codes]),
        'codes_length': int64_feature([codes_length]),
        'codes_width': int64_feature([codes.shape[1]]),
    }))
    write_tfrecord(example, filename)


def write_preprocessed_source_data(_id: int, key: str, source: np.ndarray, text, speaker_id, lang, filename: str):
    raw_source = source.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'id': int64_feature([_id]),
        'key': bytes_feature([key.encode('utf-8')]),
        'source': bytes_feature([raw_source]),
        'source_length': int64_feature([len(source)]),
        'text': bytes_feature([text.encode('utf-8')]),
        'speaker_id': bytes_feature([speaker_id]),
        'lang': bytes_feature([lang]),
    }))
    write_tfrecord(example, filename)


class SpeakerInfo(namedtuple("SpeakerInfo", ["id", "lang"])):
    pass


class TxtCodeRecord(namedtuple("TxtCodeRecord", ["id", "key", "txt_path", "code_path", "speaker_info"])):
    pass


#class MelStatistics(namedtuple("MelStatistics", ["id", "key", "max", "min", "sum", "length", "moment2"])):
#    pass


class TargetRDD:
    def __init__(self, rdd: RDD):
        self.rdd = rdd

    def keys(self):
        return self.rdd.map(lambda s: s.key).collect()

    def max(self):
        return self.rdd.map(lambda s: s.max).reduce(lambda a, b: np.maximum(a, b))

    def min(self):
        return self.rdd.map(lambda s: s.min).reduce(lambda a, b: np.minimum(a, b))

    def average(self):
        total_value = self.rdd.map(lambda s: s.sum).reduce(lambda a, b: a + b)
        total_length = self.rdd.map(lambda s: s.length).reduce(lambda a, b: a + b)
        return total_value / total_length

    def moment2(self):
        total_value = self.rdd.map(lambda s: s.moment2).reduce(lambda a, b: a + b)
        total_length = self.rdd.map(lambda s: s.length).reduce(lambda a, b: a + b)
        return total_value / total_length


class CODES:

    def __init__(self, in_dir, out_dir, hparams, speaker_info_filename='siwis-speaker-info.txt'):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.speaker_info_filename = speaker_info_filename

    def list_files(self):
        def code_files(speaker_info: SpeakerInfo):
            code_dir = self.in_dir
            return [os.path.join(code_dir, code_file) for code_file in sorted(os.listdir(code_dir)) if code_file.endswith('.txt')]

        def text_files(speaker_info: SpeakerInfo):
            txt_dir = self.in_dir
            return [os.path.join(txt_dir, txt_file) for txt_file in sorted(os.listdir(txt_dir)) if txt_file.endswith('.txt')]

        def text_and_code_records(file_pairs, speaker_info):
            def create_record(txt_f, code_f, speaker_info):
                key1 = os.path.basename(code_f).strip('.txt')
                key2 = os.path.basename(txt_f).strip('.txt')
                assert key1 == key2
                return TxtCodeRecord(0, key1, txt_f, code_f, speaker_info)

            return [create_record(txt_f, code_f, speaker_info) for txt_f, code_f in file_pairs]

        records = sum(
            [text_and_code_records(zip(text_files(si), code_files(si)), si) for si in self._load_speaker_info()], [])
        return [TxtCodeRecord(i, r.key, r.txt_path, r.code_path, r.speaker_info) for i, r in enumerate(records)]

    def process_sources(self, rdd: RDD):
        return map(self._process_txt, rdd)
    
    def process_targets(self, rdd: RDD):
        return map(self._process_code, rdd)

    def _load_speaker_info(self):
        with open(self.speaker_info_filename, mode='r', encoding='utf8') as f:
            for l in f.readlines():
                si = l.split()
                yield SpeakerInfo(si[0], si[1])
                    
    def _process_code(self, record: TxtCodeRecord):
        with open(os.path.join(self.in_dir, record.code_path), mode='r', encoding='utf8') as f:
            txt = f.readline().rstrip("\n")
            if len(txt.split("\t")) == 2:
#                print(f)
#                print("text: ", txt.split("\t")[0])
#                print("codes:", txt.split("\t")[1])
                txt = txt.split("\t")[1]
                codelist = txt.split(" ")
                codeints = [int(c) for c in codelist if c != ""]

                # use this when using half of the codes
                codeints = codeints[::2]
                a = np.array(codeints)
                codes = np.zeros((a.size, 512))
                codes[np.arange(a.size),a] = 1
                codes = np.array(codes, np.float32)
                codes_length = a.size
                file_path = os.path.join(self.out_dir, f"{record.key}.target.tfrecord")
                write_preprocessed_target_data(record.id, record.key, codes, codes_length, file_path)
                return record.key

    def _process_txt(self, record: TxtCodeRecord):
        with open(os.path.join(self.in_dir, record.txt_path), mode='r', encoding='utf8') as f:
            txt = f.readline().rstrip("\n").split("\t")[0]
            sequence, clean_text = text_to_sequence(txt, basic_cleaners)
            source = np.array(sequence, dtype=np.int64)
            file_path = os.path.join(self.out_dir, f"{record.key}.source.tfrecord")
            write_preprocessed_source_data(record.id, record.key, source, clean_text, record.speaker_info.id, record.speaker_info.lang, file_path)
            return record.key
