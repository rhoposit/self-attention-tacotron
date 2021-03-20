
# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """

from datasets.codes.dataset import DatasetSource as CODES_DatasetSource
from datasets.codes_siwis.dataset import DatasetSource as CODES_SIWIS_DatasetSource
#from datasets.vctk.dataset import DatasetSource as VCTKDatasetSource
#from datasets.ljspeech.dataset import DatasetSource as LJSpeechDatasetSource


def dataset_factory(source, target, hparams):
    if hparams.dataset == "codes.dataset.DatasetSource":
        return CODES_DatasetSource(source, target, hparams)
    elif hparams.dataset == "codes_siwis.dataset.DatasetSource":
        return CODES_SIWIS_DatasetSource(source, target, hparams)
    else:
        raise ValueError("Unkown dataset")


def create_from_tfrecord_files(source_files, target_files, hparams, cycle_length=4,
                               buffer_output_elements=None,
                               prefetch_input_elements=None):
    if hparams.dataset == "codes.dataset.DatasetSource":
        return CODES_DatasetSource.create_from_tfrecord_files(source_files, target_files, hparams, cycle_length=cycle_length, buffer_output_elements=buffer_output_elements, prefetch_input_elements=prefetch_input_elements)
    elif hparams.dataset == "codes_siwis.dataset.DatasetSource":
        return CODES_SIWIS_DatasetSource.create_from_tfrecord_files(source_files, target_files, hparams, cycle_length=cycle_length, buffer_output_elements=buffer_output_elements, prefetch_input_elements=prefetch_input_elements)
    else:
        raise ValueError("Unkown dataset")
