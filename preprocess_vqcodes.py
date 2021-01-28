# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""
Preprocess VCTK dataset
usage: preprocess_vctk.py [options] <in_dir> <out_dir> <version> <num_codes>

options:
    --hparams=<parmas>                  Ad-hoc replacement of hyper parameters. [default: ].
    --version=<version>                 Version number of VCTK.
    --hparam-json-file=<path>           JSON file contains hyper parameters.
    --source-only                       Process source only.
    --target-only                       Process target only.
    -h, --help                          Show help message.

"""

import csv
import os, sys
import numpy as np
import json
from pyspark import SparkContext
from docopt import docopt
from hparams import hparams, hparams_debug_string

if __name__ == "__main__":
    args = docopt(__doc__)
    in_dir = args["<in_dir>"]
    out_dir = args["<out_dir>"]
    version = args["<version>"]
    num_codes = args["<num_codes>"]
    source_only = args["--source-only"]
    target_only = args["--target-only"]


    if args["--hparam-json-file"]:
        with open(args["--hparam-json-file"]) as f:
            hparams_json = "".join(f.readlines())
            hparams.parse_json(hparams_json)

    hparams.parse(args["--hparams"])
    print(hparams_debug_string())

    if source_only:
        process_source = True
        process_target = False
    elif target_only:
        process_source = False
        process_target = True
    else:
        process_source = True
        process_target = True

    from preprocess.codes import CODES
    instance = CODES(in_dir, out_dir, version, num_codes, hparams)

#    sc = SparkContext()

    record_rdd = instance.list_files()

    if process_source:
        keys = instance.process_sources(record_rdd)

    if process_target:
        keys = instance.process_targets(record_rdd)
        
    with open(os.path.join(out_dir, 'list.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for path in keys:
            writer.writerow([path])
