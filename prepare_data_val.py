# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Produces TFRecords files and modifies data configuration file
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import texar as tx
import sys
import nltk

# pylint: disable=no-name-in-module
sys.path.append('./bert/utils')
import data_utils, tokenization

# pylint: disable=intest-name, too-many-locals, too-many-statements
nltk.download('punkt')
flags = tf.flags
flags.DEFINE_string(
    "task", "SST",
    "The task to run experiment on. ")
flags.DEFINE_string(
    "vocab_file", 'bert/bert_config/all.vocab.txt',
    "The one-wordpiece-per-line vocabary file directory.")
flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maxium length of sequence, longer sequence will be trimmed.")
flags.DEFINE_string(
    "tfrecords_output_dir", 'bert/E2E',
    "The output directory where the TFRecords files will be generated.")
flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
flags.DEFINE_string(
    "expr_name", "rule",
    "The output directory where main_ours.py generate")
flags.DEFINE_string(
    "step", "0",
    "The step to compute two scores")
tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = flags.FLAGS

expr_name = FLAGS.expr_name
e2e_data_dir = "e2e_data/val"
step = FLAGS.step
refs = ['', '_ref']


def prepare_data():
    """
    Builds the model and runs.
    """
    # Loads data
    tf.logging.info("Loading data")

    # task_datasets_rename = {
    #     "SST": "E2E",
    # }

    data_dir = 'bert/{}'.format('E2E')
    # if FLAGS.task.upper() in task_datasets_rename:
    #     data_dir = 'data/{}'.format(
    #         task_datasets_rename[FLAGS.task])

    if FLAGS.tfrecords_output_dir is None:
        tfrecords_output_dir = data_dir
    else:
        tfrecords_output_dir = FLAGS.tfrecords_output_dir
    tx.utils.maybe_create_dir(tfrecords_output_dir)

    processors = {
        'SST': data_utils.SSTProcessor
    }
    processor = processors[FLAGS.task]()

    num_classes = len(processor.get_labels())
    num_train_data = len(processor.get_train_examples(data_dir))

    tf.logging.info(
        'num_classes:%d; num_train_data:%d' % (num_classes, num_train_data))
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file,
        do_lower_case=FLAGS.do_lower_case)
    # TO DO:Prepare data for the transformer classifier
    # i.e. Concat x' with y and see whether x' was compressed in y
    ref = refs[1]
    with open(os.path.join(e2e_data_dir, "x{}_type.valid.txt".format(ref)), 'r') as f_type:
        lines_type = f_type.readlines()
    with open(os.path.join(e2e_data_dir, "x{}_value.valid.txt".format(ref)), 'r') as f_entry:
        lines_entry = f_entry.readlines()
    with open(os.path.join(e2e_data_dir, "x{}_value.valid.txt".format(refs[0])), 'r') as f_entry_x:
        lines_entry_x = f_entry_x.readlines()
    with open("e2ev14_output_new/{}/ckpt/hypos.step{}.val.txt".format(expr_name, step), 'r') as f_sent:
        lines_sent = f_sent.readlines()
        for (idx_line, line_type) in enumerate(lines_type):
            line_type = line_type.strip('\n').split(' ')
            for (idx_val, attr) in enumerate(line_type):
                entry_list = lines_entry[idx_line].strip('\n').split(' ')
                if (lines_entry_x[idx_line].find(entry_list[idx_val]) == -1):
                    neg_samp = attr + ' : ' + entry_list[idx_val] + ' | ' + lines_sent[idx_line]
                    with open("bert/E2E/{}.step{}.2.tsv".format(expr_name, step), 'a') as f_w:
                        f_w.write(neg_samp)

    # Concat x with y and see whether x was compressed in y
    ref = refs[0]
    with open(os.path.join(e2e_data_dir, "x{}_type.valid.txt".format(ref)), 'r') as f_type:
        lines_type = f_type.readlines()
    with open(os.path.join(e2e_data_dir, "x{}_value.valid.txt".format(ref)), 'r') as f_entry:
        lines_entry = f_entry.readlines()
    with open("e2ev14_output_new/{}/ckpt/hypos.step{}.val.txt".format(expr_name, step), 'r') as f_sent:
        lines_sent = f_sent.readlines()
        for (idx_line, line_type) in enumerate(lines_type):
            line_type = line_type.strip('\n').split(' ')
            for (idx_val, attr) in enumerate(line_type):
                entry_list = lines_entry[idx_line].strip('\n').split(' ')
                pos_samp = attr + ' : ' + entry_list[idx_val] + ' | ' + lines_sent[idx_line]
                with open("bert/E2E/{}.step{}.1.tsv".format(expr_name, step), 'a') as f_w:
                    f_w.write(pos_samp)

    # Produces TFRecords files
    data_utils.prepare_TFRecord_data(
        processor=processor,
        tokenizer=tokenizer,
        data_dir=data_dir,
        max_seq_length=FLAGS.max_seq_length,
        output_dir=tfrecords_output_dir,
        expr_name=expr_name,
        step=step)
    modify_config_data(FLAGS.max_seq_length, num_train_data, num_classes)


def modify_config_data(max_seq_length, num_train_data, num_classes):
    # Modify the data configuration file
    config_data_exists = os.path.isfile('./config_data.py')
    if config_data_exists:
        with open("./config_data.py", 'r') as file:
            filedata = file.read()
            filedata_lines = filedata.split('\n')
            idx = 0
            while True:
                if idx >= len(filedata_lines):
                    break
                line = filedata_lines[idx]
                if (line.startswith('num_classes =') or
                        line.startswith('num_train_data =') or
                        line.startswith('max_seq_length =')):
                    filedata_lines.pop(idx)
                    idx -= 1
                idx += 1

            if len(filedata_lines) > 0:
                insert_idx = 1
            else:
                insert_idx = 0
            filedata_lines.insert(
                insert_idx, '{} = {}'.format(
                    "num_train_data", num_train_data))
            filedata_lines.insert(
                insert_idx, '{} = {}'.format(
                    "num_classes", num_classes))
            filedata_lines.insert(
                insert_idx, '{} = {}'.format(
                    "max_seq_length", max_seq_length))

        with open("./config_data.py", 'w') as file:
            file.write('\n'.join(filedata_lines))
        tf.logging.info("config_data.py has been updated")
    else:
        tf.logging.info("config_data.py cannot be found")

    tf.logging.info("Data preparation finished")


def main():
    """ Starts the data preparation
    """
    prepare_data()


if __name__ == "__main__":
    main()
