#!/usr/bin/env python3
"""
Text Content Manipulation
3-gated copy net.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, no-member, too-many-locals

import importlib
import os
import numpy as np
import tensorflow as tf
import texar as tx
import pickle
from utils import *
from get_xx import get_match
from get_xy import get_align
from ie import get_precrec

flags = tf.flags
flags.DEFINE_string("config_data", "config_data_nba", "The data config.")
flags.DEFINE_string("expr_name", "nba", "The experiment name. "
                    "Used as the directory name of run.")
flags.DEFINE_boolean("sd_path", False, "Whether to add structured data path.")
flags.DEFINE_boolean("verbose", False, "verbose.")
flags.DEFINE_boolean("eval_ie", False, "Whether evaluate IE.")
flags.DEFINE_integer("eval_ie_gpuid", 0, "ID of GPU on which IE runs.")
FLAGS = flags.FLAGS

config_data = importlib.import_module(FLAGS.config_data)
expr_name = FLAGS.expr_name

dir_summary = os.path.join(expr_name, 'log')
dir_model = os.path.join(expr_name, 'ckpt')
dir_best = os.path.join(expr_name, 'ckpt-best')
ckpt_model = os.path.join(dir_model, 'model.ckpt')
ckpt_best = os.path.join(dir_best, 'model.ckpt')


def get_replaced(text00, text01, text02, text10, text11, text12, sent_text):
    """Combining match and align. All texts must not contain BOS.
    """
    matches = get_match(text00, text01, text02, text10, text11, text12)
    aligns = get_align(text10, text11, text12, sent_text)
    match = {i: j for i, j in matches}
    n = len(text00)
    m = len(sent_text)
    a = np.zeros([n, m], dtype=np.float32)
    for i in range(n):
        try:
            k = match[i]
        except KeyError:
            continue
        align = aligns[k]
        a[i][:len(align)] = align

    if FLAGS.verbose:
        print(' ' * 20 + ' '.join(map(
            '{:>12}'.format, strip_special_tokens_of_list(text00))))
        for j, sent_token in enumerate(strip_special_tokens_of_list(sent_text)):
            print('{:>20}'.format(sent_token) + ' '.join(map(
                lambda x: '{:>12}'.format(x) if x != 0 else ' ' * 12,
                a[:, j])))

    for j in range(m):
        for i in range(n):
            if a[i][j] != 0:
                sent_text[j] = text00[i]
                break

    return sent_text

def batch_get_replaced(*texts):
    return np.array(batchize(get_replaced)(*texts))


def build_model(data_batch, data):
    batch_size, num_steps = [
        tf.shape(data_batch["value_text_ids"])[d] for d in range(2)]
    vocab = data.vocab('sent')

    tplt_ref_flag, sd_ref_flag = 1, 0
    texts = []
    for ref_flag in [sd_ref_flag, tplt_ref_flag]:
        texts.extend(data_batch['{}{}_text'.format(field, ref_strs[ref_flag])][:, 1:-1]
                     for field in sd_fields)
    texts.extend(data_batch['{}{}_text'.format(field, ref_strs[tplt_ref_flag])][:, 1:]
                 for field in sent_fields)
    replaced = tf.py_func(
        batch_get_replaced, texts, tf.string, stateful=False,
        name='replaced')
    replaced.set_shape(texts[-1].shape)

    return replaced


def main():
    # data batch
    datasets = {mode: tx.data.MultiAlignedData(hparams)
                for mode, hparams in config_data.datas.items()}
    data_iterator = tx.data.FeedableDataIterator(datasets)
    data_batch = data_iterator.get_next()

    predicted_texts = build_model(data_batch, datasets['train'])


    def _eval_epoch(sess, summary_writer, mode):
        print('in _eval_epoch with mode {}'.format(mode))

        data_iterator.restart_dataset(sess, mode)
        feed_dict = {
            tx.global_mode(): tf.estimator.ModeKeys.EVAL,
            data_iterator.handle: data_iterator.get_handle(sess, mode)
        }

        ref_hypo_pairs = []
        fetches = [
            [data_batch['sent_text'], data_batch['sent_ref_text']],
            predicted_texts,
        ]

        if not os.path.exists(dir_model):
            os.makedirs(dir_model)

        hypo_file_name = os.path.join(
            dir_model, "hypos.step{}.{}.txt".format(0, mode))
        hypo_file = open(hypo_file_name, "w")

        cnt = 0
        while True:
            try:
                target_texts, output_texts = sess.run(fetches, feed_dict)
                target_texts = [
                    tx.utils.strip_special_tokens(
                        texts[:, 1:].tolist(), is_token_list=True)
                    for texts in target_texts]
                output_texts = [
                    tx.utils.strip_special_tokens(
                        texts.tolist(), is_token_list=True)
                    for texts in output_texts]
                #output_texts = tx.utils.map_ids_to_strs(
                #    ids=output_ids.tolist(), vocab=datasets[mode].vocab('sent'),
                #    join=False)

                target_texts = list(zip(*target_texts))

                for ref, hypo in zip(target_texts, output_texts):
                    if cnt < 10:
                        print('cnt = {}'.format(cnt))
                        for i, s in enumerate(ref):
                            print('ref{}: {}'.format(i, ' '.join(s)))
                        print('hypo: {}'.format(' '.join(hypo)))
                    print(' '.join(hypo), file=hypo_file)
                    cnt += 1
                print('processed {} samples'.format(cnt))

                ref_hypo_pairs.extend(zip(target_texts, output_texts))

            except tf.errors.OutOfRangeError:
                break

        hypo_file.close()

        if FLAGS.eval_ie:
            gold_file_name = os.path.join(
                config_data.dst_dir, "gold.{}.txt".format(
                    config_data.mode_to_filemode[mode]))
            inter_file_name = "{}.h5".format(hypo_file_name[:-len(".txt")])
            prec, rec = get_precrec(
                gold_file_name, hypo_file_name, inter_file_name,
                gpuid=FLAGS.eval_ie_gpuid)

        refs, hypos = zip(*ref_hypo_pairs)
        bleus = []
        get_bleu_name = '{}_BLEU'.format
        print('In {} mode:'.format(mode))
        for i in range(len(fetches[0])):
            refs_ = list(map(lambda ref: ref[i:i+1], refs))
            bleu = corpus_bleu(refs_, hypos)
            print('{}: {:.2f}'.format(get_bleu_name(i), bleu))
            bleus.append(bleu)

        summary = tf.Summary()
        for i, bleu in enumerate(bleus):
            summary.value.add(
                tag='{}/{}'.format(mode, get_bleu_name(i)), simple_value=bleu)
        if FLAGS.eval_ie:
            for name, value in {'precision': prec, 'recall': rec}.items():
                summary.value.add(tag='{}/{}'.format(mode, name),
                                  simple_value=value)
        summary_writer.add_summary(summary, 0)
        summary_writer.flush()

        bleu = bleus[0]

        print('end _eval_epoch')
        return bleu


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        summary_writer = tf.summary.FileWriter(
            dir_summary, flush_secs=30)

        val_bleu = _eval_epoch(sess, summary_writer, 'val')
        test_bleu = _eval_epoch(sess, summary_writer, 'test')

        print('val BLEU: {:.2f}, test BLEU: {:.2f}'.format(
            val_bleu, test_bleu))

        _eval_epoch(sess, summary_writer, 'train')


if __name__ == '__main__':
    main()
