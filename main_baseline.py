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
from collections import namedtuple
from copy_net import CopyNetWrapper
from texar.core import get_train_op
from utils import *
from ie import get_precrec

flags = tf.flags
flags.DEFINE_string("dataset", "nba", "The dataset.")
flags.DEFINE_string("config_data", "config_data", "The data config.")
flags.DEFINE_string("config_model", "config_model", "The model config.")
flags.DEFINE_string("config_train", "config_train_baseline", "The training config.")
flags.DEFINE_float("bt_w", 1, "Back-translation weight.")
flags.DEFINE_float("adv_w", 0.5, "Adversarial loss weight.")
## MAST -- bt_w=0.1
## AdvST -- bt_w=1 adv_w=0.5

flags.DEFINE_string("save_path", "nba_baseline", "The save path of your "
                                             "experiment. ")
flags.DEFINE_string("restore_from", "", "The specific checkpoint path to "
                    "restore from. If not specified, the latest checkpoint in "
                    "expr_name is used.")
flags.DEFINE_boolean("copy_x", False, "Whether to copy from x.")
flags.DEFINE_boolean("coverage", False, "Whether to add coverage onto the copynets.")
flags.DEFINE_float("exact_cover_w", 0., "Weight of exact coverage loss.")
flags.DEFINE_float("eps", 1e-10, "epsilon used to avoid log(0).")
flags.DEFINE_integer("disabled_vocab_size", 0 , "Disabled vocab size.")
flags.DEFINE_boolean("attn_x", False, "Whether to attend x.")
flags.DEFINE_boolean("attn_y_", False, "Whether to attend y'.")
flags.DEFINE_boolean("verbose", False, "verbose.")
flags.DEFINE_boolean("eval_ie", False, "Whether evaluate IE.")
flags.DEFINE_integer("eval_ie_gpuid", 0, "ID of GPU on which IE runs.")
FLAGS = flags.FLAGS

copy_flag = FLAGS.copy_x or FLAGS.copy_y_
attn_flag = FLAGS.attn_x or FLAGS.attn_y_

config_data = importlib.import_module(FLAGS.config_data)
config_model = importlib.import_module(FLAGS.config_model)
config_train = importlib.import_module(FLAGS.config_train)
expr_name = FLAGS.expr_name
restore_from = FLAGS.restore_from

dir_summary = os.path.join(expr_name, 'log')
dir_model = os.path.join(expr_name, 'ckpt')
dir_best = os.path.join(expr_name, 'ckpt-best')
ckpt_model = os.path.join(dir_model, 'model.ckpt')
ckpt_best = os.path.join(dir_best, 'model.ckpt')


class Encoded(namedtuple('Encoded', ['ids', 'embeds', 'enc_outputs', 'length'])):
    pass


def get_optimistic_restore_variables(ckpt_path, graph=tf.get_default_graph()):
    reader = tf.train.NewCheckpointReader(ckpt_path)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([
        (var.name, var.name.split(':')[0]) for var in tf.global_variables()
        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    for var_name, saved_var_name in var_names:
        var = graph.get_tensor_by_name(var_name)
        var_shape = var.get_shape().as_list()
        if var_shape == saved_shapes[saved_var_name]:
            restore_vars.append(var)
    return restore_vars


def get_optimistic_saver(ckpt_path, graph=tf.get_default_graph()):
    return tf.train.Saver(
        get_optimistic_restore_variables(ckpt_path, graph=graph))


def build_model(data_batch, data, step):
    batch_size, num_steps = [
        tf.shape(data_batch["x_value_text_ids"])[d] for d in range(2)]
    vocab = data.vocab('y_aux')

    id2str = '<{}>'.format
    bos_str, eos_str = map(id2str, (vocab.bos_token_id, vocab.eos_token_id))

    def single_bleu(ref, hypo):
        ref = [id2str(u if u != vocab.unk_token_id else -1) for u in ref]
        hypo = [id2str(u) for u in hypo]

        ref = tx.utils.strip_special_tokens(
            ' '.join(ref), strip_bos=bos_str, strip_eos=eos_str)
        hypo = tx.utils.strip_special_tokens(
            ' '.join(hypo), strip_eos=eos_str)

        return 0.01 * tx.evals.sentence_bleu(references=[ref], hypothesis=hypo)

    def batch_bleu(refs, hypos):
        return np.array(
            [single_bleu(ref, hypo) for ref, hypo in zip(refs, hypos)],
            dtype=np.float32)


    # losses
    losses = {}

    # embedders
    embedders = {
        name: tx.modules.WordEmbedder(
            vocab_size=data.vocab(name).size, hparams=hparams)
        for name, hparams in config_model.embedders.items()}

    # encoders
    y_encoder = tx.modules.BidirectionalRNNEncoder(
        hparams=config_model.y_encoder)
    x_encoder = tx.modules.BidirectionalRNNEncoder(
        hparams=config_model.x_encoder)


    def concat_encoder_outputs(outputs):
        return tf.concat(outputs, -1)


    def encode_y(ids, length):
        embeds = embedders['y_aux'](ids)
        enc_outputs, _ = y_encoder(embeds, sequence_length=length)
        enc_outputs = concat_encoder_outputs(enc_outputs)
        return Encoded(ids, embeds, enc_outputs, length)


    def encode_x(ids, length):
        embeds = tf.concat(
            [embedders['x_'+field](ids[field]) for field in x_fields],
            axis=-1)
        enc_outputs, _ = x_encoder(embeds, sequence_length=length)
        enc_outputs = concat_encoder_outputs(enc_outputs)
        return Encoded(ids, embeds, enc_outputs, length)


    y_encoded = [
        encode_y(
            data_batch['{}_text_ids'.format(ref_str)],
            data_batch['{}_length'.format(ref_str)])
        for ref_str in y_strs]
    x_encoded = [
        encode_x(
            {field: data_batch['x{}_{}_text_ids'.format(ref_str, field)][:, 1:-1]
             for field in x_fields},
            data_batch['x{}_{}_length'.format(ref_str, x_fields[0])] - 2)
        for ref_str in ref_strs]

    # get rnn cell
    rnn_cell = tx.core.layers.get_rnn_cell(config_model.rnn_cell)


    def get_decoder(cell, y_, x, tgt_ref_flag, beam_width=None):
        output_layer_params = \
            {'output_layer': tf.identity} if copy_flag else \
            {'vocab_size': vocab.size}

        if attn_flag: # attention
            if FLAGS.attn_x and FLAGS.attn_y_:
                memory = tf.concat(
                    [y_.enc_outputs,
                     x.enc_outputs],
                    axis=1)
                memory_sequence_length = None
            elif FLAGS.attn_y_:
                memory = y_.enc_outputs
                memory_sequence_length = y_.length
            elif FLAGS.attn_x:
                memory = x.enc_outputs
                memory_sequence_length = x.length
            attention_decoder = tx.modules.AttentionRNNDecoder(
                cell=cell,
                memory=memory,
                memory_sequence_length=memory_sequence_length,
                hparams=config_model.attention_decoder,
                **output_layer_params)
            if not copy_flag:
                return attention_decoder
            cell = attention_decoder.cell if beam_width is None else \
                   attention_decoder._get_beam_search_cell(beam_width)

        if copy_flag: # copynet
            kwargs = {
                'y__ids': y_.ids[:, 1:],
                'y__states': y_.enc_outputs[:, 1:],
                'y__lengths': y_.length - 1,
                'x_ids': x.ids['value'],
                'x_states': x.enc_outputs,
                'x_lengths': x.length,
            }

            if tgt_ref_flag is not None:
                kwargs.update({
                'input_ids': data_batch[
                     '{}_text_ids'.format(y_strs[tgt_ref_flag])][:, :-1]})

            memory_prefixes = []

            if FLAGS.copy_y_:
                memory_prefixes.append('y_')

            if FLAGS.copy_x:
                memory_prefixes.append('x')

            if beam_width is not None:
                kwargs = {
                    name: tile_batch(value, beam_width)
                    for name, value in kwargs.items()}

            def get_get_copy_scores(memory_ids_states_lengths, output_size):
                memory_copy_states = [
                    tf.layers.dense(
                        memory_states,
                        units=output_size,
                        activation=None,
                        use_bias=False)
                    for _, memory_states, _ in memory_ids_states_lengths]

                def get_copy_scores(query, coverities=None):
                    ret = []

                    if FLAGS.copy_y_:
                        memory = memory_copy_states[len(ret)]
                        if coverities is not None:
                            memory = memory + tf.layers.dense(
                                coverities[len(ret)],
                                units=output_size,
                                activation=None,
                                use_bias=False)
                        memory = tf.nn.tanh(memory)
                        ret_y_ = tf.einsum("bim,bm->bi", memory, query)
                        ret.append(ret_y_)

                    if FLAGS.copy_x:
                        memory = memory_copy_states[len(ret)]
                        if coverities is not None:
                            memory + memory + tf.layers.dense(
                                coverities[len(ret)],
                                units=output_size,
                                activation=None,
                                use_bias=False)
                        memory = tf.nn.tanh(memory)
                        ret_x = tf.einsum("bim,bm->bi", memory, query)
                        ret.append(ret_x)

                    return ret

                return get_copy_scores

            cell = CopyNetWrapper(
                cell=cell, vocab_size=vocab.size,
                memory_ids_states_lengths=[
                    tuple(kwargs['{}_{}'.format(prefix, s)]
                          for s in ('ids', 'states', 'lengths'))
                    for prefix in memory_prefixes],
                input_ids=\
                    kwargs['input_ids'] if tgt_ref_flag is not None else None,
                get_get_copy_scores=get_get_copy_scores,
                coverity_dim=config_model.coverity_dim if FLAGS.coverage else None,
                coverity_rnn_cell_hparams=config_model.coverity_rnn_cell if FLAGS.coverage else None,
                disabled_vocab_size=FLAGS.disabled_vocab_size,
                eps=FLAGS.eps)

        decoder = tx.modules.BasicRNNDecoder(
            cell=cell, hparams=config_model.decoder,
            **output_layer_params)
        return decoder

    def get_decoder_and_outputs(
            cell, y_, x, tgt_ref_flag, params,
            beam_width=None):
        decoder = get_decoder(
            cell, y_, x, tgt_ref_flag,
            beam_width=beam_width)
        if beam_width is None:
            ret = decoder(**params)
        else:
            ret = tx.modules.beam_search_decode(
                decoder_or_cell=decoder,
                beam_width=beam_width,
                **params)
        return (decoder,) + ret

    get_decoder_and_outputs = tf.make_template(
        'get_decoder_and_outputs', get_decoder_and_outputs)

    def teacher_forcing(cell, y_, x_ref_flag, loss_name, add_bleu_weight=False):
        x = x_encoded[x_ref_flag]
        tgt_ref_flag = x_ref_flag
        tgt_str = y_strs[tgt_ref_flag]
        sequence_length = data_batch['{}_length'.format(tgt_str)] - 1
        decoder, tf_outputs, final_state, _ = get_decoder_and_outputs(
            cell, y_, x, tgt_ref_flag,
            {'decoding_strategy': 'train_greedy',
             'inputs': y_encoded[tgt_ref_flag].embeds,
             'sequence_length': sequence_length})

        tgt_y_ids = data_batch['{}_text_ids'.format(tgt_str)][:, 1:]
        loss = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=tgt_y_ids,
            logits=tf_outputs.logits,
            sequence_length=sequence_length,
            average_across_batch=False)
        if add_bleu_weight:
            w = tf.py_func(
                batch_bleu, [y_.ids, tgt_y_ids],
                tf.float32, stateful=False, name='W_BLEU')
            w.set_shape(loss.get_shape())
            loss = w * loss
        loss = tf.reduce_mean(loss, 0)

        if copy_flag and FLAGS.exact_cover_w != 0:
            sum_copy_probs = list(map(lambda t: tf.cast(t, tf.float32), final_state.sum_copy_probs))
            memory_lengths = [lengths for _, _, lengths in decoder.cell.memory_ids_states_lengths]
            exact_coverage_losses = [
                tf.reduce_mean(tf.reduce_sum(
                    tx.utils.mask_sequences(
                        tf.square(sum_copy_prob - 1.), memory_length),
                    1))
                for sum_copy_prob, memory_length in zip(sum_copy_probs, memory_lengths)]
            for i, exact_coverage_loss in enumerate(exact_coverage_losses):
                loss += FLAGS.exact_cover_w * exact_coverage_loss

        losses[loss_name] = loss

        return decoder, tf_outputs, loss


    def beam_searching(cell, y_, x, beam_width):
        start_tokens = tf.ones_like(data_batch['y_aux_length']) * \
            vocab.bos_token_id
        end_token = vocab.eos_token_id

        decoder, bs_outputs, _, bs_length = get_decoder_and_outputs(
            cell, y_, x, None,
            {'embedding': embedders['y_aux'],
             'start_tokens': start_tokens,
             'end_token': end_token,
             'max_decoding_length': config_train.infer_max_decoding_length},
            beam_width=beam_width)

        return decoder, bs_outputs, bs_length


    def greedy_decoding(cell, y_, x):
        decoder, bs_outputs, bs_length = beam_searching(cell, y_, x, 1)
        return bs_outputs.predicted_ids[:, :, 0], bs_length[:, 0]

    discriminator = tx.modules.UnidirectionalRNNClassifier(hparams=config_model.discriminator)
    def disc(y, x):
        return discriminator(tf.concat([x.enc_outputs, y], axis=1))

    joint_loss = 0.
    disc_loss = 0.
    for flag in range(2):
        xe_decoder, xe_outputs, xe_loss = teacher_forcing(rnn_cell, y_encoded[1-flag], flag, 'XE')
        rec_decoder, rec_outputs, rec_loss = teacher_forcing(rnn_cell, y_encoded[1-flag], 1-flag, 'REC')

        # print('[info]rec_outputs is:{}'.format(rec_outputs.cell_output))

        greedy_ids, greedy_length = greedy_decoding(rnn_cell, y_encoded[1-flag], x_encoded[flag])
        greedy_ids = tf.concat([data_batch['y_aux_text_ids'][:, :1], tf.cast(greedy_ids, tf.int64)], axis=1)
        greedy_length = 1 + greedy_length
        greedy_encoded = encode_y(greedy_ids, greedy_length)

        bt_decoder, _, bt_loss = teacher_forcing(rnn_cell, greedy_encoded, 1-flag, 'BT')


        adv_loss = 2. * disc(rec_outputs.cell_output, x_encoded[1-flag])[0][:, 1] \
                 +      disc(xe_outputs.cell_output, x_encoded[flag])[0][:, 0] \
                 +      disc(rec_outputs.cell_output, x_encoded[flag])[0][:, 0]
        adv_loss = tf.reduce_mean(adv_loss, 0)

        disc_loss = disc_loss + adv_loss
        joint_loss = joint_loss + (rec_loss + FLAGS.bt_w * bt_loss + FLAGS.adv_w * adv_loss)

    disc_loss = -disc_loss

    losses['joint'] = joint_loss
    losses['disc'] = disc_loss

    tiled_decoder, bs_outputs, _ = beam_searching(
        rnn_cell, y_encoded[1], x_encoded[0], config_train.infer_beam_width)

    disc_variables = discriminator.trainable_variables
    other_variables = list(filter(lambda var: var not in disc_variables, tf.trainable_variables()))
    variables_of_train_op = {
        'joint': other_variables,
        'disc': disc_variables,
    }

    train_ops = {
        name: get_train_op(
            losses[name],
            variables=variables_of_train_op.get(name, None),
            hparams=config_train.train[name])
        for name in config_train.train}

    return train_ops, bs_outputs


def main():
    # data batch
    datasets = {mode: tx.data.MultiAlignedData(hparams)
                for mode, hparams in config_data.datas.items()}
    data_iterator = tx.data.FeedableDataIterator(datasets)
    data_batch = data_iterator.get_next()

    global_step = tf.train.get_or_create_global_step()

    train_ops, bs_outputs\
        = build_model(data_batch, datasets['train'], global_step)

    summary_ops = {
        name: tf.summary.merge(
            tf.get_collection(
                tf.GraphKeys.SUMMARIES,
                scope=get_scope_name_of_train_op(name)),
            name=get_scope_name_of_summary_op(name))
        for name in train_ops.keys()}

    saver = tf.train.Saver(max_to_keep=None)

    global best_ever_val_bleu
    best_ever_val_bleu = 0.


    def _save_to(directory, step):
        print('saving to {} ...'.format(directory))

        saved_path = saver.save(sess, directory, global_step=step)

        print('saved to {}'.format(saved_path))


    def _restore_from_path(ckpt_path):
        print('restoring from {} ...'.format(ckpt_path))

        try:
            saver.restore(sess, ckpt_path)
        except tf.errors.NotFoundError:
            print('Some variables are missing. Try optimistically restoring.')
            (get_optimistic_saver(ckpt_path)).restore(sess, ckpt_path)

        print('done.')


    def _restore_from(directory):
        if os.path.exists(directory):
            ckpt_path = tf.train.latest_checkpoint(directory)
            _restore_from_path(ckpt_path)

        else:
            print('cannot find checkpoint directory {}'.format(directory))

    def _train_epoch(sess, summary_writer, mode, train_ops, summary_ops, names):
        print('in _train_epoch')

        data_iterator.restart_dataset(sess, mode)
        feed_dict = {
            tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
            data_iterator.handle: data_iterator.get_handle(sess, mode),
        }

        while True:
            try:
                losses, summaries = sess.run((train_ops, summary_ops), feed_dict)

                step = tf.train.global_step(sess, global_step)

                print('step {:d}:\t{}'.format(step, '\t'.join('{}: {:.6f}'.format(name, losses[name]) for name in names)))

                for summary in summaries.values():
                    summary_writer.add_summary(summary, step)

                if step % config_train.steps_per_eval == 0:
                    _eval_epoch(sess, summary_writer, 'val')

            except tf.errors.OutOfRangeError:
                break

        print('end _train_epoch')


    def _eval_epoch(sess, summary_writer, mode):
        global best_ever_val_bleu

        print('in _eval_epoch with mode {}'.format(mode))

        data_iterator.restart_dataset(sess, mode)
        feed_dict = {
            tx.global_mode(): tf.estimator.ModeKeys.EVAL,
            data_iterator.handle: data_iterator.get_handle(sess, mode)
        }

        step = tf.train.global_step(sess, global_step)

        ref_hypo_pairs = []
        fetches = [
            [data_batch['y_aux_text'], data_batch['y_ref_text']],
            [data_batch['x_value_text'], data_batch['x_ref_value_text']],
            bs_outputs.predicted_ids,
        ]

        if not os.path.exists(dir_model):
            os.makedirs(dir_model)

        hypo_file_name = os.path.join(
            dir_model, "hypos.step{}.{}.txt".format(step, mode))
        hypo_file = open(hypo_file_name, "w")

        cnt = 0
        while True:
            try:
                target_texts,entry_texts, output_ids = sess.run(fetches, feed_dict)
                target_texts = [
                    tx.utils.strip_special_tokens(
                        texts[:, 1:].tolist(), is_token_list=True)
                    for texts in target_texts]
                entry_texts = [
                    tx.utils.strip_special_tokens(
                        texts[:, 1:].tolist(), is_token_list=True)
                    for texts in entry_texts]
                output_ids = output_ids[:, :, 0]
                output_texts = tx.utils.map_ids_to_strs(
                    ids=output_ids.tolist(), vocab=datasets[mode].vocab('y_aux'),
                    join=False)

                target_texts = list(zip(*target_texts))
                entry_texts = list(zip(*entry_texts))
                for ref, hypo in zip(target_texts, output_texts):
                    if cnt < 10:
                        print('cnt = {}'.format(cnt))
                        for i, s in enumerate(ref):
                            print('ref{}: {}'.format(i, ' '.join(s)))
                        print('hypo: {}'.format(' '.join(hypo)))
                    print(' '.join(hypo), file=hypo_file)
                    cnt += 1
                print('processed {} samples'.format(cnt))

                ref_hypo_pairs.extend(zip(target_texts, entry_texts, output_texts))

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

        refs, entrys, hypos = zip(*ref_hypo_pairs)
        bleus = []
        get_bleu_name = '{}_BLEU'.format
        print('In {} mode:'.format(mode))
        for i in range(1, 2):
            refs_ = list(map(lambda ref: ref[i:i+1], refs))
            ents_ = list(map(lambda ent: ent[i:i+1], entrys))
            bleu = corpus_bleu(refs_, hypos)
            print('==========={}: {:.2f}'.format(get_bleu_name(i), bleu))
            bleus.append(bleu)

        summary = tf.Summary()
        for i, bleu in enumerate(bleus):
            summary.value.add(
                tag='{}/{}'.format(mode, get_bleu_name(1)), simple_value=bleu)
        if FLAGS.eval_ie:
            for name, value in {'precision': prec, 'recall': rec}.items():
                summary.value.add(tag='{}/{}'.format(mode, name),
                                  simple_value=value)
        summary_writer.add_summary(summary, step)
        summary_writer.flush()

        bleu = bleus[0]
        if mode == 'val':
            if bleu > best_ever_val_bleu:
                best_ever_val_bleu = bleu
                print('updated best val bleu: {}'.format(bleu))

                _save_to(ckpt_best, step)

        print('end _eval_epoch')
        return bleu


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        if restore_from:
            _restore_from_path(restore_from)
        else:
            _restore_from(dir_model)

        summary_writer = tf.summary.FileWriter(
            dir_summary, sess.graph, flush_secs=30)

        epoch = 0
        while epoch < config_train.max_epochs:
            names = {'joint', 'disc'}
            train_ops = {name: train_ops[name] for name in names}
            summary_ops = {name: summary_ops[name] for name in names}

            val_bleu = _eval_epoch(sess, summary_writer, 'val')
            test_bleu = _eval_epoch(sess, summary_writer, 'test')

            step = tf.train.global_step(sess, global_step)

            print('epoch: {} ({}), step: {}, '
                  'val BLEU: {:.2f}, test BLEU: {:.2f}'.format(
                epoch, ', '.join(names), step, val_bleu, test_bleu))

            _train_epoch(sess, summary_writer, 'train', train_ops, summary_ops, names)

            epoch += 1

            step = tf.train.global_step(sess, global_step)
            _save_to(ckpt_model, step)

        test_bleu = _eval_epoch(sess, summary_writer, 'test')
        print('epoch: {}, test BLEU: {}'.format(epoch, test_bleu))


if __name__ == '__main__':
    main()
