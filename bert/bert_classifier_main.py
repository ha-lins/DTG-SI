# Copyright 2018 The Texar Authors. All Rights Reserved.
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
"""Example of building a sentence classifier based on pre-trained BERT
model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import importlib
import tensorflow as tf
import texar as tx

from utils import data_utils, model_utils, tokenization

# pylint: disable=invalid-name, too-many-locals, too-many-statements

flags = tf.flags

FLAGS = flags.FLAGS
flags.DEFINE_string(
	"task", "sst",
	"The task to run experiment on. One of "
	"{'cola', 'mnli', 'mrpc', 'xnli', 'sst'}.")
flags.DEFINE_string(
	"config_bert_pretrain", 'bert_config',
	"The architecture of pre-trained BERT model to use.")
flags.DEFINE_string(
	"config_format_bert", "json",
	"The configuration format. Set to 'json' if the BERT config file is in "
	"the same format of the official BERT config file. Set to 'texar' if the "
	"BERT config file is in Texar format.")
flags.DEFINE_string(
	"config_downstream", "config_classifier",
	"Configuration of the downstream part of the model and optmization.")
flags.DEFINE_string(
	"config_data", "config_data",
	"The dataset config.")
flags.DEFINE_string(
	"checkpoint", "classifier_ckpt/ckpt/model.ckpt-13625",
	"Path to a model checkpoint (including bert modules) to restore from.")
flags.DEFINE_string(
	"output_dir", "e2e_output/",
	"The output directory where the model checkpoints will be written.")
flags.DEFINE_bool(
	"do_lower_case", True,
	"Whether to lower case the input text. Should be True for uncased "
	"models and False for cased models.")
flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_test", False, "Whether to run test on the test set.")
flags.DEFINE_bool("do_pred", False, "Whether to run test on the test set.")
flags.DEFINE_bool("distributed", False, "Whether to run in distributed mode.")

config_data = importlib.import_module(FLAGS.config_data)
config_downstream = importlib.import_module(FLAGS.config_downstream)
output_dir = FLAGS.output_dir
dir_model = os.path.join(output_dir, 'ckpt')
dir_best = os.path.join(output_dir, 'ckpt-best')
ckpt_model = os.path.join(dir_model, 'model.ckpt')
ckpt_best = os.path.join(dir_best, 'model.ckpt')


def main(_):
	"""
	Builds the model and runs.
	"""

	if FLAGS.distributed:
		import horovod.tensorflow as hvd
		hvd.init()

	tf.logging.set_verbosity(tf.logging.INFO)
	tx.utils.maybe_create_dir(FLAGS.output_dir)
	bert_pretrain_dir = '%s' % FLAGS.config_bert_pretrain

	# Loads BERT model configuration
	if FLAGS.config_format_bert == "json":
		bert_config = model_utils.transform_bert_to_texar_config(
			os.path.join(bert_pretrain_dir, 'bert_config.json'))
	elif FLAGS.config_format_bert == 'texar':
		bert_config = importlib.import_module(
			'bert_config_lib.config_model_%s' % FLAGS.config_bert_pretrain)
	else:
		raise ValueError('Unknown config_format_bert.')

	num_classes = config_data.num_classes

	"""
	train_dataset = data_utils.get_dataset(
		processor, tokenizer, config_data.data_dir, config_data.max_seq_length,
		config_data.train_batch_size, mode='train', output_dir=FLAGS.output_dir,
		is_distributed=FLAGS.distributed)

	eval_dataset = data_utils.get_dataset(
		processor, tokenizer, config_data.data_dir, config_data.max_seq_length,
		config_data.eval_batch_size, mode='eval', output_dir=FLAGS.output_dir)
	test_dataset = data_utils.get_dataset(
		processor, tokenizer, config_data.data_dir, config_data.max_seq_length,
		config_data.test_batch_size, mode='test', output_dir=FLAGS.output_dir)
	"""
	train_dataset = tx.data.TFRecordData(hparams=config_data.train_hparam)
	eval_dataset = tx.data.TFRecordData(hparams=config_data.eval_hparam)
	test_dataset_1 = tx.data.TFRecordData(hparams=config_data.test_hparam_1)
	test_dataset_2 = tx.data.TFRecordData(hparams=config_data.test_hparam_2)
	iterator = tx.data.FeedableDataIterator({
		'train': train_dataset, 'eval': eval_dataset, 'test1': test_dataset_1,
		'test2': test_dataset_2})

	batch = iterator.get_next()
	input_ids = batch["input_ids"]
	batch_size = tf.shape(input_ids)[0]
	input_length = tf.reduce_sum(1 - tf.to_int32(tf.equal(input_ids, 0)),
								 axis=1)

	# Builds BERT
	with tf.variable_scope('bert'):
		embedder = tx.modules.WordEmbedder(
			vocab_size=bert_config.vocab_size,
			hparams=bert_config.embed)
		word_embeds = embedder(input_ids)

		# Creates segment embeddings for each type of tokens.
		"""segment_embedder = tx.modules.WordEmbedder(
			vocab_size=bert_config.type_vocab_size,
			hparams=bert_config.segment_embed)
		segment_embeds = segment_embedder(segment_ids)

		input_embeds = word_embeds + segment_embeds
		"""
		input_embeds = word_embeds
		# The BERT model (a TransformerEncoder)
		encoder = tx.modules.TransformerEncoder(hparams=bert_config.encoder)
		output = encoder(input_embeds, input_length)

		# Builds layers for downstream classification, which is also initialized
		# with BERT pre-trained checkpoint.
		with tf.variable_scope("pooler"):
			# Uses the projection of the 1st-step hidden vector of BERT output
			# as the representation of the sentence
			bert_sent_hidden = tf.squeeze(output[:, 0:1, :], axis=1)
			bert_sent_output = tf.layers.dense(
				bert_sent_hidden, config_downstream.hidden_dim,
				activation=tf.nn.relu)
			output = tf.layers.dropout(
				bert_sent_output, rate=0.05, training=tx.global_mode_train())

	# Adds the final classification layer
	logits = tf.layers.dense(
		output, num_classes,
		kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
	preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
	accu = tx.evals.accuracy(batch['label_ids'], preds)

	# Optimization

	loss = tf.losses.sparse_softmax_cross_entropy(
		labels=batch["label_ids"], logits=logits)
	global_step = tf.Variable(0, trainable=False)

	# Builds learning rate decay scheduler

	static_lr = config_downstream.lr['static_lr']
	#    num_train_steps = int(num_train_data / config_data.train_batch_size)
	#    num_warmup_steps = int(num_train_steps * config_data.warmup_proportion)
	#    lr = model_utils.get_lr(global_step, num_train_steps, # lr is a Tensor
	#                            num_warmup_steps, static_lr)
	lr = static_lr
	# tf.train.exponential_decay(
	# learning_rate=0.1, global_step=global_step, decay_steps=1000, decay_rate=0.9, staircase=False)

	opt = tx.core.get_optimizer(
		global_step=global_step,
		learning_rate=lr,
		hparams=config_downstream.opt
	)

	if FLAGS.distributed:
		opt = hvd.DistributedOptimizer(opt)

	train_op = tf.contrib.layers.optimize_loss(
		loss=loss,
		global_step=global_step,
		learning_rate=None,
		optimizer=opt)

	global best_dev_accu
	best_dev_accu = 0.
	# Train/eval/test routine
	def _is_head():
		if not FLAGS.distributed:
			return True
		else:
			return hvd.rank() == 0

	def _train_epoch(sess):
		"""Trains on the training set, and evaluates on the dev set
		periodically.
		"""
		iterator.restart_dataset(sess, 'train')
		cum_acc = 0.0
		cum_loss = 0.0
		nsamples = 0
		fetches = {
			'train_op': train_op,
			'accu': accu,
			'loss': loss,
			'batch_size': batch_size,
			'step': global_step
		}

		while True:
			try:
				feed_dict = {
					iterator.handle: iterator.get_handle(sess, 'train'),
					tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
				}
				rets = sess.run(fetches, feed_dict)
				step = rets['step']
				cum_acc += rets['accu'] * rets['batch_size']
				cum_loss += rets['loss'] * rets['batch_size']
				nsamples += rets['batch_size']

				dis_steps = config_data.display_steps
				if _is_head() and dis_steps > 0 and step % dis_steps == 0:
					tf.logging.info('step:%d; loss:%f' % (step, rets['loss']))

				eval_steps = config_data.eval_steps
				if _is_head() and eval_steps > 0 and step % eval_steps == 0:
					_eval_epoch(sess)

			except tf.errors.OutOfRangeError:
				break
		tf.logging.info('train accu: {0:.4f}; loss: {1:.4f}; nsamples: {2:.4f}'.format(
			cum_acc / nsamples, cum_loss / nsamples, nsamples))

	def _eval_epoch(sess):
		"""Evaluates on the dev set.
		"""
		global best_dev_accu
		iterator.restart_dataset(sess, 'eval')

		cum_acc = 0.0
		cum_loss = 0.0
		nsamples = 0
		fetches = {
			'accu': accu,
			'loss': loss,
			'batch_size': batch_size,
		}
		while True:
			try:
				feed_dict = {
					iterator.handle: iterator.get_handle(sess, 'eval'),
					tx.context.global_mode(): tf.estimator.ModeKeys.EVAL,
				}
				rets = sess.run(fetches, feed_dict)

				cum_acc += rets['accu'] * rets['batch_size']
				cum_loss += rets['loss'] * rets['batch_size']
				nsamples += rets['batch_size']

			except tf.errors.OutOfRangeError:
				break

		if (cum_acc / nsamples) > best_dev_accu:
			best_dev_accu = cum_acc / nsamples
			saver.save(sess, ckpt_best, global_step=global_step)
			print('updated best dev accu : {}'.format(best_dev_accu))
		tf.logging.info('eval accu: {0:.4f}; loss: {1:.4f}; nsamples: {2:.4f}'.format(
			cum_acc / nsamples, cum_loss / nsamples, nsamples))

	def _test_epoch(sess):
		"""Does predictions on the test set.
		"""
		iterator.restart_dataset(sess, 'test')

		_all_preds = []
		while True:
			try:
				feed_dict = {
					iterator.handle: iterator.get_handle(sess, 'test'),
					tx.context.global_mode(): tf.estimator.ModeKeys.PREDICT,
				}
				_preds = sess.run(preds, feed_dict=feed_dict)
				_all_preds.extend(_preds.tolist())
			except tf.errors.OutOfRangeError:
				break

		output_file = os.path.join(FLAGS.output_dir, "results.tsv")
		with tf.gfile.GFile(output_file, "w") as writer:
			writer.write('\n'.join(str(p) for p in _all_preds))
		print('content score is :{}'.format(sum(_all_preds) / len(_all_preds)))

	def _pred_epoch(sess, which_score):
		"""Does predictions on the test set.
		"""
		iterator.restart_dataset(sess, 'test{}'.format(which_score))

		_all_preds = []
		while True:
			try:
				feed_dict = {
					iterator.handle: iterator.get_handle(sess, 'test{}'.format(which_score)),
					tx.context.global_mode(): tf.estimator.ModeKeys.PREDICT,
				}
				_preds = sess.run(preds, feed_dict=feed_dict)
				_all_preds.extend(_preds.tolist())
			except tf.errors.OutOfRangeError:
				break

		output_file = os.path.join(FLAGS.output_dir, "rule_{}.tsv".format(which_score))
		# with tf.gfile.GFile(output_file, "w") as writer:
		# 	writer.write('\n'.join(str(p) for p in _all_preds))
		with tf.gfile.GFile(output_file, "w") as writer:
			writer.write('content score_{0} is : {1:.4f}'.format(which_score, sum(_all_preds) / len(_all_preds)))

		if which_score == 1:
			print('content score_{0} is : {1:.4f}'.format(which_score, sum(_all_preds) / len(_all_preds)))
		else:
			print('content score_{0} is : {1:.4f}'.format(which_score, 1 - sum(_all_preds) / len(_all_preds)))

	# Loads pretrained BERT model parameters
	# init_checkpoint = os.path.join(bert_pretrain_dir, 'bert_model.ckpt')
	# model_utils.init_bert_checkpoint(init_checkpoint)

	# Broadcasts global variables from rank-0 process
	if FLAGS.distributed:
		bcast = hvd.broadcast_global_variables(0)

	session_config = tf.ConfigProto()
	if FLAGS.distributed:
		session_config.gpu_options.visible_device_list = str(hvd.local_rank())

	with tf.Session(config=session_config) as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		sess.run(tf.tables_initializer())

		if FLAGS.distributed:
			bcast.run()

		# Restores trained model if specified
		saver = tf.train.Saver()
		if FLAGS.checkpoint:
			saver.restore(sess, FLAGS.checkpoint)

		iterator.initialize_dataset(sess)

		if FLAGS.do_train:
			for i in range(config_data.max_train_epoch):
				_train_epoch(sess)
				saver.save(sess, ckpt_model, global_step=global_step)

		if FLAGS.do_eval:
			_eval_epoch(sess)

		if FLAGS.do_test:
			_test_epoch(sess)

		if FLAGS.do_pred:
			_pred_epoch(sess, 1)
			_pred_epoch(sess, 2)

if __name__ == "__main__":
	tf.app.run()
