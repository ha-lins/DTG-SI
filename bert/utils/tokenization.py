# coding=utf-8
# Copied from google BERT repo.

# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import nltk
import sys
import collections
import unicodedata

import tensorflow as tf
from nltk.tokenize import sent_tokenize, word_tokenize

def convert_to_unicode(text):
	"""Returns the given argument as a unicode string."""
	return tf.compat.as_text(text)


def printable_text(text):
	"""Returns text encoded in a way suitable for print or `tf.logging`."""
	return tf.compat.as_str_any(text)


def load_vocab(vocab_file):
	"""Loads a vocabulary file into a dictionary."""
	vocab = collections.OrderedDict()
	index = 0
	with tf.gfile.GFile(vocab_file, "r") as reader:
		while True:
			token = tf.compat.as_text(reader.readline())
			if not token:
				break
			token = token.strip()
			vocab[token] = index
			index += 1
	return vocab


def convert_by_vocab(vocab, items):
	"""Converts a sequence of [tokens|ids] using the vocab."""
	output = []
	for item in items:
		output.append(vocab[item])
	return output


def convert_tokens_to_ids(vocab, tokens):
	return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
	return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
	"""Runs basic whitespace cleaning and splitting on a peice of text."""
	text = text.strip()
	if not text:
		return []
	tokens = text.split()
	return tokens


class FullTokenizer(object):
	"""Runs end-to-end tokenziation."""

	def __init__(self, vocab_file, do_lower_case=True):
		self.vocab = load_vocab(vocab_file)
		self.inv_vocab = {v: k for k, v in self.vocab.items()}
		self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

	def tokenize(self, text):
		split_tokens = []
		for token in self.basic_tokenizer.tokenize(text):
			split_tokens.append(token)

		return split_tokens

	def convert_tokens_to_ids(self, tokens):
		return convert_by_vocab(self.vocab, tokens)

	def convert_ids_to_tokens(self, ids):
		return convert_by_vocab(self.inv_vocab, ids)


class BasicTokenizer(object):
	"""Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

	def __init__(self, do_lower_case=False):
		"""Constructs a BasicTokenizer.

		Args:
			do_lower_case: Whether to lower case the input.
		"""
		self.do_lower_case = do_lower_case

	def tokenize(self, text):
		"""Tokenizes a piece of text."""
		text = tf.compat.as_text(text)
		return word_tokenize(text)

