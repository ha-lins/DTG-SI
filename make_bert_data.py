# coding:utf-8
import os
import numpy as np
import tensorflow as tf

flags = tf.flags
flags.DEFINE_float("which_score", "0", "score type:0--x_score, 1--x'_score")
FLAGS = flags.FLAGS
modes = ['test']  # , 'val', 'test']
refs = ['', '_ref']
fields = ['sent', 'attribute', 'entry', 'value']
data_dir = 'e2e_data_v14'
if __name__ == '__main__':
	if FLAGS.which_score: #compute whether x' exits in y
		ref = refs[1]
		with open(os.path.join(data_dir, "e2e.attribute{}.test.txt".format(
				ref)), 'r') as f_type,\
			open(os.path.join(data_dir, "e2e.entry{}.test.txt".format(ref)),
				'r') as f_entry,\
		    open(os.path.join(data_dir, "e2e.entry{}.test.txt".format(refs[
			  0])), 'r') as f_entry_x,\
			open("e2ev14_output/rule/ckpt/hypos.step0.test.txt", 'r') as \
					f_sent, \
		    open("/media/data1/linshuai/manip/examples/bert/data/E2E/rule.step0"
			   ".2.tsv", 'a') as f_w:


			lines_type = f_type.readlines()
			lines_entry = f_entry.readlines()
			lines_entry_x = f_entry_x.readlines()
			lines_sent = f_sent.readlines()

			for (idx_line, line_type) in enumerate(lines_type):
				line_type = line_type.strip('\n').split(' ')
				for (idx_val, attr) in enumerate(line_type):
					entry_list = lines_entry[idx_line].strip('\n').split(' ')
					if(lines_entry_x[idx_line].find(entry_list[idx_val]) == -1):
						pos_samp = attr + ' : ' + entry_list[idx_val] + ' | ' + lines_sent[idx_line]
						f_w.write(pos_samp)

	else:	#compute whether x exits in y
		ref = refs[0]
		with open(os.path.join(data_dir, "e2e.attribute{}.test.txt".format(
				ref)), 'r') as f_type,\
			open(os.path.join(data_dir, "e2e.entry{}.test.txt".format(\
					ref)), 'r') as f_entry,\
			open("e2ev14_output/rule/ckpt/hypos.step0.test.txt", 'r') as \
					f_sent,\
			open("/media/data1/linshuai/manip/examples/bert/data/E2E/rule.step0.1.tsv", 'a') as f_w:

			lines_type = f_type.readlines()
			lines_entry = f_entry.readlines()
			lines_sent = f_sent.readlines()

			for (idx_line, line_type) in enumerate(lines_type):
				line_type = line_type.strip('\n').split(' ')
				for (idx_val, attr) in enumerate(line_type):
					entry_list = lines_entry[idx_line].strip('\n').split(' ')
					pos_samp = attr + ' : ' + entry_list[idx_val] + ' | ' + lines_sent[idx_line]
					f_w.write(pos_samp)

