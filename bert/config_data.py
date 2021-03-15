tfrecord_data_dir = "e2e_preparation"
max_seq_length = 128
num_classes = 2
num_train_data = 174449

train_batch_size = 64
max_train_epoch = 20
display_steps = 500 # Print training loss every display_steps; -1 to disable
eval_steps = 2500    # Eval on the dev set every eval_steps; -1 to disable
warmup_proportion = 0.1 # Proportion of training to perform linear learning
                        # rate warmup for. E.g., 0.1 = 10% of training.
eval_batch_size = 32
test_batch_size = 32


feature_original_types = {
    # Reading features from TFRecord data file.
    # E.g., Reading feature "input_ids" as dtype `tf.int64`;
    # "FixedLenFeature" indicates its length is fixed for all data instances;
    # and the sequence length is limited by `max_seq_length`.
    "input_ids": ["tf.int64", "FixedLenFeature", max_seq_length],
    "input_mask": ["tf.int64", "FixedLenFeature", max_seq_length],
    "segment_ids": ["tf.int64", "FixedLenFeature", max_seq_length],
    "label_ids": ["tf.int64", "FixedLenFeature"]
}

feature_convert_types = {
    # Converting feature dtype after reading. E.g.,
    # Converting the dtype of feature "input_ids" from `tf.int64` (as above)
    # to `tf.int32`
    "input_ids": "tf.int32",
    "input_mask": "tf.int32",
    "label_ids": "tf.int32",
    "segment_ids": "tf.int32"
}

train_hparam = {
    "allow_smaller_final_batch": False,
    "batch_size": train_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_convert_types": feature_convert_types,
        "feature_original_types": feature_original_types,
        "files": "{}/record/train.tf_record".format(tfrecord_data_dir)
    },
    "shuffle": True,
    "shuffle_buffer_size": 100
}

eval_hparam = {
    "allow_smaller_final_batch": True,
    "batch_size": eval_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_convert_types": feature_convert_types,
        "feature_original_types": feature_original_types,
        "files": "{}/record/eval.tf_record".format(tfrecord_data_dir)
    },
    "shuffle": False
}

test_hparam = {
    "allow_smaller_final_batch": True,
    "batch_size": test_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_convert_types": feature_convert_types,
        "feature_original_types": feature_original_types,
        "files": "{}/record/test.tf_record".format(tfrecord_data_dir)
    },

    "shuffle": False
}

test_hparam_1 = {
    "allow_smaller_final_batch": True,
    "batch_size": test_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_convert_types": feature_convert_types,
        "feature_original_types": feature_original_types,
        "files": "{}/record/e2e_s2s.1.tf_record".format(tfrecord_data_dir) # you should replace `e2e_s2s` with other methods manually
    },
    
    "shuffle": False
}

test_hparam_2 = {
    "allow_smaller_final_batch": True,
    "batch_size": test_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_convert_types": feature_convert_types,
        "feature_original_types": feature_original_types,
        "files": "{}/record/e2e_s2s.2.tf_record".format(tfrecord_data_dir) # you should replace `e2e_s2s` with other methods manually
    },

    "shuffle": False
}
