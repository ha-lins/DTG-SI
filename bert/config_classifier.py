hidden_dim = 384  # 768
"""
opt = {
    
}
"""
opt = {
	'optimizer': {
		'type': 'AdamWeightDecayOptimizer',
		'kwargs': {
			'weight_decay_rate': 0.01,
			'beta_1': 0.9,
			'beta_2': 0.999,
			'epsilon': 1e-6,
			'exclude_from_weight_decay': ['LayerNorm', 'layer_norm', 'bias'],
			'learning_rate': 2e-4,

		},
	},
	'gradient_clip': {
		'type': 'clip_by_global_norm',
		'kwargs': {
			'clip_norm': 15,
		},
	},
}
"""
	'learning_rate_decay': {
		'type': '',
		'kwargs': {
			'decay_rate': 0.9,
			'decay_steps': 10000,
		},
	},"""


# By default, we use warmup and linear decay for learinng rate
lr = {
	'static_lr': 1e-4,  # 2e-5
}
