import os,sys
from importlib import import_module

class opts:
	def __init__(self):
		self.num_workers=4
		self.resume=False
		self.pretrained=True
		self.save_dir='./checkpoints'
		self.print_iter=2
		self.gpus=[0]
		self.batch_size=4
		self.lr=1e-4
		self.num_epochs=500

		self.master_batch_size=-1
		if self.master_batch_size == -1:
			self.master_batch_size = self.batch_size // len(self.gpus)
		rest_batch_size = (self.batch_size - self.master_batch_size)
		self.chunk_sizes = [self.master_batch_size]
		for i in range(len(self.gpus) - 1):
			slave_chunk_size = rest_batch_size // (len(self.gpus) - 1)
			if i < rest_batch_size % (len(self.gpus) - 1):
				slave_chunk_size += 1
			self.chunk_sizes.append(slave_chunk_size)

	def add_dict2classmem(self, dict_):
		assert isinstance(dict_, dict)
		for name, value in dict_.items():
			exec("self.{}='{}'".format(name, value)) if isinstance(value, str) else exec('self.{}={}'.format(name, value))

	def from_file(self, file):
		config_path = os.path.abspath(os.path.dirname(__file__))+'/configs'
		
		filename = os.path.join(config_path, os.path.basename(file))
		if filename.endswith('.py'):
			module_name = os.path.basename(filename)[:-3]
			if '.' in module_name:
				raise ValueError('Dots are not allowed in config file path.')
			config_dir = os.path.dirname(filename)
			sys.path.insert(0, config_dir)
			mod = import_module(module_name)
			sys.path.pop(0)
			for name, value in mod.__dict__.items():
				if not name.startswith('__'):
					if isinstance(value, dict):
						self.add_dict2classmem(value)
					else:
						exec("self.{}='{}'".format(name,value)) if isinstance(value, str) else exec('self.{}={}'.format(name,value))
						