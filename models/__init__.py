from utils.registry import get_model
import os,sys,importlib
net_path = os.path.join(os.path.dirname(__file__), 'networks')
for file in os.listdir(net_path):
	if '.py' in file and 'base' not in file and '__init__' not in file:
		file_abspath = os.path.join(net_path, file.split('.')[0])
		rel_path = os.path.relpath(file_abspath, os.getcwd())
		importlib.import_module(rel_path.replace('/', '.'))