from utils.registry import get_trainer
import os,sys,importlib
root_path = os.path.dirname(__file__)
for file in os.listdir(root_path):
	if '.py' in file and 'base' not in file and '__init__' not in file:
		file_abspath = os.path.join(root_path, file.split('.')[0])
		rel_path = os.path.relpath(file_abspath, os.getcwd())
		importlib.import_module(rel_path.replace('/', '.'))