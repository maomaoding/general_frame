import sys
import fnmatch
from collections import defaultdict
from string import Template

def _natural_key(string_):
		return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]

code_template = \
'''
_module_to_${arg}s = defaultdict(set)
_${arg}_entrypoints = {}

def register_${arg}(fn):
	# lookup containing module
	mod = sys.modules[fn.__module__]
	module_name_split = fn.__module__.split('.')
	module_name = module_name_split[-1] if len(module_name_split) else ''

	# add fn to __all__ in module
	${arg}_name = fn.__name__
	if hasattr(mod, '__all__'):
		mod.__all__.append(${arg}_name)
	else:
		mod.__all__ = [${arg}_name]

	# add entries to registry dict/sets
	_${arg}_entrypoints[${arg}_name] = fn
	_module_to_${arg}s[module_name].add(${arg}_name)

	return fn

def ${arg}_entrypoint(${arg}_name):
	"""Fetch a model entrypoint for specified model name
	"""
	return _${arg}_entrypoints[${arg}_name]

def list_${arg}s(filter='', module='', exclude_filters=''):
	""" Return list of available ${arg} names, sorted alphabetically

		Args:
				filter (str) - Wildcard filter string that works with fnmatch
				module (str) - Limit ${arg} selection to a specific sub-module
				exclude_filters (str or list[str]) - Wildcard filters to exclude ${arg}s after including them with filter

		Example:
				model_list('gluon_resnet*') -- returns all models starting with 'gluon_resnet'
				model_list('*resnext*, 'resnet') -- returns all models with 'resnext' in 'resnet' module
	"""
	if module:
		${arg}s = list(_module_to_${arg}s[module])
	else:
		${arg}s = _${arg}_entrypoints.keys()
	if filter:
		${arg}s = fnmatch.filter(${arg}s, filter)	# include these ${arg}s
	if exclude_filters:
		if not isinstance(exclude_filters, list):
			exclude_filters = [exclude_filters]
		for xf in exclude_filters:
			exclude_${arg}s = fnmatch.filter(${arg}s, xf)	# exclude these ${arg}s
			if len(exclude_${arg}s):
				${arg}s = set(${arg}s).difference(exclude_${arg}s)
	return list(sorted(${arg}s, key=_natural_key))

def list_modules():
	"""Return list of module names that contain ${arg}s / ${arg} entrypoints
	"""
	modules = _module_to_${arg}s.keys()
	return list(sorted(modules))

def get_${arg}(opt):
	return ${arg}_entrypoint('_'.join([opt.task, '${arg}']))(opt)
'''

exec(Template(code_template).substitute(arg='trainer'))
exec(Template(code_template).substitute(arg='recognizor'))
exec(Template(code_template).substitute(arg='model'))