#!/usr/bin/env python
""" @package unfold_param
Creates parameter files usable by the webots controller


"""
import yaml
import csv
import sys
import copy
import os
import utils.file_utils as fu


MAP_VALUE_FILE = "../../data/references/map_geyer_py_sym.yaml"
REFERENCE_FILE_PY="../../data/references/geyer_py_default.yaml"
REFERENCE_FILE_CPP="../../data/references/geyer_cpp_default.yaml"


class ParamMapper():
	""" Creates the file read by the reflex controller from parameter values
	"""

	def __init__(self,ref_file):
		self.ref_params=yaml.load(open(ref_file,'r'),Loader=yaml.FullLoader)
	

	def save_file(self,save_path,data):
		""" Saves dict as yaml"""
		with open(save_path, 'w') as outfile:
			yaml.dump(data, outfile, default_flow_style=False)

	def complete(self,unfolded_dct):
		""" Completes input parameters with reference_file

			Creates a copy of parameter - value pairs in reference file, and
			modifies the values of parameters present input  dict

			Input :
			unfolded_dct -- dict containing all the parameters to modify and
			the desired value
			Output :
			completed -- dict containing all the parameters to be used by the 
			reflex controller, with default values  according to reference file
		
		"""
		completed = copy.deepcopy(self.ref_params)
		for param_name, param_value in unfolded_dct.items():
			try:
				completed[param_name] = param_value
			except KeyError as e:
				print("Parameter ", param_name, "missing from reference",\
						self.ref_params, ". Ignoring \n")
			except Exception as e:
				print("\nRef",self.ref_params)
				print("Unknown error ", e)
				raise e
		return completed

	def is_parameter_valid(self,params):
		""" Checks if input parameters are valid (exists in reference params)
		"""
		if type(params) is dict:
			return params.keys()==self.reference.keys()
		elif fu.assert_file_exists(params):
			param_dct=yaml.load(open(params),'r')
			return param_dct.keys()==self.reference.keys()
		else:
			print("Cannot compare",params,"with reference")
			raise ValueError

class PythonMapper(ParamMapper):
	""" Extension of ParamMapper, to handle Python controller param file format
	"""
	def __init__(self):
		super(PythonMapper,self).__init__(REFERENCE_FILE_PY)
		self.map_file=yaml.load(open(MAP_VALUE_FILE, 'r'))

	def complete_and_save(self,folded_dct,save_file_path,split=True):
		""" Completes input reflex paramters with reference, and write to file

			Maps the input parameters to their corresponding names in the python
			controller reflex parameter format according to map_file, and 
			completes with ref_params.
			If split is set to True creates 3 files instead of one:
				- current_r.yaml -- Default parameters for left leg, new 
				parameters for right leg
				- current_l.yaml -- Default parameters for right leg, new 
				parameters for left leg
				- current.yaml -- New parameters for both legs

			This is used by a modified version of the python controller, where
			the parameters of the reflex model are updated only when the leg is 
			in swing phase (to avoid some instabilities introduced by the shift
			if parameter value in stance phase) 
		"""

		unf=self.unfold(folded_dct)

		completed=self.complete(unf)
		if split:
			right,left=self.split(completed)

			dirp, base = os.path.split(save_file_path)
			save_path = base.split('.')
			if len(save_path) != 2:
				print("\n", save_path, "\n")
				raise ValueError
			f_name, f_extension = save_path[0], save_path[1]
			right_file = os.path.join(dirp, f_name + "_r." + f_extension)
			left_file = os.path.join(dirp, f_name + "_l." + f_extension)
			self.save_file(right_file, right)
			self.save_file(left_file, left)
		self.save_file(save_file_path, completed)
	
	def inverse_mapping(self,unfolded_dct, rounded=False):
		""" Returns a more readable version of python controller reflex
		parameters file 
		"""
		folded_dct = {}
		for u_name, value in unfolded_dct.items():
			for f_name, u_name_lst in self.map_file.items():
				if u_name in u_name_lst and f_name not in folded_dct.keys():
					if rounded:
						folded_dct[f_name] = round(value, 2)
					else:
						folded_dct[f_name] = value
		return folded_dct
		
	def split(self,completed):
		""" Splits parameters by side (left/right) and completes with ref params
		"""
		dct_r = {}
		dct_l = {}
		for p_name, p_val in completed.items():
			if p_name[-2:] == "_r":
				dct_r[p_name] = p_val
			elif p_name[-2:] == "_l":
				dct_l[p_name] = p_val
			else:
				print("\nParam \t", p_name, "\tfor both legs")
				dct_l[p_name] = p_val
				dct_r[p_name] = p_val
		return self.complete(dct_r),self.complete(dct_l)
	def unfold(self,folded_dct):
		""" Maps values in input dict to keys in map_file

		As  there is not a one to one relation between parameters (as defined in 
		the geyer reflex model original  paper) and the parameter file read by 
		the python controller (example in data/geyer_python_default.yaml), a 
		mapping is performed ( see data/map_geyer_syme.yaml) 
		"""
		unfolded_dct = {}
		for param_name, param_value in folded_dct.items():
			try:
				keys = self.map_file[param_name]
			except KeyError:
				print("Parameter \t", param_name,"not found in \t", MAP_VALUE_FILE)
			for key in keys:
				unfolded_dct[key] = param_value
			if param_name == "kp":
				for key in map_file["063kp"]:# special case
					unfolded_dct[key] = 0.63 * param_value
		return unfolded_dct

class CppMapper(ParamMapper):
	""" Extension of ParamMapper, to handle Cpp controller param file format
	"""
	def __init__(self):
		super(CppMapper,self).__init__(REFERENCE_FILE_CPP)
		
	def save_file(self,save_path,data):
		""" Saves param dict to format readable by the cpp reflex controller
				Ex : 
				param_name1 param_value1
				param_name2 param_value2
		"""
		with open(save_path, 'w') as outfile:
			w = csv.writer(outfile,delimiter=" ")
			for key,val in data.items():
				w.writerow([key,val])
	def complete_and_save(self,folded_dct,save_file_path):
		""" Completes input reflex parameters with reference, and write to file
		"""
		complete=self.complete(folded_dct)
		self.save_file(save_file_path, complete)