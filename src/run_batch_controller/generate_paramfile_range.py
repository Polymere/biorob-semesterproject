#!/usr/bin/env python
""" @package generate_paramfile_range
File generation for sensitivity analysis

Mutiple methods to generate (minimal) parameter files for sensitivity analysis
The files generated MUST be unfolded (see unfold_param.py) before simulation
"""
import sys
import os
import utils.file_utils as fu
import numpy as np
import yaml

LOG_DEBUG=1
LOG_INFO=2
LOG_WARNING=3
LOG_ERROR=4
LOG_LEVEL=LOG_WARNING

def gen_all_file(param_file,output_dir=None,standalone=False,nested=False,
	save=False):
	""" Generates all pairs {param_name:value} required by param_file 
	(see data/sensi_template,yaml for reference)

	"""
	_check_error_dir(output_dir, save,standalone)
	all_params=yaml.load(open(param_file,'r'),Loader=yaml.FullLoader)
	file_counter=1
	files=[]
	for name,parameters in all_params.items():
		if LOG_LEVEL<=LOG_DEBUG:
			print("[DEBUG] Reading:\t Name:",name,"\nParameters:",parameters)
		if nested and save:
			rdir=os.path.join(output_dir,name)
			fu.assert_dir(rdir,should_be_empty=True)
		elif save:
			rdir=output_dir
		else:
			rdir=None

		if parameters[0]=="single":
			files.append(gen_single(parameters[1:], name, output_dir=rdir,
				save=save))

		elif parameters[0]=="range":
			files.append(gen_range(parameters[1:], name, output_dir=rdir,
				save=save))

		elif parameters[0]=="modrange":
			files.append(gen_modrange(parameters[1:], name, output_dir=rdir,
				save=save))
		elif parameters[0]=="dual_modrange":
			files.append(gen_dual_modrange(parameters[1:], name, 
				output_dir=rdir,save=save))
	return files,list(all_params.keys())

def gen_single(values,param_name,output_dir=None,standalone=False,save=False):
	""" Generates a pair {param_name: value} for each single value

		Example :
		values = [0,1,2,3,4 ] val1 = 0, val2= 1 ...
	"""
	_check_error_dir(output_dir, save, standalone)
	file_counter=1
	files=[]
	for val in values:
		if save:
			print(val)
			file_path=os.path.join(output_dir,(param_name+str(file_counter)+".yaml"))
			fu.dict_from_keyvals(param_name,float(val),file_path)
		else:
			files.append({param_name:float(val)})
		file_counter+=1
	if not save:
		return files
	else:
		return None

def gen_range(values,param_name,output_dir=None,standalone=False,save=False):
	""" Generates pairs {param_name:value} according to range in values

		Example
		values = [0,10,3] min value = 0, max value = 10,number of points = 3 

	"""
	_check_error_dir(output_dir, save,standalone)
	if len(values)!=3:
		print("Unexpected range format\t",values,"\t Should be min max number")
	min_bound=float(values[0])
	max_bound=float(values[1])
	number=int(values[2])
	file_counter=1
	files=[]
	for val in np.linspace(min_bound,max_bound,number):
		if save:
			print(val)
			file_path=os.path.join(output_dir,(param_name+str(file_counter)+".yaml"))
			fu.dict_from_keyvals(param_name,float(val),file_path)
		else:
			files.append({param_name:float(val)})
		file_counter+=1
	if not save:
		return files
	else:
		return None



def gen_modrange(values,param_name,output_dir=None,standalone=False,save=False):
	""" Generates pairs {param_name:value} according to ranges in values, with 
	different step sizes

	Example :
	values = [solsol_wf, 0,4,1,10]
	 min value = 0, max value = 4, center value = 1
	 number of points = 10 -> 5 points between 0 and 1, 5 points between 1 and 4
	"""
	_check_error_dir(output_dir, save,standalone)

	if len(values)!=4:
		print("Unexpected range format\t",values,"\t Should be min max center number")
	file_counter=1
	vals=_modlinespace(values[0], values[1],values[2], values[3])
	files=[]
	for val in vals:
		if save:
			print(val)
			file_path=os.path.join(output_dir,(param_name+str(file_counter)+".yaml"))
			fu.dict_from_keyvals(param_name,float(val),file_path)
			file_counter+=1
		else:
			files.append({param_name:float(val)})
	if not save:
		return files
	else:
		return None
		
def gen_dual_modrange(values,gen_name,output_dir=None,standalone=False,save=False):
	""" Generates combinations of two pairs {param_name1:value1,
	 param_name2:value2} according to ranges in values, with different step size 

	Example:
	values = [ [solsol_wf, 0,4,1,10], [gasgas_wf, 0,4,1,10] ]
	"""
	_check_error_dir(output_dir, save,standalone)
	p1=values[0]
	p2=values[1]
	name1=p1[0]
	name2=p2[0]
	vals1=_modlinespace(p1[1], p1[2], p1[3], p1[4])
	vals2=_modlinespace(p2[1], p2[2], p2[3], p2[4])
	params=[name1,name2]
	files=[]
	file_counter=1

	for val1 in vals1:
		for val2 in vals2:
			values=[val1,val2]
			if save:
				print(values)
				file_path=os.path.join(output_dir,(gen_name+str(file_counter)+".yaml"))
				fu.dict_from_keyvals(params, values,file_path)
				file_counter+=1
			else:
				files.append(dict(zip(params,values)))
	if not save:	
		return files
	else:
		return None
def _modlinespace(min_bound,max_bound,center_val,npoints):
	""" Composite linespace
	"""
	npoints=int(npoints)
	nlow=int(npoints/2)
	nhigh=npoints-nlow
	lin_low=np.linspace(float(min_bound),float(center_val),nlow,endpoint=False)
	lin_high=np.linspace(float(center_val),float(max_bound),nhigh)
	return np.concatenate((lin_low, lin_high))

def _check_error_dir(output_dir,save,standalone):
	""" Input consistency check
	"""
	if save and output_dir is None:
		print("Specify output_dir to save parameter files")
		raise ValueError
	if standalone and save:
		fu.assert_dir(output_dir,should_be_empty=True)

if __name__ == '__main__':
	if len(sys.argv)>1:
		mode=sys.argv[1]
		output_dir=sys.argv[2]
		print("****************************************\n")
		print("Generating parameters files mode:\t",mode,"\n")
		print("Output directory: \t",output_dir,"\n")
		print("Arguments: \t",sys.argv[3:],"\n")
		print("****************************************\n")
		if mode=="single":
			param_name=sys.argv[3]
			values=sys.argv[4:]
			gen_single(values, param_name, output_dir,standalone=True)
		elif mode=="range":
			
			param_name=sys.argv[3]
			values=sys.argv[4:]
			gen_range(values, param_name, output_dir,standalone=True)
		elif mode=="file":
			gen_sourcefile=sys.argv[3:] # formated as a single element list
			gen_all_file(gen_sourcefile[0], output_dir,standalone=True)
		print("Parameter files :",fu.file_list(output_dir),"\n")
		print("have been generated !")