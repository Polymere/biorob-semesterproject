import sys
import os
import utils.file_utils as fu
import numpy as np
import yaml
"""
Mutiple methods to generate (minimal) parameter files for sensitivity analysis
The files generated MUST be unfolded (see unfold_param.py) before simulation

Methods:
	1. gen_all_file -> meta-method, syntax example in /examples/ex_sensitivity.yaml
	TODO -> create expale file

	2. gen_single :
	...
	COMPLETE
"""

def gen_all_file(all_params_file,output_dir,standalone=False,nested=False):
	if standalone:
		fu.assert_dir(output_dir,should_be_empty=True)
	all_params=yaml.load(open(all_params_file,'r'))
	file_counter=1
	for name,parameters in all_params.items():
		if nested:
			rdir=os.path.join(output_dir,name)
			fu.assert_dir(rdir,should_be_empty=True)
		else:
			rdir=output_dir
		if parameters[0]=="single":
			gen_single(parameters[1:], name, rdir)
		elif parameters[0]=="range":
			gen_range(parameters[1:], name, rdir)
		elif parameters[0]=="modrange":
			gen_modrange(parameters[1:], name, rdir)
		elif parameters[0]=="dual_modrange":
			gen_dual_modrange(parameters[1:], name, rdir)

def gen_single(values,param_name,output_dir,standalone=False):
	if standalone:
		fu.assert_dir(output_dir,should_be_empty=True)
	file_counter=1
	for value in values:
		try: 
			val=float(value)
			file_path=os.path.join(output_dir,(param_name+str(file_counter)+".yaml"))
			fu.save_dict(file_path,param_name,val)
		except ValueError:
			print("Error",value,"is not a number")
		file_counter+=1

def gen_range(values,param_name,output_dir,standalone=False):
	if standalone:
		fu.assert_dir(output_dir,should_be_empty=True)
	if len(values)!=3:
		print("Unexpected range format\t",values,"\t Should be min max number")
	min_bound=float(values[0])
	max_bound=float(values[1])
	number=int(values[2])
	file_counter=1
	for val in np.linspace(min_bound,max_bound,number):
		print(val)
		file_path=os.path.join(output_dir,(param_name+str(file_counter)+".yaml"))
		fu.save_dict(file_path,param_name,float(val))
		file_counter+=1

def _modlinespace(min_bound,max_bound,center_val,npoints):
	npoints=int(npoints)
	nlow=int(npoints/2)
	nhigh=npoints-nlow
	lin_low=np.linspace(float(min_bound),float(center_val),nlow,endpoint=False)
	lin_high=np.linspace(float(center_val),float(max_bound),nhigh)
	return np.concatenate((lin_low, lin_high))

def gen_modrange(values,param_name,output_dir,standalone=False):
	if standalone:
		fu.assert_dir(output_dir,should_be_empty=True)
	if len(values)!=4:
		print("Unexpected range format\t",values,"\t Should be min max center number")
	file_counter=1
	vals=_modlinespace(values[0], values[1],values[2], values[3])

	for val in vals:
		print(val)
		file_path=os.path.join(output_dir,(param_name+str(file_counter)+".yaml"))
		fu.save_dict(file_path,param_name,float(val))
		file_counter+=1
		
def gen_dual_modrange(values,gen_name,output_dir,standalone=False):
	if standalone:
		fu.assert_dir(output_dir,should_be_empty=True)

	p1=values[0]
	p2=values[1]
	name1=p1[0]
	name2=p2[0]
	vals1=_modlinespace(p1[1], p1[2], p1[3], p1[4])
	vals2=_modlinespace(p2[1], p2[2], p2[3], p2[4])
	params=[name1,name2]
	file_counter=1
	for val1 in vals1:
		for val2 in vals2:
			values=[val1,val2]
			print(values)
			file_path=os.path.join(output_dir,(gen_name+str(file_counter)+".yaml"))
			fu.save_dict(file_path, params, values)
			file_counter+=1

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