import sys
import os
import utils.file_utils as fu
import numpy as np
import yaml
"""
EXAMPLE
python generate_paramfile_range.py \
single \
/data/prevel/params/gen_param \
phik_off \
0 0.5 1

or 

python generate_paramfile_range.py \
range \
/data/prevel/params/gen_param \
phik_off \
0 1 2

or 

python generate_paramfile_range.py \
file \
/data/prevel/params/gen_param_1/ \
/data/prevel/params/test_min_max.yaml 
"""


def gen_file(file_path,param,value):
	result={param:value}
	with open(file_path, 'w') as outfile:
		yaml.dump(result,outfile)

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

def gen_single(values,param_name,output_dir,standalone=False):
	if standalone:
		fu.assert_dir(output_dir,should_be_empty=True)
	file_counter=1
	for value in values:
		try: 
			val=float(value)
			file_path=os.path.join(output_dir,(param_name+str(file_counter)+".yaml"))
			gen_file(file_path,param_name,val)
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
		gen_file(file_path,param_name,float(val))
		file_counter+=1

def gen_modrange(values,param_name,output_dir,standalone=False):
	if standalone:
		fu.assert_dir(output_dir,should_be_empty=True)
	if len(values)!=4:
		print("Unexpected range format\t",values,"\t Should be min max center number")
	min_bound=float(values[0])
	max_bound=float(values[1])
	center_val=float(values[2])
	number=int(values[3])
	nlow=int(number/2)
	nhigh=number-nlow

	file_counter=1

	for val in np.linspace(min_bound,center_val,nlow,endpoint=False):
		print(val)
		file_path=os.path.join(output_dir,(param_name+str(file_counter)+".yaml"))
		gen_file(file_path,param_name,float(val))
		file_counter+=1
	for val in np.linspace(center_val,max_bound,nhigh):
		print(val)
		file_path=os.path.join(output_dir,(param_name+str(file_counter)+".yaml"))
		gen_file(file_path,param_name,float(val))
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