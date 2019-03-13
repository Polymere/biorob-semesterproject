import sys
import os
import utils.file_utils as fu
import numpy as np
import yaml
"""
EXAMPLE
python generate_paramfile_range.py \
phik_off \
single \
/data/prevel/params/gen_param \
0 0.5 1

or 

python generate_paramfile_range.py \
phik_off \
range \
/data/prevel/params/gen_param \
0 1 2o
"""


def gen_file(file_path,param,value):
	result={param:value}
	 	
	with open(file_path, 'w') as outfile:
		yaml.dump(result,outfile)

if __name__ == '__main__':
	if len(sys.argv)>1:
		param_name=sys.argv[1]
		mode=sys.argv[2]
		param_dir=sys.argv[3]
		values=sys.argv[4:]
		print(values)
		file_counter=1
		fu.assert_dir(param_dir,should_be_empty=True)

		if mode=="single":
			for value in values:
				try: 
					val=float(value)
					file_path=os.path.join(param_dir,(param_name+str(file_counter)+".yaml"))
					gen_file(file_path,param_name,val)
				except ValueError:
					print("Error",value,"is not a number")
				file_counter+=1
		elif mode=="range":
			if len(values)!=3:
				print("Unexpected range format\t",values,"\t Should be min max number")
			min_bound=float(values[0])
			max_bound=float(values[1])
			number=int(values[2])
			for val in np.linspace(min_bound,max_bound,number):
				print(val)
				file_path=os.path.join(param_dir,(param_name+str(file_counter)+".yaml"))
				gen_file(file_path,param_name,float(val))
				file_counter+=1