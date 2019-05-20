import sys
import os
import utils.file_utils as fu
from copy import deepcopy
import numpy as np
import yaml
valid_inputs=[	"ANGLE_HIP_LEFT", "ANGLE_HIPCOR_LEFT", "ANGLE_KNEE_LEFT", "ANGLE_ANKLE_LEFT",
	 			"ANGLE_HIP_RIGHT", "ANGLE_HIPCOR_RIGHT", "ANGLE_KNEE_RIGHT", 
	 			"ANGLE_ANKLE_RIGHT", "Unnamed: 8"]
NB_STEPS=1000

def _kwargs_utils(kwargs_dct,key,def_val):
	if key in kwargs_dct.keys():
		return kwargs_dct[key]
	else:
		return def_val

def get_func_arr(func):#,**kopt_func=None):
	if func=="step":
		amp=1#_kwargs_utils(kopt_func, "amp", 1)
		delay=0#_kwargs_utils(kopt_func, "delay", 0)
		arr=np.ones((NB_STEPS), dtype=np.float)*amp
		arr[0:delay]=0
		return arr
	if func=="zeros":
		return np.zeros((NB_STEPS), dtype=np.float)

def write_angles_input(inputs_dct,file_path="./angles.txt"):
	file=open(file_path, "w+")
	missing_inputs=deepcopy(valid_inputs)
	vals=[]
	headers=[]
	for joint,func in inputs_dct.items():
		if joint not in valid_inputs:
			print(joint,valid_inputs)
			raise KeyError
		headers.append(joint)
		vals.append(get_func_arr(func))
		#del(missing_inputs[joint])
	for joint in missing_inputs:
		headers.append(joint)
		vals.append(get_func_arr("zeros"))
	#print (vals)
	
	np.savetxt(file,vals,header=headers)






if __name__ == '__main__':
	dct={"ANGLE_HIP_LEFT":"step","ANGLE_HIP_RIGHT":"step"}
	write_angles_input(dct)
	