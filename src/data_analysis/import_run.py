import numpy as np
import os
import sys
import matplotlib.pyplot as plt
#import pandas as pd


def file_list(path,recursive=False):
	is_dir=os.path.isdir(path)
	is_file=os.path.isfile(path)
	if is_dir:
		return [os.path.join(path,f) for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
	elif is_file:
		return [path] # formating to list to avoid iter errors
	else:
		print("Nothing in ", path)
		return None

def import_run(path,verbose=False):

	for file_path in file_list(path):
		print("*************************** \n")
		if verbose:
			print ("Importing file",file_path)
		file=open(file_path,"r")

		array=np.genfromtxt(file,delimiter=",",skip_header=1)
		#df=pd.read_csv(file,delim_whitespace=True)
		#print (df)
		print (array.shape)

		print("*************************** \n")





if __name__=="__main__":
	if len(sys.argv)==2:
		import_run(sys.argv[1], verbose=True)
	elif len(sys.argv)<2:
		print("Not enough args, should have at least one run path", sys.argv)
	else:
		print ("Args are ", sys.argv,"multiple run import not implemented yet")



