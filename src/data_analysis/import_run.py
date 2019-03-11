import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import re
import pandas as pd

CSV_SAVE_DIR="/data/prevel/repos/biorob-semesterproject/data"



def file_list(path,recursive=False):
	is_dir=os.path.isdir(path)
	is_file=os.path.isfile(path)
	if is_dir:
		print ("Importing files in dir",path)
		return [os.path.join(path,f) for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
	elif is_file:
		return [path] # formating to list to avoid iter errors
	else:
		print("Nothing in ", path)
		return None
def concat_field(file_name,header):
	concat_lst=[]
	for field in header:
		if field=="\n":
		#	print("Removing line break")
			pass
		else:
			print(file_name)
			concat_lst.append(file_name+"_"+field)
	return concat_lst
def import_run(path,verbose=False,save_to_single=True,save_name="default_name",save_path="."):
	first_call=True
	for file_path in file_list(path):

		array,header=import_file(file_path,verbose=False)
		if save_to_single:
			file=os.path.basename(file_path) # just the file name
			file=os.path.splitext(file)[0] # remove extension
			fields=concat_field(file, header)
			new_name=export_single_file(array,fields,save_name,save_path,first_call,verbose)
			first_call=False
	return

def import_file(path,verbose=False):
	if verbose:
		print ("Importing file",path)
	with open(path) as f:
		header=f.readline().split("\t")
		extracted=re.findall(r'\[(.*?) *\]', f.read())
	idx=0
	array=None

	for line in extracted:
		clean=re.sub('\[|\]', '', line).split(",")
		if array is None:
			array=np.empty([len(extracted),len(clean)], dtype=np.float)
			array[0,:]=clean
		else:
			array[idx,:]=clean
		idx=idx+1

	return array,header

def assert_file_exists(file_path,should_exist):
	"""
	Checks if a file exists. If it is the case and the file should not exist, propose to replace name (recursive)

	-> WIP, doesn't update name

	"""
	if os.path.isfile(file_path):
		if should_exist:
			return
		else:
			print("A file already exists at location",file_path)
			replace=input ("Replace file : Y/N\n")
			if replace=="N":
				new_name=input("Insert a new file name (foo.csv) to create a new file, or N to exit:\n")
				if new_name=="N":
					sys.exit()
				else:
					cdir=os.path.dirname(file_path)
					assert_file_exists(os.path.join(cdir,new_name), should_exist=False)
			elif replace=="Y":
				print("Deleting file",file_path)
				os.remove(file_path)
				open(file_path, "w")
			return
	else:
		if should_exist:
			open(file_path, "w")
			return
		else:
			return

def export_single_file(data,fields,outputname,outputdir,first_call,verbose=False):

	file_path=os.path.join(outputdir,outputname)
	file_path=file_path+".csv"
	if first_call:
		assert_file_exists(file_path, should_exist=False)
		df=pd.DataFrame()
	else:
		assert_file_exists(file_path, should_exist=True)
		df=pd.read_csv(file_path)
		
	length,width=data.shape
	if width != len(fields):
		print("Dimension mismatch", width,len(fields))
		print ("--------------------------------------------")
		print (data[:,0])
		print ("--------------------------------------------")
		print (fields)
		print ("--------------------------------------------")
		return
	for col in range(width):
		df[fields[col]]=data[:,col]
	df.to_csv(file_path,index=False)




if __name__=="__main__":
	if len(sys.argv)==2:
		data=import_run(sys.argv[1], verbose=False, save_path=CSV_SAVE_DIR)
		#length,width=data.shape
		#for field in range(width):
		#	plt.plot(data[:,field])
		#plt.show()
	elif len(sys.argv)<2:
		print("Not enough args, should have at least one run path", sys.argv)
		
	else:
		print ("Args are ", sys.argv,"multiple run import not implemented yet")
		
	





