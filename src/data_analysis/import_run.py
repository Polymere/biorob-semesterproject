import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import re
import pandas as pd

import utils.file_utils as fu

CSV_SAVE_DIR="/data/prevel/repos/biorob-semesterproject/data"



def import_run(path,verbose=False,save_to_single=True,save_name="default_name",save_path="."):
	first_call=True
	if verbose:
		print("-------------------------------\n")
		print ("Importing files in",path,"\n")
		print("Saving as",save_name," in",save_path,"\n")
		print("-------------------------------\n")
	
	for file_path in fu.file_list(path):

		array,header=import_file(file_path,verbose=False)
		if save_to_single:
			file=os.path.basename(file_path) # just the file name
			file=os.path.splitext(file)[0] # remove extension
			fields=fu.concat_field(file, header)
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


def export_single_file(data,fields,outputname,outputdir,first_call,verbose=False):

	file_path=os.path.join(outputdir,outputname)
	file_path=file_path+".csv"
	if first_call:
		fu.assert_file_exists(file_path, should_exist=False)
		df=pd.DataFrame()
	else:
		fu.assert_file_exists(file_path, should_exist=True)
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

def parse_field(field):
	if field[-2:]=="_r":
		side="right"
		metric=field[:-2]
	elif field[-2:]=="_l":
		side="left"
		metric=field[:-2]
	#elif fi

def export_multi_index(data,fields,outputname,outputdir,first_call,verbose=False):
	file_path=os.path.join(outputdir,outputname)
	file_path=file_path+".csv"
	if first_call:
		fu.assert_file_exists(file_path, should_exist=False)
		df=pd.DataFrame()
	else:
		fu.assert_file_exists(file_path, should_exist=True)
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
		