import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import re
import pandas as pd
from shutil import copyfile

import utils.file_utils as fu

CSV_SAVE_DIR="/data/prevel/repos/biorob-semesterproject/data"

DEFAULT_PATH="/data/prevel/human_2d/webots/controllers/GeyerReflex/Raw_files"

IGNORE_FILES=["f_ce.txt","f_se.txt","l_ce.txt","stim.txt","v_ce.txt"]

def import_run(path,verbose=False,save_to_single=True,save_name="default_name",save_path="."):
	first_call=True
	if verbose:
		print("-------------------------------\n")
		print ("Importing files in",path,"\n")
		print("Saving as",save_name," in",save_path,"\n")
		print("-------------------------------\n")
	
	for file_path in fu.file_list(path,file_format=".txt",verbose=False):
		
		array,header=import_file(file_path,verbose=verbose)
		if save_to_single and (os.path.basename(file_path) not in IGNORE_FILES):
			file=os.path.basename(file_path) # just the file name
			file=os.path.splitext(file)[0] # remove extension
			fields=fu.concat_field(file, header)
			new_name=export_single_file(array,fields,save_name,save_path,first_call,verbose)
			first_call=False
	for file_path in fu.file_list(path,file_format=".csv",verbose=False):
		file_name=os.path.basename(file_path)
		save_dst=os.path.join(save_path,file_name)
		copyfile(file_path, save_dst)
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
	mode=sys.argv[1]


	if mode=="import":
		"""
		python import_run.py /data/prevel/Raw_files_reference /data/prevel/runs/078_17:26/result/reference
		
		"""
		try:
			path=sys.argv[2]
		except:
			path=DEFAULT_PATH
		try:
			save_path=sys.argv[3]
		except:
			save_path=CSV_SAVE_DIR

		import_run(path, verbose=True, save_path=save_path)

	elif len(sys.argv)<2:
		print("Not enough args, should have at least one run path", sys.argv)
		
	else:
		print ("Args are ", sys.argv,"multiple run import not implemented yet")
