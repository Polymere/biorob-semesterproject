import os
import sys
from shutil import rmtree
from functools import reduce

def assert_one_dim(lst,critical=False,verbose=True):
	print(len(lst))
	if len(lst)>1:
		if not critical and verbose: 
			if verbose:
				print("Multiple elements in,",lst,"taking only first one \n")
			return lst[0]
		elif critical:
			print("Error ! list",lst,"has multiple elements. Returning None")
			return None
	else:
		return lst[0]
	
def file_list(path,recursive=False,file_format="any_file",pattern="",verbose=True):
	"""
	Returns a list with absolute path to allfiles with 
	extension file_format in directory path.
	If path is not a directory (single file), returns the path to this file
	UPDT

	TODO -> Change to return False is resulting list is empty
	"""
	if verbose:
		print("Listing files in",path)
		print(os.listdir(path))
	path_lst=[]
	if file_format=="any_file":
		is_file=os.path.isfile(path)
		is_dir=os.path.isdir(path)
	else:
		if recursive:
			for root,dirs,files in os.walk(path):
				for file in files: 
					f=os.path.join(root,file)
					#print("\n File \t",f)
					if f.endswith(file_format):
						print("Root \t",root,"files\t",f,"\n")
						path_lst.append(f)

		else:
			path_lst=[os.path.join(path,f) for f in os.listdir(path) \
						if (f.endswith(file_format)) 
						and (os.path.isfile(os.path.join(path,f))) \
						and pattern in f]
		if len(path_lst)==1:
			is_file=True
			path_lst=path_lst[0] # unfold path (result from filter was a list)
			is_dir=False
		elif len(path_lst)==0:
			print("No file with extension",file_format,"in",path_lst)
			is_dir=False
			is_file=False
		else:
			is_dir=True
			is_file=False
	if is_dir:
		if file_format=="any_file":
			if recursive:

				for root,dirs,files in os.walk(path):
					for file in files: 
						f=os.path.join(root,file)
						path_lst.append(f)
				return path_lst
			else:
				return [os.path.join(path,f) for f in os.listdir(path) \
							if os.path.isfile(os.path.join(path,f))]
		else: # already listed
			return path_lst
	elif is_file:
		return [path_lst]  # formating to list to avoid iter errors
	else:
		print("Nothing in ", path)
		return None
def dir_list(path,pattern,recursive=False):
	if recursive:
		dir_lst=[]
		for root,dirs,files in os.walk(path):
			for cdir in dirs: 
				if pattern in cdir:
					f=os.path.join(root,cdir)
					dir_lst.append(f)
	else:
		dir_lst=[os.path.join(path,d) for d in os.listdir(path) \
				if os.path.isdir(os.path.join(path,d)) and pattern in d]
	return dir_lst
def assert_dir(dir_path,should_be_empty=True):
	"""
	Checks if dir_path leads to an existing directory. 
	If the directory already exist and should be empty, waits for input to clear it
	If the path syntax correspond to a file, extract the dir path and check if it exists (recursive)
	Otherwise create the directory (and parents if nested)
	"""
	#print("Asserting directory",dir_path)
	#aaa=input("Proceed Y/N")
	#if aaa!="Y":
	#	return

	if os.path.exists(dir_path):
		if len(file_list(dir_path))>0 and should_be_empty:
			msg="Directory:\t "+str(dir_path)+" \t is not empty. Remove files? Y/N \n"
			remove=input(msg)
			if remove=="Y":
				rmtree(dir_path)
				os.makedirs(dir_path)
			else:
				return
	else:
		print("Creating directory",dir_path)
		os.makedirs(dir_path) 
		return
		


def assert_file_exists(file_path,should_exist):
	"""
	Checks if a file exists. If it is the case and the file should not exist, 
	propose to replace name (recursive)

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
			print("Error, file at",file_path,"should exist!!")
			return
		else:
			open(file_path, "w")
			return
def concat_field(file_name,suffix):
	concat_lst=[]
	for field in suffix:

		if field!="\n":
			field=field.replace('\n','')
			#print(file_name)
			concat_lst.append(file_name+"_"+field)
			
	return concat_lst