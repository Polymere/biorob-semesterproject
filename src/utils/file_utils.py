import os
import sys
from shutil import rmtree

def file_list(path,recursive=False,file_format="any_file"):
	print("Listing files in",path)
	print(os.listdir(path))
	if file_format=="any_file":
		is_file=os.path.isfile(path)
		is_dir=os.path.isdir(path)
	else:
		path=[f for f in os.listdir(path) if f.endswith(file_format)]
		if len(path)==1:
			is_file=True
			path=path[0] # unfold path (result from filter was a list)
			is_dir=False
		elif len(path)==0:
			print("No file with extension",file_format,"in",path)
			is_dir=False
		else:
			is_dir=True
	if is_dir:
		if file_format=="any_file":
			return [os.path.join(path,f) for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
		else: # already listed
			return path
	elif is_file:
		return [path] # formating to list to avoid iter errors
	else:
		print("Nothing in ", path)
		return None
def assert_dir(dir_path,should_be_empty=True):
	if os.path.isdir(dir_path):
		if file_list(dir_path)is not None and should_be_empty:
			msg="Directory:"+str(dir_path)+"is not empty. Remove files? Y/N \n"
			remove=input(msg)
			if remove=="Y":
				rmtree(dir_path)
		return
	else:
		print("Creating directory",dir_path)
		os.makedirs(dir_path) 
		return
		


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
			print("Error, file at",file_path,"should exist!!")
			return
		else:
			open(file_path, "w")
			return
def concat_field(file_name,suffix):
	concat_lst=[]
	for field in suffix:
		if field=="\n":
		#	print("Removing line break")
			pass
		else:
			print(file_name)
			concat_lst.append(file_name+"_"+field)
	return concat_lst