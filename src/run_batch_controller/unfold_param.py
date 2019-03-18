"""
EXAMPLE
python unfold_param.py \
/data/prevel/params/geyer-structured-ISB_copy.yaml \
/data/prevel/params/gen_param \
/data/prevel/params/completed

"""

import yaml
import sys
import copy
import os
import utils.file_utils as fu



MAP_VALUE_FILE="/data/prevel/repos/biorob-semesterproject/data/map_geyer_syme.yaml"


def complete_map_value(folded_file):
	values=yaml.load(open(folded_file,'r'))

	unfolded_values={}
	map_file=yaml.load(open(MAP_VALUE_FILE,'r'))
	for param_name,param_value in values.items():
		try:
			keys=map_file[param_name]
		except KeyError:
			print("Parameter \t",param_name,"not found in \t",MAP_VALUE_FILE)
		for key in keys:
			unfolded_values[key]=param_value
		if param_name=="kp":
			for key in map_file["063kp"]:
				unfolded_values[key]=0.63*param_value
	return unfolded_values




def compare_files(reference_file,verbose=False,test_file=None,test_dict=None):
	if test_file is None and test_dict is None:
		print("Missing arguments (test file or dict)")
		return
	elif test_file is not None:
		test_dict=yaml.load(open(test_file,'r'))
	reference=yaml.load(open(reference_file,'r'))
	default_values_keys=[]
	missing_values_keys=[]

	for reference_key in reference.keys():
		if reference_key not in test_dict.keys():
			default_values_keys.append(reference_key)
	for test_key in test_dict.keys():
		if test_key not in reference.keys():
			missing_values_keys.append(test_key)
	if verbose:
		print ("\n***************************")
		for default in default_values_keys:
			print ("\n Using default value (",reference[default],") for parameter",default)
		if len(missing_values_keys)>0:
			print("\n -----------------------------")
		for missing in missing_values_keys:
			print ("\n Parameter",missing,"does not exist in reference file",reference_file)
		print ("\n***************************")
	return default_values_keys,missing_values_keys

def create_file(folded_file,reference_file,result_file,verbose=False):
	if verbose:
		print("Unfolding file",folded_file,"to",result_file,"\n")

	unfolded=complete_map_value(folded_file)
	compare_files(reference_file,verbose=False,test_dict=unfolded)

	reference=yaml.load(open(reference_file,'r'))
	result=copy.deepcopy(reference)
	for param_name,param_value in unfolded.items():
		try:
			result[param_name]=param_value
		except KeyError as e:
			print("Parameter ",param_name,"missing from reference",reference_file,". Ignoring \n")
		except Exception as e:
			print("Unknown error ",e)
			pass
	with open(result_file, 'w') as outfile:
		yaml.dump(result,outfile,default_flow_style=False)

def complete_files(folded_dir,reference_file,result_dir):
	fu.assert_dir(folded_dir,should_be_empty=False)
	fu.assert_dir(result_dir,should_be_empty=True)
	for folded_file in fu.file_list(folded_dir,file_format=".yaml"):
		file_name=os.path.basename(folded_file)
		result_file=os.path.join(result_dir,file_name)
		create_file(folded_file,reference_file,result_file,verbose=False)
		
if __name__ == '__main__':
	if len(sys.argv)==4:
		reference_file=sys.argv[1]
		folded_path=sys.argv[2] # can be either file or dir
		result_params_dir=sys.argv[3]
		complete_files(folded_path, reference_file, result_params_dir)

		for result_file in fu.file_list(result_params_dir,file_format=".yaml"):
			compare_files(reference_file,test_file=result_file,verbose=True)

