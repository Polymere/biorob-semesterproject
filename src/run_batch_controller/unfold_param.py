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


MAP_VALUE_FILE = "/data/prevel/repos/biorob-semesterproject/data/map_geyer_syme.yaml"


def unfold_map_value(folded_file):
	values = yaml.load(open(folded_file, 'r'))

	unfolded_values = {}
	map_file = yaml.load(open(MAP_VALUE_FILE, 'r'))
	for param_name, param_value in values.items():
		try:
			keys = map_file[param_name]
		except KeyError:
			print("Parameter \t", param_name,
				  "not found in \t", MAP_VALUE_FILE)
		for key in keys:
			unfolded_values[key] = param_value
		if param_name == "kp":
			for key in map_file["063kp"]:
				unfolded_values[key] = 0.63 * param_value
	return unfolded_values


def inverse_map_value(unfolded_file, result_file, rounded=False):
	values = yaml.load(open(unfolded_file, 'r'))
	map_file = map_file = yaml.load(open(MAP_VALUE_FILE, 'r'))
	folded_values = {}
	for u_name, value in values.items():
		for f_name, u_name_lst in map_file.items():
			if u_name in u_name_lst and f_name not in folded_values.keys():
				if rounded:
					folded_values[f_name] = round(value, 2)
				else:
					folded_values[f_name] = value

	with open(result_file, 'w') as outfile:
		yaml.dump(folded_values, outfile, default_flow_style=False)
	return folded_values


def compare_files(reference_file, verbose=False, test_file=None, test_dict=None):
	if test_file is None and test_dict is None:
		print("Missing arguments (test file or dict)")
		return
	elif test_file is not None:
		test_dict = yaml.load(open(test_file, 'r'))
	reference = yaml.load(open(reference_file, 'r'))
	default_values_keys = []
	missing_values_keys = []

	for reference_key in reference.keys():
		if reference_key not in test_dict.keys():
			default_values_keys.append(reference_key)
	for test_key in test_dict.keys():
		if test_key not in reference.keys():
			missing_values_keys.append(test_key)
	if verbose:
		print("\n***************************")
		for default in default_values_keys:
			print("\n Using default value (",
				  reference[default], ") for parameter", default)
		if len(missing_values_keys) > 0:
			print("\n -----------------------------")
		for missing in missing_values_keys:
			print("\n Parameter", missing,
				  "does not exist in reference file", reference_file)
		print("\n***************************")
	if len(default_values_keys) == 0 and len(missing_values_keys) == 0:
		return True
	else:
		return False  # default_values_keys,missing_values_keys


def compare_values(file1, file2, verbose=True):
	f1 = yaml.load(open(file1, 'r'))
	f2 = yaml.load(open(file2, 'r'))
	dif_val = {}
	dif_keys = []
	for key1, val1 in f1.items():
		if key1 not in f2.keys():
			dif_keys.append(key1)
		elif f2[key1] != val1:
			dif_val[key1] = (val1, f2[key1])
	for key2 in f2.keys():
		if key2 not in f1.keys():
			dif_keys.append(key2)
	if verbose:
		for key, dval in dif_val.items():
			print("\n Values for param", key,
				  "are different \t:", dval[0], dval[1])
		for param in dif_keys:
			print("\n Different parameters: \t", param)


def split_by_leg(dct):
	dct_r = {}
	dct_l = {}
	for p_name, p_val in dct.items():
		if p_name[-2:] == "_r":
			dct_r[p_name] = p_val
		elif p_name[-2:] == "_l":
			dct_l[p_name] = p_val
		else:
			print("\nParam \t", p_name, "\tfor both legs")
			dct_l[p_name] = p_val
			dct_r[p_name] = p_val
	return dct_r, dct_l
def complete_with_reference(unfolded,reference,verbose=True):
	result = copy.deepcopy(reference)
	dif_count=0
	for param_name, param_value in unfolded.items():
		try:
			dif_count+=1
			result[param_name] = param_value
		except KeyError as e:
			print(	"Parameter ", param_name, "missing from reference",\
					reference_file, ". Ignoring \n")
		except Exception as e:
			print("Unknown error ", e)
			pass
	if verbose:
		same_count=len(result)-dif_count
		print("\nSame :",same_count,"\tDiff: ",dif_count)
	return result

def create_file(folded_file, reference_file, result_file,
				split=True, verbose=False, copy_path=None):
	if verbose:
		print("Unfolding file", folded_file, "to", result_file, "\n")

	unfolded = unfold_map_value(folded_file)
	compare_files(reference_file, verbose=False, test_dict=unfolded)

	reference = yaml.load(open(reference_file, 'r'))
	result=complete_with_reference(unfolded, reference)

	if split:
		right, left = split_by_leg(result)
		right=complete_with_reference(right, reference)
		left=complete_with_reference(left, reference)
		dirp, base = os.path.split(result_file)
		sp = base.split('.')
		if len(sp) != 2:
			print("\n", sp, "\n")
			raise(ValueError)
		f_name, f_extension = sp[0], sp[1]
		right_file = os.path.join(dirp, f_name + "_r." + f_extension)
		left_file = os.path.join(dirp, f_name + "_l." + f_extension)
		with open(right_file, 'w') as outfile:
			yaml.dump(right, outfile, default_flow_style=False)
		with open(left_file, 'w') as outfile:
			yaml.dump(left, outfile, default_flow_style=False)

	with open(result_file, 'w') as outfile:
		yaml.dump(result, outfile, default_flow_style=False)
	if copy_path:
		with open(copy_path, 'w') as outfile:
			yaml.dump(result, outfile, default_flow_style=False)


def complete_files(folded_dir, reference_file, result_dir):
	fu.assert_dir(folded_dir, should_be_empty=False)
	fu.assert_dir(result_dir, should_be_empty=True)
	for folded_file in fu.file_list(folded_dir, file_format=".yaml"):
		file_name = os.path.basename(folded_file)
		result_file = os.path.join(result_dir, file_name)
		create_file(folded_file, reference_file, result_file, verbose=False)


if __name__ == '__main__':
	mode = sys.argv[1]

	if mode == "unfold_dir":
		reference_file = sys.argv[2]
		folded_path = sys.argv[3]  # can be either file or dir
		result_params_dir = sys.argv[4]
		complete_files(folded_path, reference_file, result_params_dir)

		for result_file in fu.file_list(result_params_dir, file_format=".yaml"):
			compare_files(reference_file, test_file=result_file, verbose=True)
	elif mode == "fold_file":
		"""
python unfold_param.py \
fold_file \
/data/prevel/human_2d/modeling/configFiles/Controllers/geyer-reflex1.yaml \
/data/prevel/params/geyer-reflex1_folded.yaml \
		"""
		unfolded_file = sys.argv[2]
		result_file = sys.argv[3]
		inverse_map_value(unfolded_file, result_file)
	elif mode == "compare_values":
		file1 = sys.argv[2]
		file2 = sys.argv[3]
		compare_values(file1, file2)
