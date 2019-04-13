"""
	Launching a run with


	Inputs : 
		Parameter file eg. '../../../modeling/configFiles/geyer-structured-ISB.yaml' or paramter files folder
		Number of folds for each parameter set
		Worlds to test (if we optimize of different terrain/slopes)
		Run result output directory


		The result directory structure is as follows
		result_dir
			/param1
				param1_file.yaml
				/world1
					world1_file.wbt
					raw1.csv
					raw2.csv
					...
					rawNFOLD.csv
				/world2
					...
			/param2

			...

	EXAMPLE:
	python run_launcher.py /data/prevel/trial/param_folder \
	4 \
	/data/prevel/trial/worlds_folder \
	/data/prevel/trial/result 

	python run_launcher.py /data/prevel/params/completed \
	1 \
	/data/prevel/trial/worlds_folder \
	/data/prevel/trial/result 
"""
import utils.file_utils as fu
import time
import data_analysis.import_run as imp
import sys
from shutil import copy, copyfile
import os
import subprocess
from run_batch_controller.generate_paramfile_range import gen_all_file
import run_batch_controller.unfold_param as up
"""

"""
# Absolute path to human_2d directory, should change between computer !
H2D_SRC_DIR = "/data/prevel/human_2d"

ROOT_RESULT_DIR = "/data/prevel/runs"


# Path relative to H2D_SRC_DIR, only change if directory stucture is different
CONTROLLER_RPATH = "webots/controllers/GeyerReflex"
PARAMFILE_RPATH = "modeling/configFiles/Controllers/current.yaml"
# changed to symetrical
REFERENCE_PARAMFILE_RPATH = "modeling/configFiles/Controllers/geyer-reflex_sym1.yaml"
WORLD_RPATH = "webots/worlds/current.wbt"

CONTROLLER_ABSPATH = os.path.join(H2D_SRC_DIR, CONTROLLER_RPATH)
PARAMFILE_ABSPATH = os.path.join(H2D_SRC_DIR, PARAMFILE_RPATH)
REFERENCE_PARAMFILE_ABSPATH = os.path.join(H2D_SRC_DIR, REFERENCE_PARAMFILE_RPATH)
WORLD_ABSPATH = os.path.join(H2D_SRC_DIR, WORLD_RPATH)
SIM_OUTPUTDIR_RPATH = os.path.join(CONTROLLER_ABSPATH, "Raw_files")


class runLauncher:
	world_counter = 1
	individual_counter = 1
	max_folds = 1

	def __init__(self, args):
		self.time_start = time.time()
		mode = args[0]
		if len(args) == 4:
			self.trial_dir = args[3]
		elif len(args) == 3:
			self.trial_dir = os.path.join(
				ROOT_RESULT_DIR, time.strftime("%j_%H:%M"))

		if mode == "param_fixed_values":
			"""
			python run_launcher.py 	param_fixed_values /data/prevel/params/test_range_GSOL.yaml /data/prevel/trial/worlds_folder 
			"""
			param_values_file = args[1]
			tmp_folded_dir = os.path.join(self.trial_dir, "tmp_folded")
			nested_runs = True
			gen_all_file(param_values_file, tmp_folded_dir, standalone=True, nested=nested_runs)
			if nested_runs:
				self.gens = [] # nested list
				param_dirs = fu.dir_list(tmp_folded_dir, "")
				for param in param_dirs:
					self.gens.append(fu.file_list(param, file_format=".yaml"))
			gen_id = None
		elif mode == "param_folder":
			param_dir == args[1]
			self.individuals = fu.file_list(self.param_dir, file_format=".yaml")
		elif mode == "single_run":
			"""python run_launcher.py single_run /data/prevel/params/geyer-florin.yaml"""
			param_file = args[1]
			self.check_run(param_file)
			return
		fu.assert_dir(self.trial_dir)
		world_dirpath = args[2]
		self.worlds = fu.file_list(world_dirpath, file_format=".wbt")

		gen_count = 1
		for gen in self.gens:
			self.individuals = gen
			gen_id = "param" + str(gen_count)
			self.run_gen(gen_id)
			gen_count += 1

	def run_gen(self, gen_id):
		if gen_id is not None:
			self.cdir = os.path.join(self.trial_dir, gen_id)
			fu.assert_dir(self.cdir, should_be_empty=True)
			self.gen_dir = os.path.join(self.trial_dir, gen_id)

			print("\n*************\t", gen_id, "\t************* \n")
		else:
			self.gen_dir = self.trial_dir
			print("\n**************************\n")
		tot_ind = len(self.individuals)
		self.individual_counter = 1
		for ind in self.individuals:
			print("Individual:\t", self.individual_counter, "\n")
			self.world_counter = 1
			self.cdir = os.path.join(self.gen_dir, "ind" + str(self.individual_counter))

			fu.assert_dir(self.cdir, should_be_empty=True)
			folded_path = os.path.join(self.cdir, "meta.yaml")
			copy(ind, folded_path)
			unfolded_path = os.path.join(self.cdir, "parameters.yaml")

			up.create_file(ind, REFERENCE_PARAMFILE_ABSPATH,PARAMFILE_ABSPATH, copy_path=unfolded_path)
			for world in self.worlds:
				if len(self.worlds) > 1:
					print("\tWorld:\t", self.world_counter, "\n")
				self.fold_counter = 0
				for self.fold_counter in range(self.max_folds):
					self.single_run(world)
					self.fold_counter += 1
				self.world_counter += 1
			self.individual_counter += 1
		print("\n**************************")

	def single_run(self, world_file):
		if self.fold_counter == 0:
			copyfile(world_file, WORLD_ABSPATH)
		subprocess.run(["webots", "--mode=fast", "--batch","--minimize", WORLD_ABSPATH])
		run_name = "raw_w" + str(self.world_counter) + "f" + str(self.fold_counter + 1)
		imp.import_run(SIM_OUTPUTDIR_RPATH,save_path=self.cdir, save_name=run_name)

	def check_run(self, param_file):
		if up.compare_files(REFERENCE_PARAMFILE_ABSPATH, test_file=param_file, verbose=False):
			copyfile(param_file, PARAMFILE_ABSPATH)
		else:
			up.create_file(param_file, REFERENCE_PARAMFILE_ABSPATH, PARAMFILE_ABSPATH)
		subprocess.run(["webots", "--mode=realtime", "--batch","--fullscreen", WORLD_ABSPATH])


def launch_run(parameter_file=None, world_file=None):

	subprocess.run(["webots", "--mode=fast", "--batch","--minimize", WORLD_ABSPATH])
	# print("RUUUNNN")


def gen_param(param_range_file, param_dst):
	gen_all_file(param_range_file, param_dst, standalone=True)


def gen_param_and_launch(param_range_file, world_dirpath, run_dir):
	print(run_dir)
	fu.assert_dir(run_dir)
	folded_param_dir = os.path.join(run_dir, "folded_params")
	print(folded_param_dir)
	fu.assert_dir(folded_param_dir)
	gen_all_file(param_range_file, folded_param_dir, standalone=True)
	complete_param_dir = os.path.join(run_dir, "complete_params")

	complete_files(folded_param_dir,
				   REFERENCE_PARAMFILE_ABSPATH, complete_param_dir)

	result_dir = os.path.join(run_dir, "result")
	fu.assert_dir(result_dir)

	launch_param_dir(complete_param_dir, world_dirpath, result_dir)


def launch_param_dir(parameter_dirpath, world_dirpath, result_dir, nfolds=1, verbose=True):
	parameter_files = fu.file_list(parameter_dirpath, file_format='.yaml')
	world_files = fu.file_list(world_dirpath, file_format='.wbt')
	fu.assert_dir(result_dir)
	if verbose:
		print("********************************************\n")
		print("Running \t", nfolds, " folds\n")
		print("Parameter files :\n", parameter_files, "\n")
		print("Testing on worlds:\n", world_files, "\n")
		print("Outputs in :\t", result_dir, "\n")
		print("********************************************* \n \n")
	param_counter = 1

	for parameter_file in parameter_files:
		param_filepath = os.path.join(parameter_dirpath, parameter_file)
		param_dir = os.path.join(result_dir, ("param" + str(param_counter)))
		fu.assert_dir(param_dir)
		copy(param_filepath, param_dir)
		# save current param file in result/param folder
		copyfile(param_filepath, PARAMFILE_ABSPATH)
		# copy current  param file to PARAMFILE_ABSPATH (replace the file to
		# change current run parameters)
		world_counter = 1
		for world_file in world_files:
			world_filepath = os.path.join(world_dirpath, world_file)
			world_dir = os.path.join(param_dir, ("world" + str(world_counter)))
			fu.assert_dir(world_dir)
			copy(world_filepath, world_dir)
			# save current world file in result/param/world folder
			copyfile(world_filepath, WORLD_ABSPATH)
			# copy current  world file to WORLD_ABSPATH (replace the file to
			# change current run world)
			for fold in range(nfolds):
				launch_run(param_filepath, world_filepath)
				imp.import_run(	SIM_OUTPUTDIR_RPATH, save_path=world_dir,\
							 	save_name="raw" + str(fold + 1))
			world_counter = world_counter + 1
		param_counter = param_counter + 1

if __name__ == '__main__':
	runLauncher(sys.argv[1:])
	"""
	mode=sys.argv[1]

	if mode=="param_dir":
		if len(sys.argv)==6:
			parameter_dirpath=sys.argv[2]
			nfolds=int(sys.argv[3])
			world_dirpath=sys.argv[4]
			result_dir=sys.argv[5]
		elif len(sys.argv)==5:
			parameter_dirpath=sys.argv[2]
			nfolds=int(sys.argv[3])
			world_dirpath=sys.argv[4]
			result_dir=os.path.join(ROOT_RESULT_DIR,time.strftime("%j_%H:%M"),"/")

		launch_param_dir(parameter_dirpath, world_dirpath, result_dir, nfolds)
	elif mode=="param_file_range":
				python run_launcher.py \
		param_file_range \
		/data/prevel/params/test_min_max.yaml \
		/data/prevel/trial/worlds_folder \
		/data/prevel/trial/result (optional, otherwise in /data/prevel/runs/date)

		
		param_range_file=sys.argv[2]
		world_dirpath=sys.argv[3]
		if len(sys.argv)==5:
			run_dir=sys.argv[4]
		elif len(sys.argv)==4:
			run_dir=os.path.join(ROOT_RESULT_DIR,time.strftime("%j_%H:%M"))
		gen_param_and_launch(param_range_file,world_dirpath, run_dir)
	"""
