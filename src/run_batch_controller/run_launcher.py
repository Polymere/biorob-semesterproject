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

	def __init__(self,worlds_dir, **kwargs):
		self.time_start = time.time()
		#mode = args[0]

		self.worlds = fu.file_list(worlds_dir, file_format=".wbt")

		if "trial_dir" in kwargs.keys()
			self.trial_dir = kwargs["trial_dir"]
		else
			self.trial_dir = os.path.join(ROOT_RESULT_DIR, time.strftime("%j_%H:%M"))
	def run_batch(self,mode)
		if mode == "param_fixed_values":
			"""
			python run_launcher.py 	param_fixed_values /data/prevel/params/test_range_GSOL.yaml /data/prevel/trial/worlds_folder 
			"""
			param_values_file = args[0]
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
			#Single generation with all individuals in param_dir
			param_dir == args[0]
			self.gens = [fu.file_list(self.param_dir, file_format=".yaml")]
		elif mode == "single_run":
			"""python run_launcher.py single_run /data/prevel/params/geyer-florin.yaml"""
			param_file = args[0]
			self.check_run(param_file)
			return
		
		fu.assert_dir(self.trial_dir)

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
					self.run_ind(world)
					self.fold_counter += 1
				self.world_counter += 1
			self.individual_counter += 1
		print("\n**************************")

	def run_ind(self, world_file):
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


if __name__ == '__main__':
	runLauncher(sys.argv[1:])
