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
import yaml
import os
import subprocess
from run_batch_controller.generate_paramfile_range import gen_all_file
import run_batch_controller.unfold_param as up
from run_batch_controller.unfold_param import PythonMapper,CppMapper
"""

"""
# Absolute path to human_2d directory, should change between computer !
H2D_SRC_DIR = "/data/prevel/human_2d"

ROOT_RESULT_DIR = "/data/prevel/runs"


# Path relative to H2D_SRC_DIR, only change if directory stucture is different
CONTROLLER_RPATH = "webots/controllers/GeyerReflex"
PARAMFILE_RPATH = "modeling/configFiles/Controllers/current.yaml"
# changed to symetrical
#REFERENCE_PARAMFILE_RPATH = "modeling/configFiles/Controllers/geyer-reflex_sym1.yaml"

WORLD_RPATH = "webots/worlds/current.wbt"

CONTROLLER_ABSPATH = os.path.join(H2D_SRC_DIR, CONTROLLER_RPATH)
PARAMFILE_ABSPATH = os.path.join(H2D_SRC_DIR, PARAMFILE_RPATH)
#REFERENCE_PARAMFILE_ABSPATH = os.path.join(H2D_SRC_DIR, REFERENCE_PARAMFILE_RPATH)
WORLD_ABSPATH = os.path.join(H2D_SRC_DIR, WORLD_RPATH)
SIM_OUTPUTDIR_RPATH = os.path.join(CONTROLLER_ABSPATH, "Raw_files")

PYTHON_LOG_PATH="/data/prevel/human_2d/webots/controllers/GeyerReflex/Raw_files" # not used,change

CPP_CONFIG_PATH="/data/prevel/repos/humanWebotsNmm/config/2D"

CPP_LOG_PATH="/data/prevel/repos/humanWebotsNmm/log/current_log"

DEFAULT_PYTHON_WORLD="/data/prevel/worlds_folder/python_worlds"

CPP_WORLD_PATH = "/data/prevel/repos/humanWebotsNmm/webots/worlds/current.wbt"


DEFAULT_CPP_WORLD="/data/prevel/worlds_folder/cpp_worlds"
class runLauncher:
	world_counter = 1
	individual_counter = 1
	max_folds = 1

	def __init__(self,worlds_dir, **kwargs):
		self.time_start = time.time()
		#mode = args[0]

		self.worlds = fu.file_list(worlds_dir, file_format=".wbt")

		if "trial_dir" in kwargs.keys():
			self.trial_dir = kwargs["trial_dir"]
		else:
			self.trial_dir = os.path.join(ROOT_RESULT_DIR, time.strftime("%j_%H:%M"))

	def run_batch(self,mode,*args,**kwargs):
		if mode == "param_fixed_values":
			param_values_file = args[0][0]
			print(param_values_file)
			self.gens=gen_all_file(param_values_file, standalone=True)
			gen_count=1
			for gen in self.gens:
				self.individuals=gen
				self.run_gen("param"+str(gen_count))
				gen_count+=1
		elif mode == "single_run":
			param_file = args[0]
			self.check_run(param_file)
			return
		elif mode == "pop":
			population=args[0]
			self.individuals=population.values()
			nb_gen=args[1]
			return self.run_gen("gen"+str(nb_gen),**kwargs)
		elif mode=="check":
			print(args[0])
			ind={args[0][0]:float(args[0][1])}
			self.check_run(ind)
			
	def run_ind(self,gen_id):
		if self.fold_counter==0:
			copyfile(self.cworld, self.world_path)

		subprocess.run(["webots", "--mode=fast", "--batch","--minimize", self.world_path])
		run_suffix="_w"+str(self.world_counter)+"_f"+str(self.fold_counter+1)
		self.import_run(self.run_import_path,save_path=self.cdir, save_name="raw"+run_suffix)
		self.dump_meta(self._get_meta_dct(gen_id=gen_id))

	def dump_meta(self,dct):
		print("Dumping",dct)
		run_suffix="_w"+str(self.world_counter)+"_f"+str(self.fold_counter+1)
		meta_file_path=os.path.join(self.cdir, "meta"+run_suffix+".yaml")
		with open(meta_file_path, 'a+') as meta:
			yaml.dump(dct,meta)
	def check_run(self,param_name,param_value):
		raise NotImplementedError
		#if self.mapper.is_parameter_valid(param_file):
			
	def _get_meta_dct(self,**kwargs):
		return {"info":{"gen_id":kwargs["gen_id"],
				"world":self.cworld,
				"ind":self.individual_counter}}


class PythonLauncher(runLauncher):
	def __init__(self,worlds=DEFAULT_PYTHON_WORLD,**kwargs):
		runLauncher.__init__(self,worlds,**kwargs)
		self.mapper=PythonMapper()
		self.world_path=PYTHON_WORLD_PATH
		self.importer=imp # change with language dep

	def run_gen(self, gen_id, **opt_args):
		tot_ind = len(self.individuals)
		if gen_id is not None:
			self.cdir = os.path.join(self.trial_dir, gen_id)
			fu.assert_dir(self.cdir, should_be_empty=True)
			self.gen_dir = os.path.join(self.trial_dir, gen_id)
			print("\n*************\t", gen_id, "\t************* \n")
		else:
			self.gen_dir = self.trial_dir
			print("\n**************************\n")
		try:
			do_split=opt_args["split"]
		except KeyError:
			do_split=True
		
		self.individual_counter = 1
		for ind in self.individuals:
			print("Individual:\t", self.individual_counter, "/",tot_ind,"\n")
			self.world_counter = 1
			self.cdir = os.path.join(self.gen_dir, "ind" + str(self.individual_counter))
			fu.assert_dir(self.cdir, should_be_empty=True)
			self.mapper.complete_and_save(ind, PARAMFILE_ABSPATH, do_split)

			folded_path = os.path.join(self.cdir, "params.yaml")
			with open(folded_path,'w+') as fold_param:
				yaml.dump(dict(ind), fold_param)
			#unfolded_path = os.path.join(self.cdir, "parameters.yaml")

			#up.create_file(ind, REFERENCE_PARAMFILE_ABSPATH,PARAMFILE_ABSPATH, copy_path=unfolded_path)
			for world in self.worlds:
				self.cworld=world
				if len(self.worlds) > 1:
					print("\tWorld:\t", self.world_counter, "\n")
				self.fold_counter = 0
				for self.fold_counter in range(self.max_folds):
					self.run_ind(gen_id)
					self.fold_counter += 1
				self.world_counter += 1
			self.individual_counter += 1
		print("\n**************************")
		return self.gen_dir

class CppLauncher(runLauncher):

	def __init__(self,worlds=DEFAULT_CPP_WORLD,**kwargs):
		runLauncher.__init__(self,worlds,**kwargs)
		self.mapper=CppMapper()
		self.import_run=imp.cpp_import_run # change with language dep
		self.world_path=CPP_WORLD_PATH
		self.param_write_path=os.path.join(CPP_CONFIG_PATH,"gaits/current")
		self.run_import_path=CPP_LOG_PATH


	def run_gen(self, gen_id, **opt_args):
		tot_ind = len(self.individuals)
		if gen_id is not None:
			self.cdir = os.path.join(self.trial_dir, gen_id)
			fu.assert_dir(self.cdir, should_be_empty=True)
			self.gen_dir = os.path.join(self.trial_dir, gen_id)
			print("\n*************\t", gen_id, "\t************* \n")
		else:
			self.gen_dir = self.trial_dir
			print("\n**************************\n")
		self.individual_counter = 1
		for ind in self.individuals:
			print("Individual:\t", self.individual_counter, "/",tot_ind,"\n")
			self.world_counter = 1
			self.cdir = os.path.join(self.gen_dir, "ind" + str(self.individual_counter))
			fu.assert_dir(self.cdir, should_be_empty=True)
			self.mapper.complete_and_save(ind, self.param_write_path)
			param_file_copypath = os.path.join(self.cdir, "full_params")
			copyfile(self.param_write_path, param_file_copypath)


			for world in self.worlds:
				self.cworld=world
				if len(self.worlds) > 1:
					print("\tWorld:\t", self.world_counter, "\n")
				self.fold_counter = 0
				for self.fold_counter in range(self.max_folds):
					self.dump_meta({"opt_params":dict(ind)})
					self.run_ind(gen_id)
					self.fold_counter += 1
				self.world_counter += 1
			self.individual_counter += 1
		print("\n**************************")
		return self.gen_dir


	def check_run(self,ind):
		self.mapper.complete_and_save(ind, self.param_write_path)
		subprocess.run(["webots", "--batch", self.world_path])
if __name__ == '__main__':
	#python run_launcher.py cpp param_fixed_values /data/prevel

	if sys.argv[1]=="cpp":
		r=CppLauncher()
	elif sys.argv[1]=="py":
		r=PythonLauncher()
	r.run_batch(sys.argv[2],sys.argv[3:])
