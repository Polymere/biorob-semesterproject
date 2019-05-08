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
from run_batch_controller.unfold_param import ParamMapper
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
		#fu.assert_dir(self.trial_dir,should_be_empty=True)
		self.mapper=ParamMapper()

	def run_batch(self,mode,*args,**kwargs):
		if mode == "param_fixed_values":
			"""
			python run_launcher.py 	param_fixed_values /data/prevel/params/test_range_GSOL.yaml /data/prevel/trial/worlds_folder 
			"""
			param_values_file = args[0][0]
			print(param_values_file)
			#tmp_folded_dir = os.path.join(self.trial_dir, "tmp_folded")
			nested_runs = True
			self.gens=gen_all_file(param_values_file, standalone=True, nested=nested_runs)
			gen_count=1
			for gen in self.gens:
				self.individuals=gen
				self.run_gen(str(gen_count),kwargs)
				gen_count+=1
			"""
				if nested_runs:
					self.gens = [] # nested list
					param_dirs = fu.dir_list(tmp_folded_dir, "")
					for param in param_dirs:
						self.gens.append(fu.file_list(param, file_format=".yaml"))
				gen_id = NoneDEPRECATED
			"""
		elif mode == "param_folder":
			#Single generation with all individuals in param_dir
			print("No longer supported")
			raise ValueError
			param_dir = args[0]
			self.gens = [fu.file_list(param_dir, file_format=".yaml")]
			gen_count = 1
			for gen in self.gens:
				self.individuals = gen
				gen_id = "param" + str(gen_count)
				self.run_gen(gen_id)
				gen_count += 1
		elif mode == "single_run":
			"""python run_launcher.py single_run /data/prevel/params/geyer-florin.yaml"""
			param_file = args[0]
			self.check_run(param_file)
			return
		elif mode == "pop":
			population=args[0]
			self.individuals=population.values()
			nb_gen=args[1]
			return self.run_gen(str(nb_gen),**kwargs)
			
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

	def run_ind(self,gen_id):
		if self.fold_counter == 0:
			copyfile(self.cworld, WORLD_ABSPATH)
		subprocess.run(["webots", "--mode=fast", "--batch","--minimize", WORLD_ABSPATH])

		run_suffix="_w"+str(self.world_counter)+"_f"+str(self.fold_counter+1)

		imp.import_run(SIM_OUTPUTDIR_RPATH,save_path=self.cdir, save_name="raw"+run_suffix)

		meta_file_path=os.path.join(self.cdir, "meta"+run_suffix+".yaml")
		with open(meta_file_path, 'w+') as meta:
			yaml.dump(self._get_meta_dct(gen_id=gen_id),meta)

	def _get_meta_dct(self,**kwargs):
		return {"gen_id":kwargs["gen_id"],"world":self.cworld,"ind":self.individual_counter}

	def check_run(self, param_file):
		raise NotImplemented
		"""if self.mapper.is_parameter_valid(param_file):
			if type(param_file)==dict:
		else:
			up.create_file(param_file, REFERENCE_PARAMFILE_ABSPATH, PARAMFILE_ABSPATH)
		subprocess.run(["webots", "--mode=realtime", "--batch","--fullscreen", WORLD_ABSPATH])
		"""

if __name__ == '__main__':
	"""python run_launcher.py 	/data/prevel/trial/worlds_folder param_fixed_values /data/prevel/params/test_range_GSOL.yaml """
	r=runLauncher(sys.argv[1])
	r.run_batch(sys.argv[2],sys.argv[3:])
