#!/usr/bin/env python
""" @package run_launcher
Launching interface with webots

"""
import utils.file_utils as fu
import time
import pandas as pd

import sys
from shutil import copy, copyfile
import yaml
import os
import subprocess
from run_batch_controller.generate_paramfile_range import gen_all_file

import run_batch_controller.unfold_param as up
from run_batch_controller.unfold_param import PythonMapper,CppMapper

from data_analysis.process_run import CppRunProcess, PythonRunProcess

# Absolute path to human_2d directory (Python)
H2D_PATH = "/data/prevel/human_2d"
# Absolute path to humanWebotsNmm repository (Cpp)
HUMAN_WEBOT_NMM_PATH="../../../humanWebotsNmm/"
# Folder where the run folders will be created w
ROOT_RESULT_DIR = "./trash"

LOG_DEBUG=1
LOG_INFO=2
LOG_WARNING=3
LOG_ERROR=4
LOG_LEVEL=LOG_INFO

class runLauncher:
	""" Launching interface with webots

		
	"""
	world_counter = 1
	fold_counter=0
	individual_counter = 1
	max_folds = 1
	trial_dir=None
	nb_ind=1

	on_cluster=False
	nb_eval=10


	def __init__(self,args):
		if args is not None:
			for arg_name,arg_value in args.items():
				if hasattr(self, arg_name):
					setattr(self, arg_name, arg_value)
			if LOG_LEVEL<=LOG_INFO:
				print("\n[INFO]Launcher ",self.__class__.__name__," initialized with\n",self.__dict__)

		if self.trial_dir is None:
			self.trial_dir = os.path.join(ROOT_RESULT_DIR, time.strftime("%j_%H:%M"))
			fu.assert_dir(self.trial_dir,should_be_empty=True)
		try:
			self.worlds = fu.file_list(self.worlds_dir, file_format=".wbt")
		except FileNotFoundError:
			if LOG_LEVEL<=LOG_WARNING:
				print("\n[WARNING]No world file in\n",worlds_dir,"\nignore if optimization")
		except AttributeError:
			if LOG_LEVEL<=LOG_WARNING:
				print("\n[WARNING]No world directory,ignore if optimization")
		self.generate_paths()

	def run_batch(self,mode,*args):
		""" Implementation of different run launching modes

			 - parallel_sensitivity_analysis :
				Generates and runs parameters set to perform a sensitivity analysis
				(see reference file). Raw data generted by each run is saved for 
				further processing
				Input : 
				parameter file, see sensi_template.yaml 
				Output : 
					Saves raw data, with one directory per line on the parameter
					file for further processing
				Dependencies:
					gen_all_file in generate_paramfile_range.py

			 - check:
				Parses the arguments to create a single individual, and runs
				webots in full screen, displaying the processed information
				Input :
					list formated as 
					[ [ param1, value1, param2, value2] ], see 
					main for reference

			- worlds:
				Runs all available worlds (in self.world_path), and returns
				the logs directory. 
				Input:
					None, but we must take care to update individuals parameters
					(with create_pop) before calling this method
				Output:
					List containing logs directories for further processing
					(see get_fitness_from_dir in data_analysis/process_run.py)
		"""
		if mode == "parallel_sensitivity_analysis":

			
			param_values_file = args[0][0]
			if LOG_LEVEL<=LOG_INFO:
				print("\n[INFO]Parameters taken from\n",param_values_file)
			all_params,gen_ids=gen_all_file(param_values_file, standalone=True)
			gen_count=1

			for gen_vals, gen_id in zip(all_params,gen_ids):
				save_dir = os.path.join(self.trial_dir, gen_id)
				fu.assert_dir(save_dir, should_be_empty=True)
				population={}
				for ind, cnt in zip(gen_vals,range(len(gen_vals))):
					uid="param"+str(gen_count)+"val"+str(cnt)
					population[uid]=ind
				self.create_pop(population)
				self.run_worlds(self.world_path,self.log_path)
				runs=self.proc_run.import_runs_from_dir(self.log_path,save=False)
				for run, cnt in zip(runs,range(len(runs))):
					run.to_csv(os.path.join(save_dir,"raw"+str(cnt)+".csv"))
				gen_count+=1

		elif mode=="check":
			ind={}
			for par_idx in range(0,len(args[0]),2):
				ind[args[0][par_idx]]=float(args[0][par_idx+1])
			if LOG_LEVEL<=LOG_INFO:
				print("\n[INFO]Checking individual\n",ind)
			self.check_run(ind)

		elif mode=="worlds":
			self.run_worlds(self.world_path,self.log_path)
			return self.log_path
		else:
			if LOG_LEVEL<=LOG_ERROR:
				print("[ERROR]Incorrect input",mode)
			raise KeyError
	def run_worlds(self,worlds,logs):
		""" Run parallel evaluation of webots worlds
				Creates slices of size nb_eval (10 by default, depending on # of
				cpu cores) and runs parallel instances of webots 
				(see launchAllWebots.sh scripts)
		"""
		nb_ind=len(worlds)
		tstart=time.time()
		part_worlds=[worlds[sl:sl+self.nb_eval] for sl in range(0,nb_ind,self.nb_eval)]
		wd=os.path.dirname(os.path.realpath(__file__))
		for slice_world in part_worlds:
			if LOG_LEVEL<=LOG_DEBUG:
				print("\n[DEBUG]Running slice\n",slice_world)
			if self.on_cluster:
				subprocess.run([os.path.join(wd,"launchAllWebots_cluster.sh")] + slice_world)
				# different webots path on the cluster
			else:
				subprocess.run([os.path.join(wd,"launchAllWebots.sh")] + slice_world)
		if LOG_LEVEL<=LOG_INFO:
			print("\n[INFO]Time elapsed (",len(worlds),"evals):\t",time.time()-tstart,"s\n")



		
	def create_pop(self,population):
		"""Writes parameter of individuals in population in param_paths

			For each individual in population, writes the parameter file used 
			for simulation (.yaml) by completing the parameters of the individual
			with the default parameters specified by the mapping method
			(CppMapper or PythonMapper)
			A meta_data file containing the uid of the individual is saved
			in the log directory that will be used by this simulation

			Inputs:
			population -- Dataframe with the individuals as rows and reflex
			parameters as columns, indexed by uid

			Dependencies:
			CppMapper/PythonMapper in unfold_param.py

			Notes : the population, param_paths and log_paths lenght should be
			the same (for parallel evaluation, 1 individual = 1 world = 1 path)
		"""
		print(self.param_path)

		if len(population)!=len(self.param_path):
			if LOG_LEVEL<=LOG_ERROR:
				print("\n[ERROR]Should have has much valid param_paths as ind:\n\tpop",
					len(population),"params",len(self.param_path))

			raise ValueError
		for ind, ind_uid, param_path, log_path in zip(population.values(), 
								population.keys(),self.param_path,self.log_path):
			self.cdir=log_path
			self.dump_meta({"opt_params":ind,
				"uid":ind_uid})
			self.mapper.complete_and_save(ind, param_path)
			if LOG_LEVEL<=LOG_DEBUG:
				print("\n[DEBUG]Created param at",param_path,"for ind",ind_uid,
					"with params\n",ind)



	def dump_meta(self,dct):
		run_suffix="_w"+str(self.world_counter)+"_f"+str(self.fold_counter+1)               
		meta_file_path=os.path.join(self.cdir, "meta"+run_suffix+".yaml")
		if LOG_LEVEL<=LOG_DEBUG:
			print("\n[DEBUG]Dumping\n",dct,"at\n",meta_file_path)
		with open(meta_file_path, 'a+') as meta:
			yaml.dump(dct,meta)

class PythonLauncher(runLauncher):
	def __init__(self,args):
		super(PythonLauncher, self).__init__(args)
		self.mapper=PythonMapper()

	def generate_paths(self):
		"""
			Python implementation not suitable for this method at the time of
			this project, see generate_paths in CppLauncher for reference
		"""
		param_paths=[H2D_PATH+"modeling/configFiles/Controllers/current"+str(i)
			+".yaml" for i in range(self.nb_ind)]
		log_paths=[H2D_PATH+"webots/controllers/GeyerReflex/Raw_files"+str(i)
					for i in range(self.nb_ind)]
		worlds_paths=[H2D_PATH+"webots/worlds/current"+str(i)+
					".wbt" for i in range(self.nb_ind)]
		for p_path,l_path,w_path in zip(param_paths, log_paths,worlds_paths):
			fu.assert_dir(l_path)
			s1=fu.assert_file_exists(p_path, should_exist=True)
			s2=fu.assert_file_exists(w_path, should_exist=True)
			if not s1 or not s2:
				raise RuntimeError("Missing worlds or parameter path, see Readme")
		self.param_path=param_paths
		self.log_path=log_paths
		self.world_path=worlds_paths
		raise NotImplementedError

	def check_run(self,ind):
		"""Single run with webots display + generated graphs using inline parameters

			Writes a single parameter file completing input parameters with reference
			and runs in full screen / real time
			Process the runs according the parameters specified for PythonRunProcess,
			and saves the output (generated logs + processed metrics) in a .csv file

			Inputs:
			ind -- dict with {param_name1:param_value1,...}
			Output:
			Saved .csv file in the current directory
			Dependencies:
			PythonRunProcess in data_analysis/process_run.py
			PythonMapper in unfold_param.py
		"""
		param_path=fu.assert_one_dim(self.param_path)
		world_path=fu.assert_one_dim(self.world_path)
		log_path=fu.assert_one_dim(self.log_path)


		self.mapper.complete_and_save(ind, param_path)
		proc=PythonRunProcess({"kinematics_compare_kind" :"winter_to_python",
					"kinematics_compare_file" : "../../data/winter_data/data_normal.csv",
					"split_how": "strike_to_strike",
					"do_plot": True})
		if LOG_LEVEL<=LOG_INFO:
			print("\n[INFO]Writing \n",ind,"\tto\n",param_path)
			print("\n[INFO]Running \n",world_path)
			
		subprocess.run(["webots", "--batch","--mode=realtime","--fullscreen", world_path])

		df=proc.import_raw(log_path)
		proc.ref.get_corr(df)
		df.to_csv("last_run.csv")

class CppLauncher(runLauncher):
	def __init__(self,args):
		#runLauncher.__init__(self,args)
		super(CppLauncher, self).__init__(args)
		self.mapper=CppMapper()
		self.proc_run=CppRunProcess(None)

	def generate_paths(self):
		param_paths=[HUMAN_WEBOT_NMM_PATH+"config/2D_ind"+str(i)+
					"/gaits/current" for i in range(1,self.nb_ind+1)]
		log_paths=[HUMAN_WEBOT_NMM_PATH+"log/log_ind"+str(i)
					for i in range(1,self.nb_ind+1)]
		worlds_paths=[HUMAN_WEBOT_NMM_PATH+"webots/worlds/tmp_2D_noObstacle_GA_2Dind"+str(i)+
					".wbt" for i in range(1,self.nb_ind+1)]
		for p_path,l_path,w_path in zip(param_paths, log_paths,worlds_paths):
			fu.assert_dir(l_path)
			s1=fu.assert_file_exists(p_path, should_exist=True)
			s2=fu.assert_file_exists(w_path, should_exist=True)
			if not s1 or not s2:
				print(s1,p_path)
				print(s2,w_path)
				raise RuntimeError("Missing worlds or parameter path, see Readme webots files generation")
		self.param_path=param_paths
		self.log_path=log_paths
		self.world_path=worlds_paths


	def check_run(self,ind):
		"""Single run with webots display + generated graphs using inline parameters

			Writes a single parameter file completing input parameters with reference
			and runs in full screen / real time
			Process the runs according the parameters specified for CppRunProcess,
			and saves the output (generated logs + processed metrics) in a .csv file

			Inputs:
			ind -- dict with {param_name1:param_value1,...}
			Output:
			Saved .csv file in the current directory
			Dependencies:
			- CppRunProcess
			- CppMapper
		"""
		param_path=fu.assert_one_dim(self.param_path)
		world_path=fu.assert_one_dim(self.world_path)
		log_path=fu.assert_one_dim(self.log_path)


		self.mapper.complete_and_save(ind, param_path)
		proc=CppRunProcess({"kinematics_compare_kind" :"winter_to_cpp",
					"kinematics_compare_file" : "../../data/winter_data/data_normal.csv",
					"split_how": "strike_to_strike",
					"do_plot": True})
		if LOG_LEVEL<=LOG_INFO:
			print("\n[INFO]Writing \n",ind,"\tto\n",param_path)
			print("\n[INFO]Running \n",world_path)
			
		subprocess.run(["webots", "--batch","--mode=realtime","--fullscreen", world_path])

		df=proc.import_raw(log_path)
		proc.ref.get_corr(df)
		df.to_csv("last_run.csv")

def launch_run(args):
	"""

	"""
	if args[0]=="cpp":
		LOG_LEVEL=LOG_INFO
		r=CppLauncher(None)
	elif args[0]=="py":
		r=PythonLauncher(None)
	else:
		if LOG_LEVEL<=LOG_ERROR:
			print("\n[ERROR] arg",args[0])
		raise ValueError
	mode = args[1]
	args = args[2:]
	r.run_batch(mode,args)


if __name__ == '__main__':
	#python run_launcher.py cpp param_fixed_values /data/prevel

	if sys.argv[1]=="cpp":
		LOG_LEVEL=LOG_INFO
		r=CppLauncher(None)
	elif sys.argv[1]=="py":
		r=PythonLauncher(None)
	else:
		if LOG_LEVEL<=LOG_ERROR:
			print("\n[ERROR] arg",sys.argv[1])
		raise ValueError
	mode = sys.argv[2]
	args = sys.argv[3:]
	r.run_batch(mode,args)