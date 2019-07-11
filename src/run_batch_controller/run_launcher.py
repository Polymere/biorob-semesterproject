"""
	Launching a run with

"""
import utils.file_utils as fu
import time
import pandas as pd
import data_analysis.import_run as imp
import sys
from shutil import copy, copyfile
import yaml
import os
import subprocess
from run_batch_controller.generate_paramfile_range import gen_all_file
from utils.meta_utils import get_uid_result
import run_batch_controller.unfold_param as up
from run_batch_controller.unfold_param import PythonMapper,CppMapper

from data_analysis.process_run import import_and_process_from_dir,CppRunProcess
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

CPP_CONFIG_PATH="/data/prevel/repos/humanWebotsNmm/config/2D_paul"

CPP_LOG_PATH="/data/prevel/repos/humanWebotsNmm/log/current_log"

DEFAULT_PYTHON_WORLD="/data/prevel/worlds_folder/python_worlds"

CPP_WORLD_PATH = "/data/prevel/repos/humanWebotsNmm/webots/worlds/current.wbt"

DEFAULT_CPP_WORLD="/data/prevel/worlds_folder/cpp_worlds"


CPP_param_path_31= ["/data/prevel/repos/humanWebotsNmm/config/2D_ind1/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind2/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind3/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind4/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind5/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind6/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind7/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind8/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind9/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind10/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind11/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind12/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind13/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind14/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind15/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind16/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind17/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind18/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind19/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind20/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind21/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind22/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind23/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind24/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind25/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind26/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind27/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind28/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind29/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind30/gaits/current",
              "/data/prevel/repos/humanWebotsNmm/config/2D_ind31/gaits/current"]

CPP_world_path_31= ["/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind1.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind2.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind3.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind4.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind5.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind6.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind7.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind8.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind9.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind10.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind11.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind12.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind13.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind14.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind15.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind16.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind17.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind18.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind19.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind20.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind21.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind22.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind23.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind24.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind25.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind26.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind27.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind28.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind29.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind30.wbt",
              "/data/prevel/repos/humanWebotsNmm/webots/worlds/tmp_2D_noObstacle_GA_2Dind31.wbt"]
CPP_log_path_31= ["/data/prevel/repos/humanWebotsNmm/log/log_ind1",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind2",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind3",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind4",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind5",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind6",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind7",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind8",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind9",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind10",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind11",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind12",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind13",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind14",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind15",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind16",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind17",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind18",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind19",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind20",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind21",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind22",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind23",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind24",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind25",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind26",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind27",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind28",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind29",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind30",
            "/data/prevel/repos/humanWebotsNmm/log/log_ind31"]

LOG_DEBUG=1
LOG_INFO=2
LOG_WARNING=3
LOG_ERROR=4
LOG_LEVEL=LOG_INFO

class runLauncher:
	world_counter = 1
	fold_counter=0
	individual_counter = 1
	max_folds = 1
	trial_dir=None

	on_cluster=False
	nb_eval=4


	def __init__(self,args):
		if args is not None:
			for arg_name,arg_value in args.items():
				if hasattr(self, arg_name):
					setattr(self, arg_name, arg_value)
			if LOG_LEVEL<=LOG_INFO:
				print("\n[INFO]Launcher ",self.__class__.__name__," initialized with\n",self.__dict__)
		else:
			if LOG_LEVEL<=LOG_WARNING:
				print("\n[WARNING]No args\n")
				return

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


	def run_batch(self,mode,*args,**kwargs):
		if mode == "sensitivity_analysis":
			param_values_file = args[0][0]
			if LOG_LEVEL<=LOG_INFO:
				print("\n[INFO]Parameters taken from\n",param_values_file)
			self.gens=gen_all_file(param_values_file, standalone=True)
			gen_count=1
			#print(self.gens)
			#return
			for gen in self.gens:

				self.individuals_ids=range(1,len(gen)+1)
				self.run_gen("param"+str(gen_count))
				gen_count+=1

		if mode == "parallel_sensitivity_analysis":
			self.nb_eval=10
			self.param_path=CPP_param_path_31
			self.world_path=CPP_world_path_31
			self.log_path=CPP_log_path_31
			param_values_file = args[0][0]
			if LOG_LEVEL<=LOG_INFO:
				print("\n[INFO]Parameters taken from\n",param_values_file)
			all_params,gen_ids=gen_all_file(param_values_file, standalone=True)
			gen_count=1
			def single_to_list(val):
				if type(val) is not list:
					if LOG_LEVEL<=LOG_WARNING:
						print("\n[WARNING]Is \t",val,"supposed to be single value?")
					return [val]
				else:
					return val
			param_paths=single_to_list(self.param_path)
			log_paths=single_to_list(self.log_path)
			world_paths=single_to_list(self.world_path)
			for gen_vals, gen_id in zip(all_params,gen_ids):

				save_dir = os.path.join(self.trial_dir, gen_id)
				fu.assert_dir(save_dir, should_be_empty=True)
				population={}
				for ind, cnt in zip(gen_vals,range(len(gen_vals))):
					uid="param"+str(gen_count)+"val"+str(cnt)
					population[uid]=ind
				param_paths=param_paths[0:cnt+1]
				log_paths=log_paths[0:cnt+1]
				world_paths=world_paths[0:cnt+1]
				self.create_pop(population, param_paths, log_paths)
				if len(world_paths)!=len(log_paths):
					if LOG_LEVEL<=LOG_ERROR:
						print("\n[ERROR]Must have as much worlds as log paths!\nWorlds:\n",
								world_paths,"\nLogs:\n",log_paths)
					raise ValueError

				self.run_worlds(world_paths, log_paths)
				runs=self.proc_run.import_runs_from_dir(log_paths,save=False)
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
			worlds=args[0]
			if len(args)>1:
				logs=args[1]
			else:
				logs=None
			self.run_worlds(worlds,logs)
			return
		else:
			raise KeyError
	def run_worlds(self,worlds,logs):
		nb_ind=len(worlds)
		tstart=time.time()
		part_worlds=[worlds[sl:sl+self.nb_eval] for sl in range(0,nb_ind,self.nb_eval)]
		wd=os.path.dirname(os.path.realpath(__file__))
		for slice_world in part_worlds:
			if LOG_LEVEL<=LOG_DEBUG:
				print("\n[DEBUG]Running slice\n",slice_world)
			if self.on_cluster:
				subprocess.run([os.path.join(wd,"launchAllWebots_cluster.sh")] + slice_world)
			else:
				subprocess.run([os.path.join(wd,"launchAllWebots.sh")] + slice_world)
		if LOG_LEVEL<=LOG_INFO:
			print("\n[INFO]Time elapsed (",len(worlds),"evals):\t",time.time()-tstart,"s\n")


	def run_ind(self):
		if self.fold_counter==0:
			copyfile(self.cworld, self.world_path)
		subprocess.run(["webots", "--mode=fast", "--batch","--minimize", self.world_path])
		run_suffix="_w"+str(self.world_counter)+"_f"+str(self.fold_counter+1)
		self.import_run(self.log_path,save_to_single=True,save_path=self.cdir, save_name="raw"+run_suffix)
		
	def create_pop(self,population,param_paths,log_paths,verbose=False):
		raise NotImplementedError
	def run_gen(self, gen_id, **opt_args):
		raise NotImplementedError
	def wait_for_fitness(self,log_paths,verbose=True):
		raise NotImplementedError

	def dump_meta(self,dct):
		run_suffix="_w"+str(self.world_counter)+"_f"+str(self.fold_counter+1)               
		meta_file_path=os.path.join(self.cdir, "meta"+run_suffix+".yaml")
		if LOG_LEVEL<=LOG_DEBUG:
			print("\n[DEBUG]Dumping\n",dct,"at\n",meta_file_path)
		with open(meta_file_path, 'a+') as meta:
			yaml.dump(dct,meta)

class PythonLauncher(runLauncher):
	def __init__(self,worlds=DEFAULT_PYTHON_WORLD,**kwargs):
		super(PythonLauncher, self).__init__(arg)
		#runLauncher.__init__(self,worlds,**kwargs)
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
			self.mapper.complete_and_save(ind, PARAMFILE_ABSPATH,do_split)

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
					self.dump_meta({"opt_params":dict(ind)})
					self.dump_meta(self._get_meta_dct(gen_id=gen_id))
					self.run_ind()
					self.fold_counter += 1
				self.world_counter += 1
			self.individual_counter += 1
		print("\n**************************")
		return self.gen_dir

class CppLauncher(runLauncher):
	def __init__(self,args):
		#runLauncher.__init__(self,args)
		super(CppLauncher, self).__init__(args)
		self.mapper=CppMapper()
		#self.import_run=imp.cpp_import_run # change with language dep
		self.proc_run=CppRunProcess(None)
		self.param_path=os.path.join(CPP_CONFIG_PATH,"gaits/current")
		self.worlds=	DEFAULT_CPP_WORLD
		self.log_path= CPP_LOG_PATH
		self.world_path= CPP_WORLD_PATH

	def run_gen(self, gen_id, **opt_args):
		if LOG_LEVEL<=LOG_WARNING:
			print("\n[WARNING]Single run is deprecated, see run_worlds\n",)
		raise DeprecationWarning
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
		for ind,ind_uid in zip(self.individuals,self.individuals_ids):
			print("Individual", ind_uid,":\t",self.individual_counter, "/",tot_ind,"\n")
			self.world_counter = 1
			self.cdir = os.path.join(self.gen_dir, "ind" + str(self.individual_counter))
			fu.assert_dir(self.cdir, should_be_empty=True)
			self.mapper.complete_and_save(ind, self.param_path)
			param_file_copypath = os.path.join(self.cdir, "full_params")
			copyfile(self.param_path, param_file_copypath)


			for world in self.worlds:
				self.cworld=world
				if len(self.worlds) > 1:
					print("\tWorld:\t", self.world_counter, "\n")
				self.fold_counter = 0
				for self.fold_counter in range(self.max_folds):
					self.dump_meta({"opt_params":dict(ind),
									"uid":ind_uid})
					#self.dump_meta({"uid":ind_uid})
					#self.dump_meta(self._get_meta_dct(gen_id=gen_id))
					self.run_ind()
				self.world_counter += 1
			self.individual_counter += 1
		print("\n**************************")
		return self.gen_dir


	def create_pop(self,population,param_paths,log_paths,verbose=False):

		if len(log_paths)!=len(param_paths) or len(population)!=len(log_paths):
			if LOG_LEVEL<=LOG_ERROR:
				print("\n[ERROR]Should have has much valid param_paths as ind:\n\tpop",
					len(population),"params",len(param_paths),"logs",len(log_paths))

			raise ValueError
		for ind, ind_uid, param_path, log_path in zip(population.values(), population.keys(),param_paths,log_paths):
			self.cdir=log_path
			self.dump_meta({"opt_params":ind,
				"uid":ind_uid})
			self.mapper.complete_and_save(ind, param_path)
			if LOG_LEVEL<=LOG_DEBUG:
				print("\n[DEBUG]Created param at",param_path,"for ind",ind_uid,"with params\n",ind)

	def wait_for_fitness(self,log_paths,verbose=True):
		evaluations_terminated=False
		cnt=0
		nb_ind=len(log_paths)
		res={}
		while not evaluations_terminated:
			for log_path in log_paths:
				res_path=os.path.join(log_path,"result.csv")
				if fu.assert_file_exists(res_path,should_exist=True):
					if LOG_LEVEL<=LOG_INFO:
						print("\n[INFO]Run in \n",log_path,"has finished")
						
					result=get_uid_result(log_path)
					if cnt==0:
						result_df=pd.DataFrame(columns=result.keys(),
												index=pd.RangeIndex((len(log_paths))))
					result_df.loc[cnt]=result
					cnt+=1
					log_paths.remove(log_path)
					yamls=fu.file_list(log_path,file_format=".yaml")
					csvs=fu.file_list(log_path,file_format=".csv")
					if LOG_LEVEL<=LOG_DEBUG:
						print("\n[DEBUG]Removing\n",yamls,"\n",csvs)
					if yamls:
						os.unlink(yamls[0])
					if csvs:
						os.unlink(csvs[0])

			if cnt==nb_ind:
				evaluations_terminated=True
			else:
				time.sleep(1.0)
		return result_df.set_index('uid')

	def check_run(self,ind):
		self.mapper.complete_and_save(ind, self.param_path)
		if LOG_LEVEL<=LOG_INFO:
			print("\n[INFO]Writing \n",ind,"\tto\n",self.param_path)
			print("\n[INFO]Running \n",self.world_path)
		subprocess.run(["webots", "--batch","--mode=realtime","--fullscreen", self.world_path])
		proc=CppRunProcess({"kinematics_compare_kind" :"winter_to_cpp",
							"kinematics_compare_file" : "../../data/winter_data/data_normal.csv",
							"split_how": "strike_to_strike",
							"do_plot": True})
		df=proc.get_raw(self.log_path)
		proc.ref.get_corr(df)
		df.to_csv("last_run.csv")
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

	r.run_batch(sys.argv[2],sys.argv[3:])


