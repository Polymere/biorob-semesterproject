import sys
import yaml
import numpy as np
import pandas as pd
import os
import time

from shutil import copyfile

from run_batch_controller.run_launcher import CppLauncher

from utils.file_utils import assert_file_exists,assert_dir

from data_analysis.process_run import CppRunProcess,PythonRunProcess

from optimisation.optimizers import PSOptimizer,GAOptimizer,NSGAIIOptimizer

NORUNMODE=False
if NORUNMODE:
	ROOT_RESULT_DIR = "./trash"
else:
	ROOT_RESULT_DIR = "/data/prevel/runs"

import warnings
warnings.filterwarnings('ignore') # pandas warning a utter trash

LOG_DEBUG=1
LOG_INFO=2
LOG_WARNING=3
LOG_ERROR=4
LOG_LEVEL=LOG_INFO

class EvolutionController():
	"""docstring for ClassName"""
	run_launcher=None
	run_processor=None
	max_nb_gen=100
	nb_ind=10
	dofs=None

	optimizer_name="NSGAII"
	init_pop_mode="rand"
	# Random values within boundaries
	
	def __init__(self, ev_config):

		self.__dict__.update(ev_config)

		self.trial_dir = os.path.join(ROOT_RESULT_DIR, time.strftime("%j_%H:%M"))
		assert_dir(self.trial_dir,should_be_empty=True)

		del ev_config["worlds_path"]
		del ev_config["log_paths"]
		del ev_config["param_paths"]
		with open(os.path.join(self.trial_dir,"ev_params.yaml"),"w+") as outparams:
			yaml.dump(ev_config,outparams,default_flow_style=False)

		if self.optimizer_name=="NSGAII":
			self.union_pop=True
			self.opti=NSGAIIOptimizer(ev_config)
		elif self.optimizer_name=="GA":
			self.union_pop=False
			self.opti=GAOptimizer(ev_config)
		elif self.optimizer_name=="PSO":
			self.union_pop=False
			self.opti=PSOptimizer(ev_config)
		else:
			if LOG_LEVEL<=LOG_ERROR:
				print("\n[ERROR]Unknown optimizer name\n",self.optimizer_name)
			raise ValueError

		self.nb_gen=0
		self.flatten_params()
		self.current_pop=None

	def set_init_pop(self):

		uids=["gen"+str(self.nb_gen)+"ind"+str(i+1) for i in range(self.nb_ind)]
		pop_df=pd.DataFrame(index=uids,columns=self.dofs.keys())
		if self.init_pop_mode=="rand":
				vals=(self._bound_high-self._bound_low)*np.random.rand(self.nb_ind,len(self._bound_low)) - self._bound_low
				pop_df[:]=vals
				return pop_df
		elif self.init_pop_mode=="one_parent":
			"""
				initial population is generated by mutating a reference parent (working initial parameter set)
				"""
			if self.initial_pop is None:
				raise ValueError
			print("\n[INFO]Initializing population from parent\n",self.initial_pop)
			for index, row in pop_df.iterrows():
				pop_df.loc[index]=self.initial_pop
			select_ids=pop_df.head(1)
			return self.opti.get_next_gen(select_ids,self.nb_gen)
		elif self.init_pop_mode=="multiple_parents":
			""" 
				initial population is generated by mutation and crossover of a full parent population
				can be used to restart evolution from a certain generation
				"""
			if self.initial_pop is None:
				raise ValueError
			parents=pd.read_csv(self.initial_pop)
			return self.opti.get_next_gen(parents,self.nb_gen)
		else:
			raise KeyError
			
	def init_pop(self):
		raise DeprecationWarning
		self.set_init_pop()
		eval_pop=self.eval_pop(self.current_pop)
		parents=self.opti.select(self.opti.sort_pop(eval_pop))
		if LOG_LEVEL<=LOG_INFO:
			print("\n[INFO]Gen",self.nb_gen," Parents\n",parents)
		self.nb_gen+=1	
		return self.opti.get_next_gen(parents,self.nb_gen)


	def evolve(self):
		childrens=self.set_init_pop()
		parents=None
		while not self.is_stop():
			print("\n[INFO]***************GEN",self.nb_gen,"***************\n")
			eval_pop=self.eval_pop(childrens)
			if self.union_pop and parents is not None:
				candidates=pd.concat([eval_pop,parents],axis=0)
			else:
				candidates=eval_pop
			parents=self.opti.select(self.opti.sort_pop(candidates),self.gen_nb)
			if LOG_LEVEL<=LOG_INFO:
				print("\n[INFO]Gen",self.nb_gen," Parents\n",parents)
			self.nb_gen+=1
			childrens=self.opti.get_next_gen(parents,self.nb_gen) # drop fitnesses and cr_dist for check_bound

			parents.to_csv(os.path.join(self.trial_dir,"parents_gen"+str(self.nb_gen)+".csv"))
			if self.nb_gen>2:
				os.unlink(os.path.join(self.trial_dir,"parents_gen"+str(self.nb_gen-1)+".csv"))


	def eval_pop(self,population,verbose=False):
		if verbose:
			print("\n[INFO]Local evaluation, population\n",population)
		self.run_launcher.create_pop(population.to_dict(orient="index"),
									self.param_paths,
									self.log_paths)

		self.run_launcher.run_batch("worlds",self.worlds_path,self.log_paths)

		scores=self.run_processor.get_fitness_from_dir(self.log_paths)
		population["gen"]=self.nb_gen
		if LOG_LEVEL<=LOG_DEBUG:
			print("\n[DEBUG]Pop\n",population)
			print("\n[DEBUG]scores\n",scores)

		saved_df=pd.concat([population,scores],axis=1)
		#saved_df["gen"]=self.nb_gen
		saved_df.to_csv(os.path.join(self.trial_dir,"gen"+str(self.nb_gen)+".csv"))

		if self.opti.is_single_obj:
			scores=scores.add(scores.fit_stable,axis='index')
			scores["fitness"]=scores.filter(self.objectives_metrics).sum(1)
			evaluated=pd.concat([population,scores.fitness],axis=1)
		else:
			scores=scores.add(scores.fit_stable,axis='index')
			evaluated=pd.concat([population,scores.filter(self.objectives_metrics)],axis=1)

		if LOG_LEVEL<=LOG_DEBUG:
			print("\n[DEBUG]Evaluated pop\n",evaluated)
		return evaluated

	def is_stop(self):
		if self.nb_gen>self.max_nb_gen:
			return True
	def flatten_params(self):
		nb_par=len(self.dofs)
		self._bound_low=np.empty(nb_par, np.float16)
		self._bound_high=np.empty(nb_par, np.float16)
		self._params=[]
		self.dof_names=[]
		i=0
		for dof_name,dof_bounds in self.dofs.items():
			self._bound_low[i]=dof_bounds[0]
			self._bound_high[i]=dof_bounds[1]
			self.dof_names.append(dof_name)
			i+=1





class CppEvolutionController(EvolutionController):
	def __init__(self,config_file):
		params=yaml.load(open(config_file, 'r'))
		EvolutionController.__init__(self,params)

		self.run_launcher=CppLauncher(params)
		self.run_processor=CppRunProcess(params)

class PythonEvolutionController(EvolutionController):
	def __init__(self,config_file):
		params=yaml.load(open(config_file, 'r'))
		EvolutionController.__init__(self,params)

		self.run_launcher=PythonLauncher(params)
		self.run_processor=PythonRunProcess(params)

if __name__ == '__main__':
	ev_file=sys.argv[1]
	model="cpp"
	

	if model=="cpp":
		config_file=os.path.join("../../data/references",ev_file)
		assert_file_exists(config_file, should_exist=True)
		c=CppEvolutionController(config_file)
		c.evolve()

	elif model=="cpp_debug":
		config_file=os.path.join("../../data/references",ev_file)
		assert_file_exists(config_file, should_exist=True)
		c=CppEvolutionController(config_file)
		LOG_LEVEL=LOG_DEBUG
		c.evolve()
