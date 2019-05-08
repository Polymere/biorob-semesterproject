import sys
import yaml
import numpy as np
import os
import time
from run_batch_controller.run_launcher import runLauncher
from run_batch_controller.unfold_param import ParamMapper
from utils.file_utils import assert_file_exists,assert_dir

ROOT_RESULT_DIR = "/data/prevel/runs"

class EvolutionController():
	"""docstring for ClassName"""
	max_nb_gen=100
	nb_ind=10
	cross_rate=0.1
	mut_rate=0.1
	dofs={"G_SOL":(0,2.0),
			"G_GAS":(0,2.0)}#"GSOL":(0,100)}
	init_pop_mode="rand"
	bound_mod="step"
	world_dir="/data/prevel/trial/worlds_folder"
	def __init__(self, ev_config_file):
		params=yaml.load(open(ev_config_file, 'r'))
		self.__dict__.update(params)
		self.trial_dir = os.path.join(ROOT_RESULT_DIR, time.strftime("%j_%H:%M"))
		assert_dir(self.trial_dir,should_be_empty=True)
		with open(os.path.join(self.trial_dir,"ev_params.yaml"),"w+") as outparams:
			yaml.dump(self.__dict__)
		print("############\n",self.__dict__,"\n############")
		self.run_launcher=runLauncher(self.world_dir,trial_dir=self.trial_dir)

		self.nb_gen=0
		self.flatten_params()
		self.current_pop={}
		#self.param_mapper=ParamMapper()
	def init_pop(self):

		if self.init_pop_mode=="rand":
			#print(self.dof_names,type(self.dof_names))
			#print(self._bound_low,type(self._bound_low))
			#print(self._bound_high,type(self._bound_high))
			for i in range(self.nb_ind):
				uid="gen"+str(self.nb_gen)+"ind"+str(i)
				vals=(self._bound_high-self._bound_low)*np.random.rand(len(self._bound_low)) - self._bound_low
				#print(vals)
				self.current_pop[uid]=dict(zip(self.dof_names,vals))
					
		eval_pop=self.eval_pop()

		selected=self.select(self.sort_pop(eval_pop))
		self.childrens=self.cross_and_mutate(selected)

	def evolve(self):
		self.init_pop()
		while  self.is_stop():
			self.eval_pop(self.childrens)
			union=None

			self.nb_gen+=1

	def cross_and_mutate(self,selected_parents):
		raise NotImplementedError

	def select(self,sorted_pop):
		raise NotImplementedError

	def sort_pop(self,eval_pop):
		raise NotImplementedError
	def eval_pop(self,population):
		result_dir=self.run_launcher.run_batch("pop",population,self.nb_gen,split=True)
		print(result_dir)
		return None
			
				


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

		


if __name__ == '__main__':
	config_file=sys.argv[1]
	assert_file_exists(config_file, should_exist=True)
	c=EvolutionController(config_file)
	c.evolve()