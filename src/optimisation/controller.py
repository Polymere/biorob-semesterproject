import sys
import yaml
import numpy as np
from run_batch_controller.run_launcher import runLauncher
from utils.file_utils import assert_file_exists
class EvolutionController():
	"""docstring for ClassName"""
	max_nb_gen=100
	nb_ind=10
	cross_rate=0.1
	mut_rate=0.1
	dofs={"GSOL":(0,2.0),
			"GGAS":(0,2.0)}#"GSOL":(0,100)}
	init_pop_mod="rand"
	bound_mod="step"
	def __init__(self, ev_config_file):
		params=yaml.load(open(ev_config_file, 'r'))
		self.__dict__.update(params)
		self.nb_gen=0
		self.flatten_params()
		self.run_launcher=runLauncher()
	def init_pop(self):
		if self.init_pop_mode=="rand":
			for i in range(nb_ind):
				self.current_pop[i]=(self._bound_high-self._bound_low)*np.random.rand(len(self._bound_low))\
														- self._bound_low
		self.eval_pop()

	def evolve(self):
		self.init_pop()
		while  self.is_stop():

	def eval_pop(self):

	def is_stop(self):
		if self.nb_gen>self.max_nb_gen:
			return True
	def flatten_params(self):
		nb_par=len(self.dofs)
		self._bound_low=np.empty(nb_par, np.float16)
		self._bound_high=np.empty(nb_par, np.float16)
		self._params=[]
		i=0
		for dof_name,dof_bounds in self.dofs:
			self._bound_low[i]=dof_bounds[0]
			self._bound_high[i]=dof_bounds[1]
			self._params.append(dof_name)
			i+=1

		


if __name__ == '__main__':
	config_file=sys.argv[1:]
	assert_file_exists(config_file, should_exist=True)
	c=EvolutionController(config_file)
	c.evolve()