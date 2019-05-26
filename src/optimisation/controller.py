import sys
import yaml
import numpy as np
import pandas as pd
import os
import time
from run_batch_controller.run_launcher import CppLauncher
from utils.file_utils import assert_file_exists,assert_dir
from data_analysis.process_run import CppRunProcess,PythonRunProcess


NORUNMODE=True
if NORUNMODE:
	ROOT_RESULT_DIR = "./trash"
else:
	ROOT_RESULT_DIR = "/data/prevel/runs"
NORUNMODE_RUNDIR="../../data/template_gen"

class EvolutionController():
	"""docstring for ClassName"""
	max_nb_gen=100
	nb_ind=10
	cross_rate=0.1
	mut_rate=1.0
	mut_amp=10
	dofs={"G_SOL":(0,2.0),
			"G_GAS":(0,2.0)}#"GSOL":(0,100)}
	init_pop_mode="rand"
	# Random values within boundaries
	bound_mod="step" 
	# only valid one for now, maybe allow going above boundaries (soft boundaries) for a certain cost (optimization param)
	
	def __init__(self, ev_config):

		self.__dict__.update(ev_config)

		self.trial_dir = os.path.join(ROOT_RESULT_DIR, time.strftime("%j_%H:%M"))

		assert_dir(self.trial_dir,should_be_empty=True)
		with open(os.path.join(self.trial_dir,"ev_params.yaml"),"w+") as outparams:
			yaml.dump(self.__dict__)
		print("############\n",self.__dict__,"\n############")
		self.run_launcher=None
		self.nb_gen=0
		self.flatten_params()
		self.current_pop=None
		self.start_time=time.time()

	def set_init_pop(self):

		uids=["gen"+str(self.nb_gen)+"ind"+str(i+1) for i in range(self.nb_ind)]
		pop_df=pd.DataFrame(index=uids,columns=self.dofs.keys())
		if self.init_pop_mode=="rand":
				vals=(self._bound_high-self._bound_low)*np.random.rand(self.nb_ind,len(self._bound_low)) - self._bound_low
				pop_df[:]=vals
				self.current_pop=pop_df
		elif self.init_pop_mode=="one_parent":
			# initial population is generated by mutating a reference parent (working initial parameter set)
			raise NotImplementedError
		elif self.init_pop_mode=="multiple_parent":
			# initial population is generated by mutation and crossover of a full parent population
			# can be used to restart evolution from a certain generation
			raise NotImplementedError
	def init_pop(self):
		self.set_init_pop()
		eval_pop=self.eval_pop(self.current_pop)
		selected=self.select(self.sort_pop(eval_pop))
		self.nb_gen+=1
		self.childrens=self.cross_and_mutate(selected)
		

	def template_evolve(self):
		while  not self.is_stop():
			self.set_init_pop()
			self.eval_pop(self.current_pop)
			self.nb_gen+=1

	def evolve(self):

		self.init_pop()
		while not self.is_stop():
			self.eval_pop(self.childrens)
			union=None

			self.nb_gen+=1
	def check_bound(self,population):
		#print(self._bound_high.repeat(len(population),axis=1))
		for row in population.values:
			row=row+row[row>self._bound_high]*(self._bound_high-row)
			row[row<self._bound_low]=self._bound_low
		#
		#print(population)
		#=self._bound_high#.reshape(population.values.shape)
		#population[population[:]<self._bound_low]=self._bound_low[:]

		print("\n[DEBUG]Clipped boundaries\n",population)
		raise NotImplementedError
	def cross_and_mutate(self,selected_parents):
		t=time.time_ns()
		print("\n[DEBUG]Parents\n",selected_parents)
		nb_dofs=len(self.dofs)
		#print("\n[DEBUG]Parents\n",selected_parents)
		couples=np.random.randint(0, len(selected_parents.index), (2,self.nb_ind))
		#print("\n[DEBUG]Couples\n",couples)
		p1_ids=selected_parents.iloc[couples[0][:]]
		p2_ids=selected_parents.iloc[couples[1][:]]
		#print("\n[DEBUG]Parent1\n",p1_ids)
		#print("\n[DEBUG]Parent2\n",p2_ids)
		
		cross_select=(np.random.randint(0,100,(self.nb_ind,nb_dofs))<100*self.cross_rate)
		# probability cross_rate to take a param from parent2 
		#print("\n[DEBUG] Cross select ids\n",cross_select)
		uids=["gen"+str(self.nb_gen)+"ind"+str(i+1) for i in range(self.nb_ind)]
		

		child_df=pd.DataFrame(index=uids,columns=self.dofs.keys())

		child_df[:]=p1_ids.values*cross_select[:]+\
					p2_ids.values*np.logical_not(cross_select[:])#.logical_not
		#print("\n[DEBUG]Crossover result\n",child_df)

		mutate=(np.random.randint(0,100,(self.nb_ind,nb_dofs))<100*self.mut_rate)
		#print("\n[DEBUG]Mutate\n",mutate)
		mutate_amp=np.random.randn(self.nb_ind,nb_dofs)*child_df[:]*self.mut_amp
		

		#print("\n[DEBUG]Mutate amplitude\n",mutate_amp)

		child_df[:]=child_df[:]+mutate_amp*mutate

		print("\n[DEBUG]Mutation result\n",child_df)
		#print("\n[DEBUG]Mutation result\n",child_df.to_dict(orient="index"))
				# probability mutrate to add a normal of std value*mut_amp to param
		#print("\n[INFO]Time",(time.time_ns()-t)/1e6)
		return self.check_bound(child_df)

	def select(self,sorted_pop):
		print(sorted_pop.head(5))
		return sorted_pop.head(5)#["uid"]

	def sort_pop(self,eval_pop):
		print(eval_pop)
		#eval_pop["fitness"]=eval_pop.filter(like="cor").sum(1)
		return eval_pop.sum(1).sort_values(ascending=False)

		
		#raise NotImplementedError
	def eval_pop(self,population):
		if NORUNMODE:
			#print(self.current_pop)
			#scores=self.run_processor.process_gen(NORUNMODE_RUNDIR)
			self.run_launcher.create_pop(population.to_dict(orient="index"),
										self.param_paths,
										self.log_paths)
			#print("\n[DEBUG] Eval pop\n",scores)
			#scores=pd.DataFrame.from_dict(self.run_launcher.wait_for_fitness(self.log_paths))
			scores=self.run_launcher.wait_for_fitness(self.log_paths.copy())
		else:	
			result_dir=self.run_launcher.run_batch("pop",population,self.nb_gen)

			scores=self.run_processor.process_gen(result_dir)
		return scores
			
				


	def is_stop(self):
		if self.nb_gen>self.max_nb_gen:
			print("\n[INFO]Finished after",self.nb_gen,"generations in",np.round(time.time()-self.start_time),"s")
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

class PythonEvolutionController(EvolutionController):
	def __init__(self,worlds,**kwargs):
		EvolutionController.__init__(self,worlds,**kwargs)
		self.run_launcher=PythonLauncher(self.world_dir,trial_dir=self.trial_dir)
		if "kinematics_compare_file" in kwargs.keys():
			compare_files=kwargs["kinematics_compare_file"]
		else:
			compare_files=None
		self.run_processor=PythonRunProcess(compare_files)

class CppEvolutionController(EvolutionController):
	def __init__(self,config_file):
		params=yaml.load(open(config_file, 'r'))
		if "kinematics_compare_file" in params.keys():
			compare_files=params["kinematics_compare_file"]
			del params["kinematics_compare_file"]
		else:
			compare_files=None

		EvolutionController.__init__(self,params)
		self.run_launcher=CppLauncher(self.world_dir,trial_dir=self.trial_dir)

		self.run_processor=CppRunProcess(compare_files)

if __name__ == '__main__':
	config_file="../../data/references/ev_config_template.yaml"
	mode="cpp_dev"
	assert_file_exists(config_file, should_exist=True)
	if mode=="cpp":
		c=CppEvolutionController(config_file)
		c.evolve()
	if mode=="cpp_dev":
		print("\nRUNNING TEMPLATE EVOLUTION")
		c=CppEvolutionController(config_file)
		c.set_init_pop()
		c.cross_and_mutate(c.current_pop)
		#c.template_evolve()