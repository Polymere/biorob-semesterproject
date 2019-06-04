import sys
import yaml
import numpy as np
import pandas as pd
import os
import time

import warnings
warnings.filterwarnings('ignore') # pandas warning a utter trash

LOG_DEBUG=1
LOG_INFO=2
LOG_WARNING=3
LOG_ERROR=4
LOG_LEVEL=LOG_WARNING

class Optimizer:
	#_bound_high=None
	#_bound_low=None
	bound_mod=None
	dofs=None
	nb_ind=None
	nb_parents=None

	def __init__(self,args):
		for arg_name,arg_value in args.items():
			if hasattr(self, arg_name):
				setattr(self, arg_name, arg_value)
		if LOG_LEVEL<=LOG_INFO:
			print("\n[INFO]Optimizer ",self.__class__.__name__," initialized with\n",self.__dict__)

		self.flatten_params()
	def check_bound(self,population):
		for index, row in population.iterrows():
			if LOG_LEVEL<=LOG_DEBUG:
				print("\n[DEBUG]check_bound\n",row)
			if self.bound_mod=="clip":
				row=row+(row>self._bound_high)*(self._bound_high-row)
				row=row+(row<self._bound_low)*(self._bound_low-row)
				population.loc[index]=row.values
			elif self.bound_mod=="exp_handicap":
				handicap=	(row>self._bound_high)*(row-self._bound_high)+\
							(row<self._bound_low)*(self._bound_low-row)
				if LOG_LEVEL<=LOG_DEBUG:
					print("\n[DEBUG]handicap\n",handicap)
					print("\n[DEBUG]type\n",type(handicap))
					print("\n[DEBUG]type\n",type(handicap.values[0]))
				fl_handicap=np.sum(np.exp(handicap.values.astype(np.float))-1)
				if LOG_LEVEL<=LOG_DEBUG:
					print("\n[DEBUG]fl_handicap\n",fl_handicap)
				population.loc[index,"fit_handicap"]=-fl_handicap
		if LOG_LEVEL<=LOG_DEBUG:
			print("\n[DEBUG]Boundaries\n",population)
		return population

	def benchmark(self,problem,nb_gen,nb_dim):
		if problem=="rastrigin":
			self.n=nb_dim
			self.dofs={}
			for i in range(nb_dim):
				self.dofs[str(i)]=[-5.12,5.12]
			self.bound_mod="clip"
			self.bench_eval=self._eval_rastrigin
		elif problem=="sphere":
			self.dofs={}
			for i in range(nb_dim):
				self.dofs[str(i)]=[-10,10]
			self.bench_eval=self._eval_sphere
		elif problem=="fonseca_fleming":
			self.n=nb_dim
			self.dofs={}
			for i in range(nb_dim):
				self.dofs[str(i)]=[-4,4]
			self.bench_eval=self._eval_fonseca_fleming

		self.flatten_params()
		gen_counter=0
		uids=["gen"+str(gen_counter)+"ind"+str(i+1) for i in range(self.nb_ind)]
		pop_df=pd.DataFrame(index=uids,columns=self.dofs.keys())
		vals=(self._bound_high-self._bound_low)*np.random.rand(self.nb_ind,len(self._bound_low)) - self._bound_low
		pop_df[:]=vals
		self.current_pop=pop_df
		while gen_counter<nb_gen:
			eval_pop=self.bench_eval(self.current_pop)
			parents=self.select(self.sort_pop(eval_pop))
			if gen_counter%10==0:
				print(parents)
			gen_counter+=1
			self.current_pop=self.get_next_gen(eval_pop,gen_counter)

	def _eval_rastrigin(self,pop):
		fit=10*self.n+(pop.values**2-10*np.cos(pop.values*2*np.pi)).sum(1)
		pop["fit"]= - fit
		return pop
	def _eval_sphere(self,pop):
		fit=(pop.values**2).sum(1)
		pop["fit"]=-fit
		return pop
	def _eval_fonseca_fleming(self,pop):
		f1= 1 - np.exp( ( - ( pop.values - 1 / np.sqrt( self.n ) )**2 ).sum(1) )
		f2= 1 - np.exp( ( - ( pop.values + 1 / np.sqrt( self.n ) )**2 ).sum(1) )
		pop["fit_1"]= - f1
		pop["fit_2"]= - f2
		return pop

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

	def sort_pop(self,eval_pop):
		srt=eval_pop.filter(like="fit").sum(1).sort_values(axis='index',ascending=False)
		return eval_pop.loc[srt.index,:]
	def select(self,sorted_pop):
		return sorted_pop.head(self.nb_parents)#["uid"]
	def set_uid(self,gen_nb,new_gen):
		uids=["gen"+str(gen_nb)+"ind"+str(i+1) for i in range(self.nb_ind)]
		new_gen.index=uids
		return new_gen

	def get_next_gen(self,parents,gen_nb):
		raise NotImplementedError

class PSOptimizer(Optimizer):

	def __init__(self, arg):
		super(PSOptimizer, self).__init__(arg)
		self.is_single_obj=True
		raise NotImplementedError


class GAOptimizer(Optimizer):
	"""docstring for GAOptimizer"""
	mut_amp=None
	mut_rate=None
	cross_rate=None
	drop_age=None
	def __init__(self, arg):
		super(GAOptimizer, self).__init__(arg)
		self.is_single_obj=True

	def check_age(self,pop,current_gen):
		if self.drop_age is None:
			return pop
		else:
			raise NotImplementedError

	def get_next_gen(self,parents,gen_nb):
		new_gen=self.check_bound(self.cross_and_mutate(parents))
		return self.set_uid(gen_nb, new_gen)

	def cross_and_mutate(self,selected_parents):
		nb_dofs=len(self.dofs)

		couples=np.random.randint(0, len(selected_parents.index), (2,self.nb_ind))
		if LOG_LEVEL<=LOG_DEBUG:
			print("\n[DEBUG]selected_parents\n",selected_parents)
			print("\n[DEBUG]couples\n",couples)
		p1_ids=selected_parents.iloc[couples[0][:]].index
		p2_ids=selected_parents.iloc[couples[1][:]].index

		cross_select=(np.random.randint(0,100,(self.nb_ind,nb_dofs))<100*self.cross_rate)
		# probability cross_rate to take a param from parent2 
		child_df=pd.DataFrame(index=pd.RangeIndex(self.nb_ind),columns=self.dofs.keys())
		if LOG_LEVEL<=LOG_DEBUG:
			print("\n[DEBUG]child_df\n",child_df.head())
		child_df[:]=(selected_parents.loc[p1_ids,self.dofs.keys()]*cross_select[:]).values+\
					(selected_parents.loc[p2_ids,self.dofs.keys()]*np.logical_not(cross_select[:])).values
		mutate=(np.random.randint(0,100,(self.nb_ind,nb_dofs))<100*self.mut_rate)
		# probability mutrate to add a normal of std value*mut_amp to param
		mutate_amp=np.random.randn(self.nb_ind,nb_dofs)*child_df[:]*self.mut_amp	
		child_df[:]=child_df[:]+mutate_amp*mutate
		if LOG_LEVEL<=LOG_DEBUG:
			print("\n[DEBUG]Child\n",child_df)
		return self.check_bound(child_df)

class NSGAIIOptimizer(GAOptimizer):
	"""docstring for NSGAIIOptimizer"""
	def __init__(self, arg):
		super(NSGAIIOptimizer, self).__init__(arg)
		self.is_single_obj=False

	def sort_pop(self,eval_pop):
		tstart=time.time_ns()
		pop_with_fronts=self.add_fronts(eval_pop)
		tf=time.time_ns()
		fronts_and_dist=self.add_crowding_distance(pop_with_fronts)
		tcr=time.time_ns()
		#print("\n[TIME]\nfront alloc\t",(tf-tstart)/1e6,"[ms]\ncrowding\t",(tcr-tf)/1e6,"[ms]")
		sorted_pop=fronts_and_dist.sort_values(by=["front","cr_dist"],axis='index',ascending=[True,False])
		if LOG_LEVEL<=LOG_INFO:
			print("\n[INFO]Sorted population\n",sorted_pop)
		return sorted_pop
	def add_fronts(self,eval_pop):
		# fast non dominated sort
		fit=eval_pop.filter(like="fit")
		fronts=pd.DataFrame(columns=eval_pop.columns)
		flag_empty_front=False
		other=pd.DataFrame(columns=fit.columns)
		nb_front=1
		pop_in_fronts=0
		while not (flag_empty_front or pop_in_fronts>self.nb_parents):
			current_front=pd.DataFrame(columns=eval_pop.columns)
			for index, indiv in fit.iterrows(): # must be a way to do it without iter
				rel=fit.le(indiv,axis=1).drop(index)
				dominant=rel[(rel.all(axis=1)==True)==True].index
				rel=fit.gt(indiv,axis=1).drop(index)
				#print(rel)
				dominated=rel[(rel.all(axis=1)==True)==True].index
				if LOG_LEVEL<=LOG_DEBUG:
					print("\n[DEBUG]",index,"Dominates\n",dominant.values,"\nDominated by\n",dominated.values)
				if len(dominated)==0:
					current_front.loc[index,eval_pop.columns]=eval_pop.loc[index].values
					current_front.loc[index,"front"]=nb_front
				else:
					other.loc[index]=fit.loc[index].values
			if LOG_LEVEL<=LOG_DEBUG:
				print("\n[DEBUG]other\n",other)
			if LOG_LEVEL<=LOG_INFO:
				print("\n[INFO]Front",nb_front,"with",len(current_front.index),"inds\n")
				print("\n[INFO]",current_front)
				print("\n[INFO]Missing",len(other),"inds\n")
				print("\n[INFO]",other)
			pop_in_fronts+=len(current_front.index)
			if pop_in_fronts>=self.nb_parents:
				if LOG_LEVEL<=LOG_INFO:
					print("\n[INFO]early stop (front",nb_front,")\n")
			fronts=fronts.append(current_front)
			flag_empty_front=(len(other)==0)
			fit=other
			other=pd.DataFrame(columns=fit.columns)
			nb_front+=1

		if LOG_LEVEL<=LOG_INFO:
			fronts=fronts[fronts.columns.drop(fronts.filter(like="Unnamed").columns)]
			print("\n[INFO]Fronts\n",fronts)
		
		return fronts

	
	def add_crowding_distance(self,pop_with_fronts):
		
		if LOG_LEVEL<=LOG_DEBUG:
			print("\n[DEBUG]Front values\n",pop_with_fronts.front.unique())

		if "fit_stable" in pop_with_fronts.columns:
			if LOG_LEVEL<=LOG_WARNING:
				print("\n[WARNING]Removing stability from objectives\n")
				fit=pop_with_fronts.filter(like="fit_")
				fit=fit.add(fit.fit_stable,axis='index')
				pop_with_fronts.loc[:,fit.columns]=fit
				pop_with_fronts.drop("fit_stable",axis=1,inplace=True)
		for front in pop_with_fronts.front.unique():
			pop_front=pop_with_fronts[pop_with_fronts.front==front]
			L=len(pop_front.index)
			pop_front["cr_dist"]=0
			if LOG_LEVEL<=LOG_DEBUG:
				print("\n[DEBUG]Front",front," with pop\n",pop_front)
			for obj in pop_front.filter(like="fit_").columns:
				if LOG_LEVEL<=LOG_DEBUG:
					print("\n[DEBUG]Objective\n",obj)
				sorted_obj=pop_front.sort_values(by=obj,ascending=False,axis='index')
				if LOG_LEVEL<=LOG_DEBUG:
					print("\n[DEBUG]Sorted\n",sorted_obj.loc[:,obj])
				sorted_obj.loc[:,"cr_dist"]=sorted_obj.shift(1).loc[:,obj]-sorted_obj.shift(-1).loc[:,obj]
				sorted_obj.ix[0,"cr_dist"]=np.inf
				sorted_obj.ix[L-1,"cr_dist"]=np.inf
				if LOG_LEVEL<=LOG_DEBUG:
					print("\n[DEBUG]Cr dist\n",sorted_obj)

				pop_front.loc[sorted_obj.index,"cr_dist"]=pop_front.loc[sorted_obj.index,"cr_dist"].add(sorted_obj.cr_dist,axis='index')
			if LOG_LEVEL<=LOG_INFO:
				print("\n[INFO]crowding_distance for front ",front,"\n",pop_front)
			pop_with_fronts.loc[pop_front.index,"cr_dist"]=pop_front.cr_dist
		if LOG_LEVEL<=LOG_INFO:
			print("\n[INFO]Full pop\n",pop_with_fronts)

		return pop_with_fronts

if __name__ == '__main__':
	pars={"mut_amp":0.5,
	"mut_rate":0.9,
	"cross_rate":0.1,
	"nb_ind":100,
	"nb_parents":10}

	opti=GAOptimizer(pars)
	opti.benchmark("fonseca_fleming", 100, 2)