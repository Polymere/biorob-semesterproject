#!/usr/bin/env python
""" @package process_run
Imports run and computes metrics
"""
import pandas as pd
import sys
import os
import yaml
import utils.file_utils as fu
from math import sqrt
from utils.plot_utils import plot_mean_std_fill,plot_correlation_window
import data_analysis.event_detection as ed

from utils.meta_utils import get_run_files
import matplotlib.pyplot as plt
TIME_STEP=1e-3 #ms

LOG_DEBUG=1
LOG_INFO=2
LOG_WARNING=3
LOG_ERROR=4

C3D_KEYS=["LANKLE","LHIP","LKNEE","RANKLE","RHIP","RKNEE"]

WINTER_KEYS=["ankle","hip","knee","ankle","hip","knee"] # same keys for both leg
# we only have data for one leg (assume perfectly symmetrical) 
SHORT_KEYS=["ankle_left","hip_left","knee_left","ankle_right","hip_right","knee_right"]
	

LOG_LEVEL=LOG_DEBUG


class referenceCompare:
	""" Comparison between two gaits (reference gait vs trial gait)

		Imports a reference gait file (.csv) (see repo/data/gaits) and 
		interpolates to 100 datapoints for comparison while handling sign and 
		scale differences between formats

		Compares this reference gait to a trial gait by computing the 
		correlation and rms distance between normalized strides
	"""
	kinematics_compare_file=None
	kinematics_compare_kind=None
	do_plot=False
	split_how="strike_to_strike"
	def __init__(self,args):
		for arg_name,arg_value in args.items():
			if hasattr(self, arg_name):
				setattr(self, arg_name, arg_value)
		if self.kinematics_compare_kind is None or self.kinematics_compare_file is None:
			if LOG_LEVEL<=LOG_WARNING:
				print("\n[WARNING]Missing arguments for referenceCompare in:\n",
					args,"\nGait comparison will not be performed ! \n")
			return None

		if LOG_LEVEL<=LOG_INFO:
			print("\n[INFO]",self.__class__.__name__," initialized with\n",self.__dict__)
		ref_df=pd.read_csv(self.kinematics_compare_file)
		self.rep_strides={}
		if self.kinematics_compare_kind=="winter_to_python":
			self._set_repstrides_winter_py(ref_df)
			self.get_corr=self._get_all_corr_python
		elif self.kinematics_compare_kind=="cpp_to_python":
			if LOG_LEVEL<=LOG_ERROR:
				print("\n[ERROR]No longer suported with initial florin format, \
					preprocess before (see comment below for reference)\n",)
				"""	#directions :
						#	ankle -
						#	knee +
						#	hip +
 
						contact_file=files[0]
						joints_file=files[1]
						contact=pd.read_csv(contact_file,sep=" ")
						joints=pd.read_csv(joints_file,sep=" ")
						self._set_repstrides_florin(contact, joints)
						def _set_repstrides_florin(self,contact,joints):
						# ankle angle is defined in the opposed direction
							for key_gen,key_flor in MAP_MET_FLORIN.items():
								mean_ref,std_ref=ed.get_repr_from_contact(contact,key_flor,joints)
								mean_ref=ed.interp_gaitprcent(mean_ref,100)
								if "ankle" in key_gen: # inversed angle orientation
									mean_ref=-mean_ref
								self.rep_strides[key_gen]=mean_ref
					"""
			raise DeprecationWarning
		elif self.kinematics_compare_kind=="winter_to_cpp":
			self._set_repstrides_winter_cpp(ref_df)
			self.get_corr=self._get_all_corr_cpp
		elif self.kinematics_compare_kind=="python_to_python":
			self._set_repstrides_py_py(ref_df)
			self.get_corr=self._get_all_corr_python
		elif self.kinematics_compare_kind=="c3d_to_cpp":
			self._set_repstrides_c3d_cpp(ref_df)
			self.get_corr=self._get_all_corr_cpp
		else:
			if LOG_LEVEL<=LOG_ERROR:
				print("\n[ERROR]Compare kind ",self.kinematics_compare_kind,
					" not supported\n",)
			raise KeyError


	def _set_repstrides_winter_py(self,win_df):
		"""
		Reference : Winter
		Controller : Python
		"""
		for key_gen,key_win in zip(SHORT_KEYS,WINTER_KEYS):
			mean_ref=ed.interp_gaitprcent(win_df[key_win],100)
			if key_win=="hip": # inversed angle orientation
				mean_ref=-mean_ref
			self.rep_strides[key_gen]=mean_ref

	def _set_repstrides_c3d_cpp(self,win_df):
		"""
		Reference : C3D (transformed to csv)
		Controller : CPP
		hip & ankle angles are defined in the opposed direction
		"""
		for key_gen,key_c3d in zip(SHORT_KEYS,C3D_KEYS):
			mean_ref=ed.interp_gaitprcent(win_df[key_c3d],100)
			if key_c3d=="LHIP"or key_c3d=="RHIP":  # inversed angle orientation
				mean_ref= - mean_ref
			if key_c3d=="LANKLE"or key_c3d=="RANKLE":  # inversed angle orientation
				mean_ref= - mean_ref
			self.rep_strides[key_gen]=mean_ref*(3.1415/180)

	def _set_repstrides_winter_cpp(self,win_df):
		"""
		Reference : Winter
		Controller : CPP
		"""
		for key_gen,key_win in zip(SHORT_KEYS,WINTER_KEYS):
			mean_ref=ed.interp_gaitprcent(win_df[key_win],100)
			if "ankle" in key_win: # inversed angle orientation
				mean_ref=-mean_ref
			if "hip" in key_win: # inversed angle orientation
				mean_ref=-mean_ref
			self.rep_strides[key_gen]=mean_ref*(3.1415/180)

	def _set_repstrides_py_py(self,ref_df):
		"""
		Reference : Python (output of a simulation)
		Controller : Python
		"""
		angles=ref_df.filter(like="angle")
		for key_gen in angles.keys():
			mean_ref,std_ref=ed.get_repr_from_grf(angles,key_gen,
				stride_choice="repmax")
			mean_ref=ed.interp_gaitprcent(mean_ref,100)
			self.rep_strides[key_gen]=mean_ref
	def _set_repstrides_cpp_cpp(self,ref_df):
		"""
		Reference : Cpp (output of a simulation)
		Controller : Cpp
		"""
		angles=ref_df.filter(like="angle")
		for key_gen in angles.keys():
			mean_ref,std_ref=ed.get_repr_from_contact(angles,key_gen,
				stride_choice="repmax")
			mean_ref=ed.interp_gaitprcent(mean_ref,100)
			self.rep_strides[key_gen]=mean_ref

	def get_corr(self,cmp_df):
		""" Placeholder
			See _get_all_corr_python and _get_all_corr_cpp
		"""
		if LOG_LEVEL<=LOG_ERROR:
			print("\n[ERROR]See class initialization\n")
		raise NotImplementedError

	def _get_all_corr_python(self,cmp_df):
		""" Computes correlation and distance between input gait and reference

			For each joint angle in input gait, split and compute a representative
			stride based on event_detection parameters. Then compare this stride
			to the reference stride by computing correlation and distance
			between normalized strides (in lenght).


			Input:
				cmp_df -- Dataframe with both joints angles and ground reaction 
				force data
			Output:
				corr_dist -- dictionnary with structure 
							{joint_name1 : [correlation,distance],
							...}
			Dependencies:
				get_repr_from_grf in event_detection.py
			Notes :
				Not tested
		"""
		corr_dist={}
		for met in self.rep_strides.keys():
			mean_cur,std_cur=ed.get_repr_from_grf(cmp_df,met,
							stride_choice="repmax",how=self.split_how)
			correl=mean_cur.corr(self.rep_strides[met])
			dist=sqrt(((self.rep_strides[met]-mean_cur)**2).sum())
			corr_dist[met]=[correl,dist]
			
		return corr_dist

	def _get_all_corr_cpp(self,cmp_df):
		""" Computes correlation and distance between input gait and reference

			For each joint angle in input gait, split and compute a representative
			stride based on event_detection parameters. Then compare this stride
			to the reference stride by compuiting correlation and distance
			between normalized strides (in lenght).
			If do_plot was set to True when initializing, shows / saves the
			comparison plots

			Input:
				cmp_df -- Dataframe with both joints angles and footfall data,
					output of import_raw in runProcess
			Output:
				corr_dist -- dictionnary with structure 
							{joint_name1 : [correlation,distance],
							...}
			Dependencies:
				get_repr_from_contact in event_detection.py
		"""

		corr_dist={}
		for met in self.rep_strides.keys():
			mean_cur,std_cur=ed.get_repr_from_contact(cmp_df,met,how=self.split_how)
			correl=mean_cur.corr(self.rep_strides[met])
			dist=sqrt(((self.rep_strides[met]-mean_cur)**2).sum())
			corr_dist[met]=[correl,dist]
			if self.do_plot:
				cor_scale=[-10,10]
				ax=plot_correlation_window(mean_cur,self.rep_strides[met],10,scale=cor_scale)
				plot_mean_std_fill(mean_cur*180/3.1415, std_cur*180/3.1415, "b",ax)
				plot_mean_std_fill(self.rep_strides[met]*180/3.1415, None, "k",ax)
				tit=met#+"(cor:"+str(round(correl,1))+", rms:"+str(round(dist,1))+")"
				#plt.title(tit)
				plt.xlim([0,100])
				plt.xlabel("stride %")
				plt.ylabel("Angle [deg] ")
				plt.legend(["Model","Reference"])
				#plt.show()
				plt.tight_layout()
				plt.savefig(tit+".pdf", dpi=None, facecolor='w', edgecolor='k',
					orientation='portrait', papertype=None, format="pdf",
					transparent=False, bbox_inches=None, pad_inches=0.1,
					frameon=True, metadata=None)
				plt.close()
				#print("Correlation:\t",round(correl,3),"\nDistance\t",round(dist,3))
		return corr_dist



class runProcess:
	""" Run processing from logged data in a directory

		Shared methods between the python and cpp implementation of the 
		controller
	"""

	def __init__(self,args):
		""" Initialization of runProcess	

			Set attributes of runProcess class and creates referenceCompare object	
		"""
		if args is None:
			return
		for arg_name,arg_value in args.items():
			if hasattr(self, arg_name):
				setattr(self, arg_name, arg_value)
		if LOG_LEVEL<=LOG_INFO:
			print("\n[INFO]Run processor ",self.__class__.__name__," initialized with\n",self.__dict__)

		self.ref=referenceCompare(args)


	def process_dir(self,gen_dir):
		""" Process all the saved runs in directory, and returns the results

			Processes all the runs in directory according to the parameters of 
			runProcess and referenceCompare, and returns the results as a
			dataframe

			Input :
			gen_dir -- directory containing a run folders
			Output :
			dataframe containing the processed metrics of all the runs, indexed
			by run uid

		"""
		ind_dirs=fu.dir_list(gen_dir,pattern="ind")
		cols=self.metrics.copy()
		cols.append("uid")
		scores=pd.DataFrame(index=pd.RangeIndex(len(ind_dirs)),
							columns=cols)
		try:
			for ind,row_idx in zip(ind_dirs,scores.index):
				dict_meta,pro_df=get_run_files(ind)
				scores.loc[row_idx,self.metrics]=self.process_run(ind)
				scores.loc[row_idx,"uid"]=dict_meta["uid"]
		except Exception as e:
			print(scores)
			raise e
		return scores


	def process_run(self,raw_dir,save=False):
		""" Process an already imported (saved) run

			Computes the metrics for the run according to the parameters of 
			runProcess and referenceCompare. Typical use case is for sensitivity
			analysis, where we perform all the runs, save the results and then
			test against different processing methods (compare with multiple
			gaits), without performing the runs again.

			Input :
			raw_dir -- directory containing a raw (.csv) file, the saved output
			of import_raw
			Output :
			dataframe containing the processed metrics of the run, eventually
			saved to processed.csv in the raw_dir
		"""

		raws=fu.file_list(raw_dir,file_format=".csv",pattern="raw")

		raws=fu.assert_one_dim(raws,critical=False)
		if verbose:
			print("\nProcessing",raws)
		df=self.get_metrics(raws,verbose=verbose)
		if save:
			fu.assert_dir(raw_dir,should_be_empty=False)
			save_path=os.path.join(raw_dir,"processed.csv")
			print("Saving to",save_path)
			df.to_csv(save_path)
			return df
		else:
			return df.values[0][1:]
	
	def get_metrics(self):
		raise NotImplementedError("Placeholder")

	def get_fitness_from_dir(self,logdir,save=False):
		""" Imports and process all runs in logdir(s)
		"""
		if type(logdir) is not list:
			run_df,run_uid,_ = self.import_run(logdir)
			if LOG_LEVEL<=LOG_DEBUG:
				print("\n[DEBUG]Run",run_df.head(5))
			fit=self.get_fitness(run_df)
			fit["uid"]=run_uid
			if LOG_LEVEL<=LOG_INFO:
				print("\n[INFO]Fitness:\t",fit,"for run in:\n\t",logdir)
			if save:
				fit.to_csv(os.path.join(logdir,"result.csv"))
			return fit.dropna(axis='columns')
		else: # recursive
			gen_fit=pd.DataFrame()
			if LOG_LEVEL<=LOG_INFO:
				print("\n[INFO]Processing runs:\n\t",logdir,"\n Save:",save)
			for single_run in logdir:
				ind_fit=self.get_fitness_from_dir(single_run,save)
				gen_fit=gen_fit.append(	ind_fit,ignore_index=True)
			if LOG_LEVEL<=LOG_INFO:
				print("\n[INFO]All fitnesses\n",gen_fit)
			return gen_fit.dropna(axis='columns').set_index('uid')

	def import_runs_from_dir(self,logdir,save=False):
		""" Imports all runs in logdir(s)

		"""
		if type(logdir) is not list:
			run_df,run_uid,meta=self.import_run(logdir)

			for k,v in meta.items():
				if type(v) is dict:
					for k1,v1 in v.items():
						run_df[k1]=v1
				else:
					run_df[k]=v
			if LOG_LEVEL<=LOG_DEBUG:
				print("\n[DEBUG]Run",run_df.head(5))
			if save:
				run_df.to_csv(os.path.join(logdir,"raw"+run_uid+".csv"))
			return run_df
		else: # recursive
			run_list=[]
			if LOG_LEVEL<=LOG_INFO:
				print("\n[INFO]Processing runs:\n\t",logdir,"\n Save:",save)
			for single_run in logdir:
				run_list.append(self.import_runs_from_dir(single_run,save))
			return run_list

	def import_run(self,run_path,save=False,save_name="run.csv",save_path="."):
		""" Imports the run files and meta data located in directory 

		 	Creates a dataframe containing all information logged in directory
		 	(from include_files), and clears the directory for a future run.
		 	In addition, the information contained in the meta data file is 
		 	return for indexing purposes 
		"""
		meta_file=fu.assert_one_dim(fu.file_list(run_path,file_format=".yaml",
			pattern="meta"),critical=True)
		dict_meta=yaml.load(open(meta_file))
		try:
			uid=dict_meta["uid"]
		except KeyError:
			print("\n[ERROR] Missing uid in",meta_file,"\nKeys are\t",
				dict_meta.keys())
			raise RuntimeError
		if LOG_LEVEL<=LOG_INFO:
			print ("\n[INFO]Importing files in",run_path,"\n")
			if save:
				print("\n[INFO]Saving as",save_name," in",save_path,"\n")

		run_df=self.import_raw(run_path)

		if save:
			run_df.to_csv(os.path.join(save_path,save_name))

		yamls=fu.file_list(run_path,file_format=".yaml")
		csvs=fu.file_list(run_path,file_format=".csv")
		if LOG_LEVEL<=LOG_DEBUG:
			print("\n[DEBUG]Removing\n",yamls,"\n",csvs)
		if yamls:
			os.unlink(yamls[0])
		if csvs:
			os.unlink(csvs[0])
		return run_df,uid,dict_meta

class CppRunProcess(runProcess):
	"""  Run processing for Cpp implementation of the controller

		Handles the logging format of the Cpp implementation of the reflex
		controller, as well as the available data to compute run metrics and
		their associated objective functions (fitness)
	"""

	include_files=[	"distance1",
					"energy1",
					"footfall1",
					"joints_angle1"]
	map_rename={'joints_angle1_ANKLE_LEFT':'ankle_left',
				'joints_angle1_HIP_LEFT':'hip_left',
				'joints_angle1_KNEE_LEFT':'knee_left',
				'joints_angle1_ANKLE_RIGHT':'ankle_right',
				'joints_angle1_HIP_RIGHT':'hip_right',
				'joints_angle1_KNEE_RIGHT':'knee_right'}
	def __init__(self,args):
		super(CppRunProcess,self).__init__(args)
	def get_metrics(self,raw_df):
		""" Returns a dataframe containing the computed metrics from run files.
		"""

		if LOG_LEVEL<=LOG_DEBUG:
			print("\n[DEBUG]Raw\n",raw_df)
		metrics=pd.DataFrame(["value"])
		metrics["maxtime"]=len(raw_df.index)*TIME_STEP
		metrics["distance"]=raw_df["distance1_distance"].max()

		metrics["mean_speed"]=metrics["distance"]/metrics["maxtime"]
		metrics["energy"]=raw_df["energy1_energy"].max()
		metrics["energy_to_dist"]=metrics["energy"]/metrics["distance"]

		if self.ref is not None:
			corr_dist_dct=self.ref.get_corr(raw_df)
			for met,vals in corr_dist_dct.items():
				metrics["cor_"+met]=vals[0]
				metrics["rms_"+met]=vals[1]
		return metrics
		
	def get_fitness(self,raw_df,keep_metrics=False):
		""" Returns a dataframe containing run metrics and fitnesses 
		
		"""
		metrics=self.get_metrics(raw_df)
		if keep_metrics:
			fit_df=metrics
		else:
			fit_df=pd.DataFrame(index=pd.RangeIndex(1))
		if LOG_LEVEL<=LOG_DEBUG:
			print("\n[DEBUG]Metrics\n",metrics)
		if metrics.maxtime.values<19.0:
			fit_df["fit_stable"]=-1000
		else:
			fit_df["fit_stable"]=1
		fit_df["fit_energy"]= - metrics["energy_to_dist"]/10 

		if self.ref is None: 
			"""no way of comparing to a reference stride 
			(issue in referenceCompare initialization)
			"""
			if LOG_LEVEL<=LOG_WARNING:
				print("[WARNING]No comparison for fitness")
			return fit_df
		fit_df["fit_cor"]= 100 *(metrics.filter(like="cor").sum(1).values[0])/\
								len(metrics.filter(like="cor").columns)			
		#Total correlation with reference (%)
		fit_df["fit_rms"]= - metrics.filter(like="rms_").sum(1).values[0]
		"""arbitrary scaling, (typical range is [-50, -20], doesn't matter if 
		multi obj, must be scaled for single obj (sum)"""
		fit_df["fit_corhip"]= 100 *(metrics.filter(like="cor_hip").sum(1).values[0])/\
									len(metrics.filter(like="cor_hip").columns)				
		#Hip correlation with reference (%)
		fit_df["fit_corknee"]= 100 *(metrics.filter(like="cor_knee").sum(1).values[0])/\
									len(metrics.filter(like="cor_knee").columns)			
		#Knee correlation with reference (%)
		fit_df["fit_corankle"]= 100 *(metrics.filter(like="cor_ankle").sum(1).values[0])/\
									len(metrics.filter(like="cor_ankle").columns)			
		#Ankle correlation with reference (%)

		fit_df["fit_rmship"]= - metrics.filter(like="rms_hip").sum(1).values[0]
		#Hip correlation with reference (%)
		fit_df["fit_rmsknee"]= - metrics.filter(like="rms_knee").sum(1).values[0]
		#Knee correlation with reference (%)
		fit_df["fit_rmsankle"]= - metrics.filter(like="rms_ankle").sum(1).values[0]
		if LOG_LEVEL<=LOG_DEBUG:
			print("\n[DEBUG]Fitness\n",fit_df)

		return fit_df



	def import_raw(self,run_path):
		""" Concatenates data in run files to a single dataframe
		"""
		first_call=True
		for file_path in fu.file_list(run_path): # could filter include files before iter
			if os.path.basename(file_path) in self.include_files:
				if LOG_LEVEL<=LOG_INFO:
					print("\n[INFO]Importing\n",os.path.basename(file_path))
				data=pd.read_csv(file_path,sep=" ",error_bad_lines=False)
				file=os.path.basename(file_path) # just the file name
				file=os.path.splitext(file)[0] # remove extension
				data=data.loc[:, ~data.columns.str.match('Unnamed')] # remove columns created due to trailing seps
				data.columns=fu.concat_field(file, data.columns)
				if first_call:
					run_df=data
				else:
					run_df=pd.concat([run_df,data],axis=1)
				first_call=False
		if LOG_LEVEL<=LOG_DEBUG:
			print("\n[DEBUG]Initial columns\n",run_df.columns)

		run_df.rename(columns=self.map_rename,inplace=True)
		if LOG_LEVEL<=LOG_DEBUG:
			print("\n[DEBUG]Renamed columns\n",run_df.columns)

		return run_df


class PythonRunProcess(runProcess):
	"""  Run processing for Python implementation of the controller

		WARNING : Not complete as logging format changed during the project !


		Handles the logging format of the Python implementation of the reflex
		controller, as well as the available data to compute run metrics and
		their associated objective functions (fitness)
	"""
	include_files=None
	map_rename={'angles_ankle_l':'ankle_left',
				'angles_hip_l':'hip_left',
				'angles_knee_l':'knee_left',
				'angles_ankle_r':'ankle_right',
				'angles_hip_r':'hip_right',
				'angles_knee_r':'knee_right'}
	def __init__(self,args):
		super().__init__(args)

	def get_metrics(self,raw_df,verbose=False):
		# Metrics computed during the run
		metrics=pd.read_csv(open(objectives_file))
		# Max simulation time
		metrics["maxtime"]=max(raw_df.index*TIME_STEP)
		# Energy as the sum of all activations
		activation=raw_df.filter(like="act",axis=1)
		metrics["energy"]=activation.sum(axis=1,skipna=True)
		if self.ref is not None:
			corr_dct=self.ref.get_all_corr(raw_df)
			for met,corrval in corr_dct.items():
				metrics["cor"+met]=corrval
		if verbose:
			print(metrics)
		return metrics

	def get_fitness(self,raw_df,keep_metrics=False):
		""" Returns a dataframe containing run metrics and fitnesses 
			\todo : Complete with fitness objectives for python data
		"""
		metrics=self.get_metrics(raw_df)
		if keep_metrics:
			fit_df=metrics
		else:
			fit_df=pd.DataFrame(index=pd.RangeIndex(1))
		if LOG_LEVEL<=LOG_DEBUG:
			print("\n[DEBUG]Metrics\n",metrics)
		
		raise NotImplementedError ("Complete with fitness objectives for \
			python data")
		if LOG_LEVEL<=LOG_DEBUG:
			print("\n[DEBUG]Fitness\n",fit_df)
		return fit_df
	def import_raw(self,run_path):
		""" Concatenates data in run files to a single dataframe
			\todo : Adapt for python logging format 
		"""
		first_call=True
		for file_path in fu.file_list(run_path): # could filter include files before iter
			if os.path.basename(file_path) in self.include_files:
				if LOG_LEVEL<=LOG_INFO:
					print("\n[INFO]Importing\n",os.path.basename(file_path))
				# TODO : Complete
				raise NotImplementedError("Adapt for python logging format")
				#data=pd.read_csv(file_path,sep=" ",error_bad_lines=False)
				#file=os.path.basename(file_path) # just the file name
				#file=os.path.splitext(file)[0] # remove extension
				#data=data.loc[:, ~data.columns.str.match('Unnamed')] # remove columns created due to trailing seps
				#data.columns=fu.concat_field(file, data.columns)
				if first_call:
					run_df=data
				else:
					run_df=pd.concat([run_df,data],axis=1)
				first_call=False
		if LOG_LEVEL<=LOG_DEBUG:
			print("\n[DEBUG]Initial columns\n",run_df.columns)

		run_df.rename(columns=self.map_rename,inplace=True)
		if LOG_LEVEL<=LOG_DEBUG:
			print("\n[DEBUG]Renamed columns\n",run_df.columns)
		return run_df


def launch_process(mode,trial_dir):
	""" 

	"""
	
	ref_raw="/data/prevel/repos/biorob-semesterproject/data/winter_data/data_normal.csv" # PATH

	if mode=="cpp":
		proc=CppRunProcess(ref_raw)
	elif mode=="py":
		proc=PythonRunProcess(ref_raw)
	else:
		print("[ERROR] Wrong input:\t",mode,"\nShould be py, cpp ")
		raise KeyError
	gen_dirs=fu.dir_list(trial_dir,"param")
	nb_gen=len(gen_dirs)

	if nb_gen>0:
		prog_gen=1
		for gen_dir in gen_dirs:
			if LOG_LEVEL<=LOG_INFO:
				print("\n[INFO] Processing folder :",prog_gen,"/",nb_gen)
			
			result=proc.process_dir(gen_dir)
			save_path=os.path.join(gen_dir,'result.csv')
			result.to_csv(save_path)
			prog_gen+=1

if __name__ == '__main__':

	try:
		mode=sys.argv[1]
		trial_dir=sys.argv[2:]
	except IndexError:
		if LOG_LEVEL<=LOG_ERROR:
			print("[ERROR] Missing arguments, see doc:\n",
				launch_process.__doc__)
	else:
		try:
			launch_process(mode,trial_dir)
		except KeyError:
			if LOG_LEVEL<=LOG_ERROR:
				print("[ERROR] Incorrect arguments:",arg,"\n see doc:\n",
					launch_process.__doc__)