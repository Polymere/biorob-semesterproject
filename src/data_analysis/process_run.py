"""
Process the run csv file located at input_file_path, and generates a csv file with the processed 
metrics at output_file_path or/and plots the results
Inputs : 
	input_file_path
	output_file_path
	plot

EXAMPLE
python process_run.py \
/data/prevel/trial/result/param1/world1 \
/data/prevel/trial/result/param1/ \
all

"""
import pandas as pd
import sys
import os
import yaml
import utils.file_utils as fu
from math import sqrt
from utils.plot_utils import plot_mean_std_fill,plot_correlation_window
import data_analysis.event_detection as ed
from data_analysis.import_run import cpp_import_run
from utils.meta_utils import get_run_files
import matplotlib.pyplot as plt
TIME_STEP=1e-3 #ms

LOG_DEBUG=1
LOG_INFO=2
LOG_WARNING=3
LOG_ERROR=4


MAP_PYTHON_CPP={'angles_ankle_l':'ANKLE_LEFT',	'angles_ankle_r':'ANKLE_RIGHT',
				'angles_hip_l':'HIP_LEFT',		'angles_hip_r':'HIP_RIGHT',
				'angles_knee_l':'KNEE_LEFT',	'angles_knee_r':'KNEE_RIGHT'}

MAP_PYTHON_WINTER={'angles_ankle_l':'ankle',	'angles_ankle_r':'ankle',
				'angles_hip_l':'hip',		'angles_hip_r':'hip',
				'angles_knee_l':'knee',		'angles_knee_r':'knee'}

MAP_CPP_WINTER={'joints_angle1_ANGLE_ANKLE_LEFT':'ankle',
				'joints_angle1_ANGLE_HIP_LEFT':'hip',
				'joints_angle1_ANGLE_KNEE_LEFT':'knee'}

MAP_CPP_C3D={'joints_angle1_ANGLE_ANKLE_LEFT':'LANKLE',
				'joints_angle1_ANGLE_HIP_LEFT':'LHIP',
				'joints_angle1_ANGLE_KNEE_LEFT':'LKNEE',
				'joints_angle1_ANGLE_ANKLE_RIGHT':'RANKLE',
				'joints_angle1_ANGLE_HIP_RIGHT':'RHIP',
				'joints_angle1_ANGLE_KNEE_RIGHT':'RKNEE'}

MAP_CPP_SHORT={'joints_angle1_ANGLE_ANKLE_LEFT':'ankle_left',
				'joints_angle1_ANGLE_HIP_LEFT':'hip_left',
				'joints_angle1_ANGLE_KNEE_LEFT':'knee_left',
				'joints_angle1_ANGLE_ANKLE_RIGHT':'ankle_right',
				'joints_angle1_ANGLE_HIP_RIGHT':'hip_right',
				'joints_angle1_ANGLE_KNEE_RIGHT':'knee_right'}

C3D_KEYS=["LANKLE","LHIP","LKNEE","RANKLE","RHIP","RKNEE"]

CPP_KEYS=[	"joints_angle1_ANGLE_ANKLE_LEFT","joints_angle1_ANGLE_HIP_LEFT","joints_angle1_ANGLE_KNEE_LEFT",
			"joints_angle1_ANGLE_ANKLE_RIGHT","joints_angle1_ANGLE_HIP_RIGHT","joints_angle1_ANGLE_KNEE_RIGHT"]


SHORT_KEYS=["ankle_left","hip_left","knee_left","ankle_right","hip_right","knee_right"]
	

INCLUDE_FILES=["distance1","energy1","footfall1","joints_angle1"]

LOG_LEVEL=LOG_WARNING

def import_and_process_from_dir(result_dir,single_val=False,save=True):
	""" Result dir is the log directory or a list of log directories.
		It must contain all files in INCLUDE_FILES
		Will create a processed.csv file containing the computed fitness metrics in the 
		log directory if save is set to True
		Will return a single fitness value/list of fitnesses values if single value is set to true
		WIP -> can't ahve single_val and save true at the same time (index issue with result df)
	"""
	if type(result_dir) is not list:
		run_df=cpp_import_run(result_dir,save_to_single=False,include_files=INCLUDE_FILES)
		proc=CppRunProcess(compare_files="../../data/patient1.csv",compare_kind="c3d_to_cpp")
		if LOG_LEVEL<=LOG_DEBUG:
			print("\n[DEBUG]Run",run_df.head(5))
		fit=proc.get_fitness(run_df,single_val=single_val)
		if LOG_LEVEL<=LOG_INFO:
			print("\n[INFO]Fitness:\t",fit,"for run in:\n\t",result_dir)
		if save:
			fit.to_csv(os.path.join(result_dir,"result.csv"))
		if single_val:
			return fit.values[0]
		else:
			return fit
	else:
		fit=[]
		if LOG_LEVEL<=LOG_INFO:
			print("\n[INFO]Processing runs:\n\t",result_dir,"\n Params",single_val,save)
		for single_run in result_dir:
			fit.append(import_and_process_from_dir(single_run,single_val,save))
		if LOG_LEVEL<=LOG_INFO:
			print("\n[INFO]All fitnesses\n",fit)
		return fit

def import_and_process_from_data(data):
	# directly for data logged in input files -> see format and if possible
	raise NotImplementedError

class referenceCompare:
	kinematics_compare_file=None
	kinematics_compare_kind=None
	do_plot=False
	def __init__(self,args):
		for arg_name,arg_value in args.items():
			if hasattr(self, arg_name):
				setattr(self, arg_name, arg_value)
		if self.kinematics_compare_kind is None or self.kinematics_compare_file is None:
			if LOG_LEVEL<=LOG_WARNING:
				print("\n[WARNING]Missing arguments for kinematic compare in:\n",args,"\nIt will not be performed ! \n")
				return None

		if LOG_LEVEL<=LOG_INFO:
			print("\n[INFO]",self.__class__.__name__," initialized with\n",self.__dict__)
		ref_df=pd.read_csv(self.kinematics_compare_file)
		self.rep_strides={}
		if self.kinematics_compare_kind=="winter_to_python":
			self.set_repstrides_winter(ref_df)
			self.get_corr=self._get_all_corr_python
		elif self.kinematics_compare_kind=="cpp_to_python":
			if LOG_LEVEL<=LOG_ERROR:
				print("\n[ERROR]No longer suported with initial florin format, preprocess before (see comment below for reference)\n",)
				"""	#directions :
						#	ankle -
						#	knee +
						#	hip +
 
						contact_file=files[0]
						joints_file=files[1]
						contact=pd.read_csv(contact_file,sep=" ")
						joints=pd.read_csv(joints_file,sep=" ")
						self.set_repstrides_florin(contact, joints)
						def set_repstrides_florin(self,contact,joints):
						# ankle angle is defined in the opposed direction
							for key_gen,key_flor in MAP_MET_FLORIN.items():
								mean_ref,std_ref=ed.get_rep_var_from_contact(contact,key_flor,joints)
								mean_ref=ed.interp_gaitprcent(mean_ref,100)
								if "ankle" in key_gen: # inversed angle orientation
									mean_ref=-mean_ref
								self.rep_strides[key_gen]=mean_ref
					"""
			raise DeprecationWarning
		elif self.kinematics_compare_kind=="winter_to_cpp":
			self.set_repstrides_winter_cpp(ref_df)
			self.get_corr=self._get_all_corr_cpp
		elif self.kinematics_compare_kind=="python_to_python":
			self.set_repstrides_raw(ref_df)
			self.get_corr=self._get_all_corr_python
		elif self.kinematics_compare_kind=="c3d_to_cpp":
			self.set_repstrides_c3d_for_cpp(ref_df)
			self.get_corr=self._get_all_corr_cpp
		else:
			if LOG_LEVEL<=LOG_ERROR:
				print("\n[ERROR]Compare kind ",self.kinematics_compare_kind," not supported\n",)
			raise KeyError


	def set_repstrides_winter(self,win_df):
		"""
		hip angle is defined in the opposed direction
		"""
		for key_gen,key_win in MAP_MET_WINTER.items():
			mean_ref=ed.interp_gaitprcent(win_df[key_win],100)
			if key_win=="hip": # inversed angle orientation
				mean_ref=-mean_ref
			self.rep_strides[key_gen]=mean_ref

	def set_repstrides_c3d_for_cpp(self,win_df):
		"""
		hip angle is defined in the opposed direction
		"""
		for key_gen,key_c3d in zip(SHORT_KEYS,C3D_KEYS):#MAP_CPP_C3D.items():
			mean_ref=ed.interp_gaitprcent(win_df[key_c3d],100)
			if key_c3d=="LANKLE"or key_c3d=="RANKLE":  # inversed angle orientation
				mean_ref= - mean_ref
			self.rep_strides[key_gen]=mean_ref*(3.1415/180)

	def set_repstrides_winter_cpp(self,win_df):
		"""
		hip angle is defined in the same direction !
		"""
		for key_gen,key_win in MAP_CPP_WINTER.items():
			mean_ref=ed.interp_gaitprcent(win_df[key_win],100)
			if "ankle" in key_win: # inversed angle orientation
				mean_ref=-mean_ref
			if "hip" in key_win: # inversed angle orientation
				mean_ref=-mean_ref
			self.rep_strides[key_gen]=mean_ref



	def set_repstrides_raw(self,ref_df):
		angles=ref_df.filter(like="angle")
		for key_gen in angles.keys():
			mean_ref,std_ref=ed.get_mean_std_stride(angles,key_gen,stride_choice="repmax")
			mean_ref=ed.interp_gaitprcent(mean_ref,100)
			self.rep_strides[key_gen]=mean_ref

	def get_corr(self,cmp_df):
		if LOG_LEVEL<=LOG_ERROR:
			print("\n[ERROR]See class initialization\n")
		raise NotImplementedError

	def _get_all_corr_python(self,cmp_df):
		all_cor={}
		for met in self.rep_strides.keys():
			mean_cur,std_cur=ed.get_mean_std_stride(cmp_df,met,stride_choice="repmax")
			all_cor[met]=mean_cur.corr(self.rep_strides[met])
		return all_cor

	def _get_all_corr_cpp(self,cmp_df):

		corr_dist={}
		contact=cmp_df.filter(like="footfall1")
		contact.columns=["left","right"]
		for met in self.rep_strides.keys():
			mean_cur,std_cur=ed.get_rep_var_from_contact(contact,met,cmp_df,how="strike_to_liftoff")
			correl=mean_cur.corr(self.rep_strides[met])
			dist=sqrt(((self.rep_strides[met]-mean_cur)**2).sum())
			corr_dist[met]=[correl,dist]
			if self.do_plot:
				ax=plot_correlation_window(mean_cur,self.rep_strides[met],10)
				plot_mean_std_fill(mean_cur, std_cur, "b",ax)
				plot_mean_std_fill(self.rep_strides[met], None, "k",ax)
				tit=met+"(cor:"+str(round(correl,1))+", rms:"+str(round(dist,1))+")"
				plt.title(tit)
				plt.show()
				#print("Correlation:\t",round(correl,3),"\nDistance\t",round(dist,3))
		return corr_dist



class runProcess:
	metrics=["maxtime","corankle","corknee"]


	def __init__(self,args):
		for arg_name,arg_value in args.items():
			if hasattr(self, arg_name):
				setattr(self, arg_name, arg_value)
		if LOG_LEVEL<=LOG_INFO:
			print("\n[INFO]Run processor ",self.__class__.__name__," initialized with\n",self.__dict__)

		self.ref=referenceCompare(args)

	def process_gen(self,gen_dir):
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


	def process_run(self,ind_dir,save=False):

		raws=fu.file_list(ind_dir,file_format=".csv",pattern="raw")

		raws=fu.assert_one_dim(raws,critical=False)
		if verbose:
			print("\nProcessing",raws)
		df=self.get_metrics(raws,verbose=verbose)
		if save:
			fu.assert_dir(ind_dir,should_be_empty=False)
			save_path=os.path.join(ind_dir,"processed.csv")
			print("Saving to",save_path)
			df.to_csv(save_path)
			return df
		else:
			#print("\n[DEBUG] Process run",df.values,"\n",df.values[0][1:-1])
			return df.values[0][1:]
	def get_metrics(self):
		raise NotImplementedError

	def get_fitness_from_dir(self,logdir,save=False):
		if type(logdir) is not list:
			run_df,run_uid=self.import_run(logdir)
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

class CppRunProcess(runProcess):
	fitnesses=["fit_cor","fit_energy","fit_stable","fit_rms"]
	include_files=[	"distance1",
					"energy1",
					"footfall1",
					"joints_angle1"]
	def __init__(self,args):
		super(CppRunProcess,self).__init__(args)
	def get_metrics(self,raw_df):

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
	def get_fitness(self,raw_df):
		metrics=self.get_metrics(raw_df)
		fit_df=pd.DataFrame(index=pd.RangeIndex(1))#,columns=self.fitnesses)
		if LOG_LEVEL<=LOG_DEBUG:
			print("\n[DEBUG]Metrics\n",metrics)
		if metrics.maxtime.values<19.0:
			fit_df["fit_stable"]=-1000
		else:
			fit_df["fit_stable"]=1
		fit_df["fit_cor"]= 100 *	(metrics.filter(like="cor").sum(1).values[0])/\
								len(metrics.filter(like="cor").columns)						#Total correlation with reference (%)
		fit_df["fit_rms"]= - metrics.filter(like="rms_").sum(1).values[0]
		fit_df["fit_energy"]= - metrics["energy_to_dist"]/10 # arbitrary scaling, (typical range is [-50, -20], doesn't matter if multi obj, must be scaled for single obj (sum)
		fit_df["fit_corhip"]= 100 *		(metrics.filter(like="cor_hip").sum(1).values[0])/\
									len(metrics.filter(like="cor_hip").columns)				#Hip correlation with reference (%)
		fit_df["fit_corknee"]= 100 *	(metrics.filter(like="cor_knee").sum(1).values[0])/\
									len(metrics.filter(like="cor_knee").columns)			#Knee correlation with reference (%)
		fit_df["fit_corankle"]= 100 *	(metrics.filter(like="cor_ankle").sum(1).values[0])/\
									len(metrics.filter(like="cor_ankle").columns)			#Ankle correlation with reference (%)

		fit_df["fit_rmship"]= - metrics.filter(like="rms_hip").sum(1).values[0]
			#Hip correlation with reference (%)
		fit_df["fit_rmsknee"]= - metrics.filter(like="rms_knee").sum(1).values[0]
				#Knee correlation with reference (%)
		fit_df["fit_rmsankle"]= - metrics.filter(like="rms_ankle").sum(1).values[0]
		if LOG_LEVEL<=LOG_DEBUG:
			print("\n[DEBUG]Fitness\n",fit_df)

		return fit_df

	def get_fitness_param(self,run_path):
		raw=self.get_raw(run_path)
		fit=self.get_fitness(raw)
	def import_run(self,run_path,do_save=False,save_name="run.csv",save_path="."):
		meta_file=fu.assert_one_dim(fu.file_list(run_path,file_format=".yaml",pattern="meta"),\
								critical=True)
		dict_meta=yaml.load(open(meta_file))
		try:
			uid=dict_meta["uid"]
		except KeyError:
			print("\n[ERROR] Missing uid in",meta_file,"\nkeys are\t",dict_meta.keys())
			raise RuntimeError
		if LOG_LEVEL<=LOG_INFO:
			print ("\n[INFO]Importing files in",run_path,"\n")
			if do_save:
				print("\n[INFO]Saving as",save_name," in",save_path,"\n")

		run_df=self.get_raw(run_path)

		if do_save:
			run_df.to_csv(os.path.join(save_path,save_name))

		yamls=fu.file_list(run_path,file_format=".yaml")
		csvs=fu.file_list(run_path,file_format=".csv")
		if LOG_LEVEL<=LOG_DEBUG:
			print("\n[DEBUG]Removing\n",yamls,"\n",csvs)
		if yamls:
			os.unlink(yamls[0])
		if csvs:
			os.unlink(csvs[0])
		return run_df,uid
	def get_raw(self,run_path):
		first_call=True
		for file_path in fu.file_list(run_path): # could filter before iter
			if os.path.basename(file_path) in self.include_files:
				if LOG_LEVEL<=LOG_INFO:
					print("\n[INFO]Importing\n",os.path.basename(file_path))
				data=pd.read_csv(file_path,sep=" ",error_bad_lines=False)
				file=os.path.basename(file_path) # just the file name
				file=os.path.splitext(file)[0] # remove extension
				data=data.loc[:, ~data.columns.str.match('Unnamed')] # remove columns due to trailing seps
				data.columns=fu.concat_field(file, data.columns)
				if first_call:
					run_df=data
				else:
					run_df=pd.concat([run_df,data],axis=1)
				first_call=False
		if LOG_LEVEL<=LOG_DEBUG:
			print("\n[DEBUG]Initial columns\n",run_df.columns)
		run_df.rename(columns=MAP_CPP_SHORT,inplace=True)
		if LOG_LEVEL<=LOG_DEBUG:
			print("\n[DEBUG]Renamed columns\n",run_df.columns)

		return run_df






class PythonRunProcess(runProcess):
	def __init__(self,compare_files):
		compare_kind="winter_to_python"
		super().__init__(compare_kind,compare_files)
	def get_metrics(self,raw_file,verbose=False):
		raw_df=pd.read_csv(open(raw_file))
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



def main(mode,run_dir):
	
	ref_raw="/data/prevel/repos/biorob-semesterproject/data/winter_data/data_normal.csv"
	if mode=="cpp":
		proc=CppRunProcess(ref_raw)
	elif mode=="py":
		proc=PythonRunProcess(ref_raw)
	elif mode=="cpp_import_process":
		fit=import_and_process_from_dir(run_dir)
		return fit
	else:
		print("[ERROR] Wrong input:\t",mode,"\nShould be py, cpp or cpp_import_process")
		raise ValueError
	gen_dirs=fu.dir_list(run_dir,"param")
	nb_gen=len(gen_dirs)

	if nb_gen>0:
		prog_gen=1
		for gen_dir in gen_dirs:
			print("\n[PROGRESS] Processing gen :",prog_gen,"/",nb_gen)
			proc.process_gen(gen_dir)
			prog_gen+=1
	else:
		raise DeprecationWarning
		ind_dirs=fu.dir_list(run_dir,pattern="ind")
		if len(ind_dirs)>0:
			for ind in ind_dirs:
				process_cpp(ind,ref_cmp=ref)
		else:
			fl=fu.file_list(run_dir,file_format=".csv")
			if len(fl)>=2:
				process_cpp(run_dir,ref_cmp=ref)

if __name__ == '__main__':
	mode=sys.argv[1]
	param=sys.argv[2:]
	main(mode,param)
	



# DEPRECATED METHODS
def metric_df(raw_file,objectives_file,ref_cmp=None,verbose=True):
	raise DeprecationWarning
	raw_df=pd.read_csv(open(raw_file))
	# Metrics computed during the run
	metrics=pd.read_csv(open(objectives_file))
	# Max simulation time
	metrics["maxtime"]=max(raw_df.index*TIME_STEP)
	# Energy as the sum of all activations
	activation=raw_df.filter(like="act",axis=1)
	metrics["energy"]=activation.sum(axis=1,skipna=True)
	if ref_cmp is not None:
		corr_dct=ref_cmp.get_all_corr(raw_df)
		for met,corrval in corr_dct.items():
			metrics["cor"+met]=corrval
	if verbose:
		print(metrics)
	return metrics

def metric_dct_cpp(raw_file,ref_cmp=None,verbose=False):
	raise DeprecationWarning
	raw_df=pd.read_csv(open(raw_file))
	# Metrics computed during the run

	metrics=pd.DataFrame(["value"])
	metrics["maxtime"]=max(raw_df.index*TIME_STEP)
	metrics["distance"]=raw_df["distance1_distance"].max()
	# Energy as the sum of all activations
	metrics["mean_speed"]=metrics["distance"]/metrics["maxtime"]
	metrics["energy"]=raw_df["energy1_energy"].max()
	metrics["energy_to_dist"]=metrics["energy"]/metrics["distance"]
	#activation=raw_df.filter(like="act",axis=1)
	#metrics["energy"]=activation.sum(axis=1,skipna=True)
	if ref_cmp is not None:
		corr_dct=ref_cmp.get_all_corr_cpp(raw_df)
		for met,corrval in corr_dct.items():
			metrics["cor"+MAP_CPP_WINTER[met]]=corrval
	if verbose:
		print(metrics)
	return metrics

def process_cpp(ind_dir,ref_cmp=None,save=True,verbose=False):
	raise DeprecationWarning
	raws=fu.file_list(ind_dir,file_format=".csv",pattern="raw")

	raws=fu.assert_one_dim(raws,critical=False)
	if verbose:
		print("\nProcessing",raws)
	df=metric_dct_cpp(raws,ref_cmp=ref_cmp,verbose=verbose)
	if save:
		fu.assert_dir(ind_dir,should_be_empty=False)
		save_path=os.path.join(ind_dir,"processed.csv")
		print("Saving to",save_path)
		df.to_csv(save_path)
		return df
	else:
		return df

def process(ind_dir,ref_cmp=None,save=True):
	raise DeprecationWarning
	raws=fu.file_list(ind_dir,file_format=".csv",pattern="raw")
	objectives=fu.file_list(ind_dir,file_format=".csv",pattern="objectives")	

	raws=fu.assert_one_dim(raws,critical=False)
	objectives=fu.assert_one_dim(objectives,critical=False)
	df=metric_df(raws, objectives,ref_cmp=ref_cmp)
	if save:
		fu.assert_dir(ind_dir,should_be_empty=False)
		save_path=os.path.join(ind_dir,"processed.csv")
		df.to_csv(save_path)
	else:
		return df