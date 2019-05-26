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
import utils.file_utils as fu
import data_analysis.event_detection as ed
from data_analysis.import_run import cpp_import_run
from utils.meta_utils import get_run_files

TIME_STEP=1e-3 #ms

MAP_PYTHON_CPP={'angles_ankle_l':'ANKLE_LEFT',	'angles_ankle_r':'ANKLE_RIGHT',
				'angles_hip_l':'HIP_LEFT',		'angles_hip_r':'HIP_RIGHT',
				'angles_knee_l':'KNEE_LEFT',	'angles_knee_r':'KNEE_RIGHT'}

MAP_PYTHON_WINTER={'angles_ankle_l':'ankle',	'angles_ankle_r':'ankle',
				'angles_hip_l':'hip',		'angles_hip_r':'hip',
				'angles_knee_l':'knee',		'angles_knee_r':'knee'}

MAP_CPP_WINTER={'joints_angle1_ANGLE_ANKLE_LEFT':'ankle',
				'joints_angle1_ANGLE_HIP_LEFT':'hip',
				'joints_angle1_ANGLE_KNEE_LEFT':'knee'}

INCLUDE_FILES=["distance1","energy1","footfall1","joints_angle1"]

def import_and_process_from_dir(result_dir,save=True,verbose=False):
	if type(result_dir) is not list:
		run_df=cpp_import_run(result_dir,save_to_single=False,include_files=INCLUDE_FILES)
		proc=CppRunProcess(compare_files="../../data/winter_data/data_normal.csv")
		if verbose:
			print("\n[DEBUG]Run",run_df.head(5))
		fit=proc.get_fitness(run_df)
		if verbose:
			print("\n[INFO]Fitness:\t",fit,"for run in:\n\t",result_dir)
		if save:
			fit.to_csv(os.path.join(result_dir,"result.csv"))
		return fit
	else:
		fit=[]
		if verbose:
			print("\n[INFO]Processing runs:\n\t",result_dir)
		for single_run in result_dir:
			fit.append(import_and_process_from_dir(single_run))
		return fit

def import_and_process_from_data(data):
	# directly for data logged in input files -> see format and if possible
	raise NotImplementedError

class reference_compare:
	rep_strides={}

	def __init__(self,kind,files=None):
		if kind=='cpp_to_python': # cpp to raw  !
			""" directions :
				ankle -
				knee +
				hip +
			""" 
			contact_file=files[0]
			joints_file=files[1]
			contact=pd.read_csv(contact_file,sep=" ")
			joints=pd.read_csv(joints_file,sep=" ")
			self.set_repstrides_florin(contact, joints)
		elif kind=='winter_to_python': # winter to raw (python) !
			""" directions :
				ankle + 
				knee +
				hip -
			""" 
			winter_file=files
			win_df=pd.read_csv(winter_file)
			self.set_repstrides_winter(win_df)

		elif kind=='winter_to_cpp': # winter to cpp (florin) !
			""" directions :
				ankle -
				knee +
				hip -
			""" 
			winter_file=files
			win_df=pd.read_csv(winter_file)
			self.set_repstrides_winter_cpp(win_df)
		elif kind=='python_to_python': # raw to raw (florin) !
			""" directions :
				ankle + 
				knee +
				hip +
			""" 
			ref_file=files
			ref_df=pd.read_csv(ref_file)
			self.set_repstrides_raw(ref_df)

	def set_repstrides_winter(self,win_df):
		"""
		hip angle is defined in the opposed direction
		"""
		for key_gen,key_win in MAP_MET_WINTER.items():
			mean_ref=ed.interp_gaitprcent(win_df[key_win],100)
			if key_win=="hip": # inversed angle orientation
				mean_ref=-mean_ref
			self.rep_strides[key_gen]=mean_ref

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

	def set_repstrides_florin(self,contact,joints):
		"""
		ankle angle is defined in the opposed direction
		"""
		for key_gen,key_flor in MAP_MET_FLORIN.items():
			mean_ref,std_ref=ed.get_rep_var_from_contact(contact,key_flor,joints)
			mean_ref=ed.interp_gaitprcent(mean_ref,100)
			if "ankle" in key_gen: # inversed angle orientation
				mean_ref=-mean_ref
			self.rep_strides[key_gen]=mean_ref

	def set_repstrides_raw(self,ref_df):
		angles=ref_df.filter(like="angle")
		for key_gen in angles.keys():
			mean_ref,std_ref=ed.get_mean_std_stride(angles,key_gen,stride_choice="repmax")
			mean_ref=ed.interp_gaitprcent(mean_ref,100)
			self.rep_strides[key_gen]=mean_ref

	def get_corre(self,cmp_df,met):
		mean_cur,std_cur=ed.get_mean_std_stride(cmp_df,met,stride_choice="repmax")
		return mean_cur.corr(self.rep_strides[met])

	def get_all_corr(self,cmp_df):
		all_cor={}
		for met in self.rep_strides.keys():
			mean_cur,std_cur=ed.get_mean_std_stride(cmp_df,met,stride_choice="repmax")
			all_cor[met]=mean_cur.corr(self.rep_strides[met])
		return all_cor

	def get_all_corr_cpp(self,cmp_df):
		all_cor={}
		for met in self.rep_strides.keys():
			mean_cur,std_cur=ed.get_rep_var_from_contact(contact_df(cmp_df),met,cmp_df)
			all_cor[met]=mean_cur.corr(self.rep_strides[met])
		return all_cor

def contact_df(full_df):
	contact=full_df.filter(like="footfall1")
	contact.columns=["left","right"]
	return contact


class runProcess:
	metrics=["maxtime","corankle","corknee"]

	def __init__(self,compare_kind,compare_files):
		if compare_files is not None:
			self.ref=reference_compare(compare_kind,compare_files)
		else:
			print("\n[WARNING] NO init ref run process")
			self.ref=None
		
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


	def process_run(self,ind_dir,save=False,verbose=False):

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

class CppRunProcess(runProcess):
	metrics=[	"maxtime",
				"distance",
				"mean_speed",
				"energy",
				"energy_to_dist",
				"corankle",
				"corhip",
				"corknee"]
	fitnesses=["fit_cor","fit_stable"]
	def __init__(self,compare_files):
		#print("\n[DEBUG]Init CPP process",compare_files)
		compare_kind="winter_to_cpp"

		super().__init__(compare_kind,compare_files)
	def get_metrics(self,raw_df,verbose=True):

		#print("\nGetting metrics CPP")
		#if os.path.isfile(raw_df):
		#	raw_df=pd.read_csv(open(raw_df))
		# Metrics computed during the run
		metrics=pd.DataFrame(["value"])
		metrics["maxtime"]=max(raw_df.index*TIME_STEP)
		metrics["distance"]=raw_df["distance1_distance"].max()

		metrics["mean_speed"]=metrics["distance"]/metrics["maxtime"]
		metrics["energy"]=raw_df["energy1_energy"].max()
		metrics["energy_to_dist"]=metrics["energy"]/metrics["distance"]

		if self.ref is not None:
			corr_dct=self.ref.get_all_corr_cpp(raw_df)
			for met,corrval in corr_dct.items():
				metrics["cor"+MAP_CPP_WINTER[met]]=corrval

		#print(metrics)
		return metrics
	def get_fitness(self,raw_df,verbose=False):
		metrics=self.get_metrics(raw_df)
		fit_df=pd.DataFrame(index=pd.RangeIndex(1),columns=self.fitnesses)
		if verbose:
			print("\n[DEBUG]Metrics\n",metrics)
		if metrics.maxtime.values<19.0:
			fit_df["fit_stable"]=-1000
		else:
			fit_df["fit_stable"]=1
		fit_df["fit_cor"]=metrics.filter(like="cor").sum(1).values[0]
		if verbose:
			print("\n[DEBUG]Fitness\n",fit_df)
		return fit_df

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
		return
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