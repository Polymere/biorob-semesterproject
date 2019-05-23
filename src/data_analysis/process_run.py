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
import numpy as np
import sys
import yaml
import os
import utils.file_utils as fu
import matplotlib.pyplot as plt
import data_analysis.event_detection as ed
from utils.plot_utils import plot_mean_std_fill

TIME_STEP=1e-3 #ms
SAVEPATH="/data/prevel/runs/figures"

DIR_REF_CPP="../../data/ref_cpp/"
REF_FORMAT='florin'
MAP_PYTHON_CPP={'angles_ankle_l':'ANKLE_LEFT',	'angles_ankle_r':'ANKLE_RIGHT',
				'angles_hip_l':'HIP_LEFT',		'angles_hip_r':'HIP_RIGHT',
				'angles_knee_l':'KNEE_LEFT',	'angles_knee_r':'KNEE_RIGHT'}

MAP_PYTHON_WINTER={'angles_ankle_l':'ankle',	'angles_ankle_r':'ankle',
				'angles_hip_l':'hip',		'angles_hip_r':'hip',
				'angles_knee_l':'knee',		'angles_knee_r':'knee'}

MAP_CPP_WINTER={'joints_angle1_ANGLE_ANKLE_LEFT':'ankle','joints_angle1_ANGLE_HIP_LEFT':'hip','joints_angle1_ANGLE_KNEE_LEFT':'knee'}


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
			winter_file=files[0]
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
			ref_file=files[0]
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
				print("opposed ankle")
				mean_ref=-mean_ref
			if "hip" in key_win: # inversed angle orientation
				print("opposed hip")
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
				print("opposed ankle")
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

def metric_df(raw_file,objectives_file,ref_cmp=None,verbose=True):
	#raise DeprecationWarning
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
	#raise DeprecationWarning
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
	#raise DeprecationWarning
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
	#raise DeprecationWarning
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

def get_run_files(ind_path,verbose=False):
	meta_file=fu.assert_one_dim(fu.file_list(ind_path,file_format=".yaml",pattern="meta"),\
								critical=True,verbose=verbose)
	dict_meta=yaml.load(open(meta_file))

	processed_file=fu.assert_one_dim(fu.file_list(ind_path,file_format=".csv",pattern="processed"),\
									critical=False,verbose=verbose)
	pro_df=pd.read_csv(processed_file)

	return dict_meta,pro_df
	
def get_label(meta,count=None):
	try:
		meta_params=meta["opt_params"]
	except KeyError:
		print(meta.keys())
		raise KeyError
	if "label" in meta.keys():
		return meta["label"]

	elif len(meta_params)==1:
		lab=list(meta_params.values())[0]
		return lab
	elif len(meta_params)>1 and count is not None:
		return count
def get_param_value(meta,param_name=None):
	if param_name is None:
		try:
			meta_params=meta["opt_params"]
		except KeyError:
			print(meta.keys())
			raise KeyError
		#if len(meta_params)==1:
		return list(meta_params.keys()),list(meta_params.values())
		#if len(meta.keys())==1:
		#	param_name=list(meta.keys())[0]
		#	return [param_name],[meta[param_name]] 
		#elif len(meta_params)>1:
		#	return list(meta_params.keys()),list(meta_params.values())
		#else:
		#	print(meta)
		#	raise ValueError
	else: # not modified 
		raise NotImplementedError
		if type(param_name)==list:
			vals=[]
			for param in param_name:
				vals.append(meta[param])
			return vals
		elif param_name in meta.keys():
			return meta[param_name]
		else:
			raise('ValueError',meta,param_name)

def get_metric_value(proc,metric,what="max_value"):
	if metric not in proc.columns:
		print("Band aid for correlation with ref \n ")
		return 1
	if what=="max_value":
		return proc[metric].max() # SEE dropna syntax
	elif what=="mean_value":
		return proc[metric].mean() # SEE dropna syntax

def get_discriminant(proc, metric, params, what="geq_thr"):
	if what=="geq_thr":
		thr=np.float(params)
		return (proc[metric].values[0]>=thr)
	else:
		print("get_discriminant args",proc,metric,params,what)
		return None
def export_meta_params(indiv_dirs,ref_dir, disc_name, disc_params,save_path=None):
	#df=pd.DataFrame()
	count=1
	for ind in indiv_dirs:
		#print("\n",os.path.basename(ind))
		ind_id=os.path.basename(ind)
		meta, proc=get_run_files(ind)
		lab = get_label(meta,count)
		#met = get_metric_value(proc, metric)
		pname, param_value = get_param_value(meta)
		disc = get_discriminant(proc, disc_name, disc_params)

		dct_row={}
		for i in range(len(pname)):
			print(pname)
			dct_row["meta_"+pname[i]]=param_value[i]
		
		for metric_name in proc.columns:
			#print(proc[metric_name])
			dct_row["metric_"+metric_name]=proc[metric_name].values[0]
		dct_row["disc"]=disc
		dct_row["label"]=lab
		if count==1:
			df=pd.DataFrame.from_records([dct_row])
		else:
			df=df.append(dct_row,ignore_index=True)

		count+=1
	print(df)
	if save_path is not None:
		df.to_csv(save_path)

class runProcess:
	metrics=["maxtime","corankle","corknee"]

	def __init__(self,compare_kind,compare_files):
		if compare_files is not None:
			#print("\n[DEBUG] Init ref run process",compare_kind,compare_files)
			self.ref=reference_compare(compare_kind,compare_files)
			
		else:
			#print("\n[DEBUG] NO init ref run process")
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
				#print("\n[DEBUG] UID\n",dict_meta["uid"])
				#print("\n[DEBUG] ILOC\n",scores.loc[row_idx])
				#print("\n[DEBUG] slice\n",scores.loc[row_idx,self.metrics])
				#print("\n[DEBUG] result\n",self.process_run(ind))
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
	def __init__(self,compare_files):
		#print("\n[DEBUG]Init CPP process",compare_files)
		compare_kind="winter_to_cpp"

		super().__init__(compare_kind,compare_files)
	def get_metrics(self,raw_file,verbose=True):

		#print("\nGetting metrics CPP")
		raw_df=pd.read_csv(open(raw_file))
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



def plot_with_success(indiv_dirs, metric, disc_name, disc_params, ref_dir=None,what="max_value",save_path=None):
	plot_qd_lst=[] # Contains (label,metric_value,param_value,discriminant)

	for ind in indiv_dirs:
		#print("\n",os.path.basename(ind))
		ind_id=os.path.basename(ind)
		meta, proc =get_run_files(ind)
		lab = get_label(meta)
		met = get_metric_value(proc, metric)
		pname, param_value = get_param_value(meta)
		disc = get_discriminant(proc, disc_name, disc_params)
		if disc:
			lab=str(round(lab,2))+"\n"+str(ind_id)
		else:
			lab=round(lab, 1)
		plot_qd_lst.append((lab, met, param_value, disc))
	if ref_dir is not None:
		meta, proc = get_run_files(ref_dir)
		lab = get_label(meta)
		met = get_metric_value(proc, metric)
		param_value = get_param_value(meta, param_name=pname)
		disc = get_discriminant(proc, disc_name, disc_params)
		plot_qd_lst.append((lab,  met,  param_value,  disc))
	sorted_qd_lst = sorted(plot_qd_lst,key=lambda val : val[2]) # sort by ascending metric value
	fig_count = 0
	fig=plt.figure(figsize=(len(sorted_qd_lst)*1.5,10))
	ax=plt.axes()
	min_val=np.inf
	max_val=-np.inf
	label_lst=[]
	for qd in sorted_qd_lst:
		#print ("\n",qd[3])
		fig_count += 1
		label_lst.append(qd[0])
		if qd[3]:
			if qd[1]<min_val:
				min_val=qd[1]
			if qd[1]>max_val:
				max_val=qd[1]
			ax.bar(fig_count, qd[1])
		else:
			ax.plot(fig_count,0,marker='x', markersize=3, color="red")
	plt.xlabel(pname)
	if max_val>0:
		ylim_up=max_val*1.05
	else:
		ylim_up=max_val*0.8

	if min_val<0:
		ylim_low=min_val*1.2
	else:
		ylim_low=min_val*0.8
	#print("Values:\t",min_val,max_val)
	#print("Limits:\t",ylim_low,ylim_up)
	plt.ylim([ylim_low,ylim_up])

	xt=np.arange(1,fig_count+1)
	plt.xticks(xt,label_lst)
	plt.ylabel(metric+" "+what)
	plt.grid(axis='y')
	if save_path is None:
		plt.show()
	else:
		
		plt.rcParams.update({'font.size': 22})
		fig_name=pname[0]+what+metric+".png"
		p=os.path.join(save_path,fig_name)
		#plt.tight_layout()
		plt.savefig(p,dpi=300,transparent=False)
		plt.close()



if __name__ == '__main__':
	mode=sys.argv[1]
	param=sys.argv[2:]
	print(mode)

	if mode=="process_and_save":
		"""
		python process_run.py process_and_save /data/prevel/runs/094_14:44 (ref_dir)
		"""
		run_dir=param[0]
		ref=None
		if len(param)>1:
			"""
		python process_run.py process_and_save /data/prevel/runs ... winter /data/prevel/comparisons/winter_data/data_normal.csv
			"""
			ref_kind=param[1]
			ref_raw=param[2:]
			
			ref=reference_compare(ref_kind,files=ref_raw)
			
		gen_dirs=fu.dir_list(run_dir,"param")
		if len(gen_dirs)>0:
			for gen_dir in gen_dirs:
				ind_dirs=fu.dir_list(gen_dir,pattern="ind")
				for ind in ind_dirs:
					process(ind,ref_cmp=ref)
		else:
			ind_dirs=fu.dir_list(run_dir,pattern="ind")
			if len(ind_dirs)>0:
				for ind in ind_dirs:
					process(ind,ref_cmp=ref)
			else:
				fl=fu.file_list(run_dir,file_format=".csv")
				if len(fl)>=2:
					process(run_dir,ref_cmp=ref)

	elif mode=="process_and_save_cpp":
		"""
		python process_run.py process_and_save_cpp /data/prevel/runs/094_14:44 (ref_dir)
		"""
		run_dir=param[0]
		ref_kind="winter_to_cpp"
		ref_raw=["/data/prevel/repos/biorob-semesterproject/data/winter_data/data_normal.csv"]
		
		ref=reference_compare(ref_kind,files=ref_raw)
			
		gen_dirs=fu.dir_list(run_dir,"param")
		nb_gen=len(gen_dirs)

		if nb_gen>0:
			prog_gen=1
			for gen_dir in gen_dirs:
				ind_dirs=fu.dir_list(gen_dir,pattern="ind")
				nb_ind=len(ind_dirs)
				prog_ind=1
				for ind in ind_dirs:
					print("Gen",prog_gen,"/",nb_gen,"\tInd",prog_ind,"/",nb_ind)

					process_cpp(ind,ref_cmp=ref)
					prog_ind+=1
				prog_gen+=1
		else:
			ind_dirs=fu.dir_list(run_dir,pattern="ind")
			if len(ind_dirs)>0:
				for ind in ind_dirs:
					process_cpp(ind,ref_cmp=ref)
			else:
				fl=fu.file_list(run_dir,file_format=".csv")
				if len(fl)>=2:
					process_cpp(run_dir,ref_cmp=ref)
	
	elif mode=="plot_with_success":
		"""
		python process_run.py \
		plot_with_success \
		../../data/da_sample/param \
		../../data/da_sample/reference \
		velocity
		4000
		"""


		run_dir = param[0]
		#ref_dir=param[1]
		metrics=["maxtime",	"distance",	"mean_speed",	"energy","energy_to_dist","corankle","corhip","corknee"]
		disc_params=[0.0, 19.0, 19.0, 19.0, 19.0, 19.0,19.0,19.0]
		do_save=True
		gen_dirs=fu.dir_list(run_dir,"param")

		for metric,disc_param in zip(metrics,disc_params):
			if do_save:
				save_path=os.path.join(run_dir,"figures",metric)
				fu.assert_dir(save_path)
			else:
				save_path=None
			if len(gen_dirs)>0:
				for gen_dir in gen_dirs:
					ind_dirs=fu.dir_list(gen_dir,pattern="ind")
					plot_with_success(	ind_dirs,metric,disc_name='maxtime',
										disc_params=disc_param,what="max_value",save_path=save_path)
			else:
				ind_dirs=fu.dir_list(run_dir,pattern="ind")
				plot_with_success(	ind_dirs,metric,disc_name='maxtime',
									disc_params=disc_param,what="max_value",save_path=save_path)
							
	elif mode=="export_joined_df":
		"""
		python process_run.py export_joined_df /data/prevel/runs/dual_modrange_SOL_TA/param1/ 4000 /data/prevel/runs/dual_modrange_SOL_TA/met.csv
		"""
		run_dir = param[0]
		ref_dir=None
		disc_param=param[1]
		save_path=param[2]
		gen_dirs=fu.dir_list(run_dir,"param")
		if len(gen_dirs)>0:
			for gen_dir in gen_dirs:
				ind_dirs=fu.dir_list(gen_dir,pattern="ind")
				export_meta_params(	ind_dirs,ref_dir, disc_name='maxtime', 
									disc_params=disc_param,save_path=save_path)
		else:
			ind_dirs=fu.dir_list(run_dir,pattern="ind")
			export_meta_params(ind_dirs,ref_dir, disc_name='maxtime', 
								disc_params=disc_param,save_path=save_path)