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
TIME_STEP=1.0 #ms
SAVEPATH="/data/prevel/runs/figures"

DIR_REF_CPP="../../data/ref_cpp/"
REF_FORMAT='florin'
MAP_MET_FLORIN={'angles_ankle_l':'ANKLE_LEFT',	'angles_ankle_r':'ANKLE_RIGHT',
				'angles_hip_l':'HIP_LEFT',		'angles_hip_r':'HIP_RIGHT',
				'angles_knee_l':'KNEE_LEFT',	'angles_knee_r':'KNEE_RIGHT'}

MAP_MET_WINTER={'angles_ankle_l':'ankle',	'angles_ankle_r':'ankle',
				'angles_hip_l':'hip',		'angles_hip_r':'hip',
				'angles_knee_l':'knee',		'angles_knee_r':'knee'}

class reference_compare:
	rep_strides={}

	def __init__(self,kind,files=None):
		if kind=='florin':
			contact_file=files[0]
			joints_file=files[1]
			contact=pd.read_csv(contact_file,sep=" ")
			joints=pd.read_csv(joints_file,sep=" ")
			self.set_repstrides_florin(contact, joints)
		elif kind=='winter':
			winter_file=files[0]
			win_df=pd.read_csv(winter_file)
			self.set_repstrides_winter(win_df)
		elif kind=='raw':
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




def metric_df(raw_file,objectives_file,ref_cmp=None,verbose=True):

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

def process(ind_dir,ref_cmp=None,save=True):
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
	#print(meta)
	if "label" in meta.keys():
		return meta["label"]
	elif len(meta.keys())==1:
		single_key=list(meta.keys())[0]
		return round(meta[single_key],5)
	elif len(meta.keys())>1 and count is not None:
		return count
def get_param_value(meta,param_name=None):
	if param_name is None:
		if len(meta.keys())==1:
			param_name=list(meta.keys())[0]
			return [param_name],[meta[param_name]] 
		elif len(meta.keys())>1:
			return list(meta.keys()),list(meta.values())
		else:
			raise('ValueError',meta)
	else:
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
		

def plot_with_success(indiv_dirs, ref_dir, metric, disc_name, disc_params, what="max_value",save=True):
	plot_qd_lst=[] # Contains (label,metric_value,param_value,discriminant)

	for ind in indiv_dirs:
		print("\n",os.path.basename(ind))
		ind_id=os.path.basename(ind)
		meta, proc=get_run_files(ind)
		lab = get_label(meta)
		met = get_metric_value(proc, metric)
		pname, param_value = get_param_value(meta)
		disc = get_discriminant(proc, disc_name, disc_params)
		if disc:
			lab=str(lab)+"\n"+str(ind_id)
		else:
			lab=round(lab, 1)
		plot_qd_lst.append((lab, met, param_value, disc))

	meta, proc = get_run_files(ref_dir)
	lab = get_label(meta)
	met = get_metric_value(proc, metric)
	param_value = get_param_value(meta, param_name=pname)
	disc = get_discriminant(proc, disc_name, disc_params)
	plot_qd_lst.append((lab,  met,  param_value,  disc))
	sorted_qd_lst = sorted(plot_qd_lst,key=lambda val : val[2]) # sort by ascending metric value
	fig_count = 0
	fig=plt.figure(figsize=(len(sorted_qd_lst),10))
	ax=plt.axes()
	
	label_lst=[]
	for qd in sorted_qd_lst:
		#print ("\n",qd[3])
		fig_count += 1
		label_lst.append(qd[0])
		if qd[3]:
			ax.bar(fig_count, qd[1])
		else:
			ax.plot(fig_count,0,marker='x', markersize=3, color="red")
	plt.xlabel(pname)
	xt=np.arange(1,fig_count+1)
	plt.xticks(xt,label_lst)
	plt.ylabel(metric+" "+what)
	plt.grid(axis='y')
	if not save:
		plt.show()
	else:
		
		plt.rcParams.update({'font.size': 22})
		fig_name=pname+what+metric+".png"
		p=os.path.join(SAVEPATH,fig_name)
		#plt.tight_layout()
		plt.savefig(p,dpi=300,transparent=False)


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
		ref_dir=param[1]
		metric=param[2]
		disc_param=param[3]
		do_save=False

		gen_dirs=fu.dir_list(run_dir,"param",)
		if len(gen_dirs)>0:
			for gen_dir in gen_dirs:
				ind_dirs=fu.dir_list(gen_dir,pattern="ind")
				plot_with_success(	ind_dirs,ref_dir,metric,disc_name='maxtime',
									disc_params=disc_param,what="max_value",save=do_save)
		else:
			ind_dirs=fu.dir_list(run_dir,pattern="ind")
			plot_with_success(	ind_dirs,ref_dir,metric,disc_name='maxtime',
								disc_params=disc_param,what="max_value",save=do_save)
							
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