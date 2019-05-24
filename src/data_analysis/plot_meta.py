import pandas as pd
import numpy as np
import sys
import yaml
import os
import utils.file_utils as fu
import matplotlib.pyplot as plt

from utils.plot_utils import plot_mean_std_fill

TIME_STEP=1e-3 #ms
SAVEPATH="/data/prevel/runs/figures"

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


def get_param_value(meta,param_name=None):
	if param_name is None:
		try:
			meta_params=meta["opt_params"]
		except KeyError:
			print(meta.keys())
			raise KeyError
		return list(meta_params.keys()),list(meta_params.values())

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
	if mode=="plot_with_success":
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