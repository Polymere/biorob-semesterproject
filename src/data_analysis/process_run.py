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

TIME_STEP=1.0 #ms
SAVEPATH="/data/prevel/runs/modrange_21/figures"
def compute_df(raw_file,process_params="all"):

	data_in=pd.read_csv(open(raw_file))
	data_out=pd.DataFrame(data_in.index)

	
	data_out["time"]=data_in.index*TIME_STEP
	data_out.set_index(data_out["time"],inplace=True)
	if "energy" in process_params  or process_params=="all":
		activation=data_in.filter(like="act",axis=1)
		data_out["energy"]=activation.sum(axis=1,skipna=True)
	return data_out

	#if "all_traj" in process_params or process_params=="all":
	#	for col in data_in.columns:
	#		data_out[col]=data_in[col]
	print (data_in)
	print(data_out)
def metric_df(raw_file,objectives_file):
	raw_in=pd.read_csv(open(raw_file))
	metrics=pd.read_csv(open(objectives_file))

	metrics["maxtime"]=max(raw_in.index*TIME_STEP)

	activation=raw_in.filter(like="act",axis=1)
	metrics["energy"]=activation.sum(axis=1,skipna=True)
	return metrics

def process(ind_dir):
	raws=fu.file_list(ind_dir,file_format=".csv",pattern="raw")
	objectives=fu.file_list(ind_dir,file_format=".csv",pattern="objectives")
	def assert_one_dim(lst):
		if len(lst)>1:
			print("Multiple folds/worlds, should take worst run (not implemented yet)",lst)
		return lst[0]
	

	raws=assert_one_dim(raws)
	objectives=assert_one_dim(objectives)
	df=metric_df(raws, objectives)
	save_processed(df, ind_dir)



def save_processed(df,path):
	fu.assert_dir(path,should_be_empty=False)
	save_path=os.path.join(path,"processed.csv")
	df.to_csv(save_path)
def compare_ref(raw_file,ref_file,metric,what="max_value"):
	df_raw=compute_df(raw_file,process_params=metric)
	df_ref=compute_df(ref_file,process_params=metric)
	if what=="max_value":
		if max(df_raw[metric])>=max(df_ref[metric]):
			return "geq"
		elif max(df_raw[metric])<=max(df_ref[metric]):
			return "leq"
	else:
		print("Wrong input")
		return "err"
def get_met(df,metric,what="max_value"):
	if what=="max_value":
		return df[metric].max()
	elif what=="mean_value":
		return df[metric].mean()
	elif what=="mean_std":
		print ("WIP")
		return None
def plot_versus_ref(ref_file,raw_files,metric,what="max_value"):
	df_ref=compute_df(ref_file,process_params=metric)
	fig=plt.axes()
	if what=="max_value":
		fig_count=1
		fig.bar(fig_count,max(df_ref[metric]),tick_label="reference")
		for raw_file in raw_files:
			fig_count+=1
			df_file=compute_df(raw_file,process_params=metric)
			fig.bar(fig_count,max(df_file[metric]))
	elif what=="mean_value":
		fig_count=1
		fig.bar(fig_count,mean(df_ref[metric]),tick_label="reference")
		for raw_file in raw_files:
			fig_count+=1
			df_file=compute_df(raw_file,process_params=metric)
			fig.bar(fig_count,mean(df_file[metric]))
	elif what=="mean_std":
		fig_count=1
		fig.bar(fig_count,mean(df_ref[metric]),yerr=df_ref,tick_label="reference") ## WIP
		for raw_file in raw_files:
			fig_count+=1
			df_file=compute_df(raw_file,process_params=metric)
			fig.bar(fig_count,mean(df_file[metric]))

	else:
		print("Comparison:\t", what,"\tnot implemented")
	plt.xlabel("Individual")
	#plt.xticks([])
	plt.ylabel(metric+" "+what)
	plt.show()
def plot_met(fig,df,metric,what,fig_count):
	if what=="max_value":
		fig.bar(fig_count,max(df[metric]))
	elif what=="mean_value":
		fig.bar(fig_count,df[metric].mean())
	elif what=="mean_std":
		print ("WIP")
		return
def get_kv_meta(wdir,metric,what,expected_meta=None,current_dict=None):
	"""
		Creates/appends to a dictionnary with format
		{uid : (srt_val,plot_val,label);
		...}

	"""
	if current_dict is None:
		current_dict={}
	raw_files=fu.file_list(wdir,file_format=".csv",pattern="raw")
	if len(raw_files)>1:
		print("Multiple folds/world not implemented yet,using only ",raw_files[0])
	raw_files=raw_files[0]

	meta_file=fu.file_list(wdir,file_format=".yaml",pattern="meta")
	if len(meta_file)>1:
		print("Error, should have only one metadata per individual. Found : \n")
		print(meta_file)
	meta=yaml.load(open(meta_file[0],'r'))
	if "label" in meta.keys():
		label=meta["label"]
		uid=label
	else:
		if len(meta.keys())>1:
			print(" \n Multiple non label values in meta for ",wdir)
		for key,srt_val in meta.items():
			uid=str(key)+str(srt_val)
			k=key
			label=round(srt_val, 3)
	if expected_meta is not None:
		k=expected_meta

	df_file=compute_df(raw_files,process_params=metric)
	plot_val=get_met(df_file, metric,what)
	srt_val=meta[k]
	current_dict[uid]=(srt_val,plot_val,label)

	return current_dict,k

def plot_versus_ref_meta(raw_dirs,metric,what="max_value",ref_file=None,save_path=None):
	kv_dict={}
	label_lst=[]
	fig_count=1

	title=None

	for raw_dir in raw_dirs:
		kv_dict,k=get_kv_meta(raw_dir, metric, what,current_dict=kv_dict)
	if ref_file is not None:
		kv_dict,k=get_kv_meta(ref_file, metric, what,current_dict=kv_dict,expected_meta=k)
	#print(kv_dict)
	srt=sorted(kv_dict.items(), key=lambda kv: kv[1])
	print(srt)
	nval=len(srt)
	fig=plt.figure(figsize=(nval,10))
	#ax=fig.axes()
	for vals in srt:
		plt.bar(fig_count,vals[1][1])
		label_lst.append(vals[1][2])
		fig_count+=1

	plt.xlabel("Value")
	xt=np.arange(1,fig_count)
	plt.xticks(xt,label_lst)

	plt.ylabel(metric+" "+what)
	plt.title(k)
	if save_path is None:
		plt.show()
	else:
		#fig.set_size([20,20])
		fig_name=k+what+metric+".png"
		p=os.path.join(SAVEPATH,fig_name)
		plt.savefig(p)

if __name__ == '__main__':
	mode=sys.argv[1]
	param=sys.argv[2:]
	if mode=="plot_versus_ref":
		"""
		python process_run.py \
		plot_versus_ref \
		/data/prevel/runs/083_16:26/param1/ \
		time \
		max_value \
		"""
		raw_path=param[0]
		metric=param[1]
		comp=param[2]
		if len(param)>2:
			ref=param[3]
		else:
			ref=None
		#if len(param)>3:
			#save_path=
		#if os.isdir(raw_path):
		raw_dirs=fu.dir_list(raw_path, pattern="ind")
		plot_versus_ref_meta(raw_dirs,metric,what=comp,ref_file=ref,save_path=False)
	elif mode=="process_and_save":
		run_dir=param[0]
		gen_dirs=fu.dir_list(run_dir,"param")
		if len(gen_dirs)>0:
			for gen_dir in gen_dirs:
				ind_dirs=fu.dir_list(gen_dir,pattern="ind")
				for ind in ind_dirs:
					process(ind)
		else:
			ind_dirs=fu.dir_list(run_dir,pattern="ind")
			if len(ind_dirs)>0:
				for ind in ind_dirs:
					process(ind)
			else:
				fl=fu.file_list(run_dir,file_format=".csv")
				if len(fl)>=2:
					process(run_dir)

	




	#python process_run.py /data/prevel/runs/078_17:26/result
	#raw_files=fu.file_list(raw_path,file_format=".csv",recursive=True)
	#plot_versus_ref("/data/prevel/repos/biorob-semesterproject/data/raw_reference.csv",raw_files,"time",what="max_value")

	#python process_run.py /data/prevel/runs/081_15:18
	#raw_dirs=fu.dir_list(raw_path, pattern="ind")

	

		