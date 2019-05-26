import yaml
import numpy as np
import utils.file_utils as fu
import pandas as pd

LOG_DEBUG=1
LOG_INFO=2
LOG_WARNING=3
LOG_ERROR=4

LOG_LEVEL=LOG_DEBUG
def get_uid_result(ind_path,verbose=False):
	processed_file=fu.assert_one_dim(fu.file_list(ind_path,file_format=".csv",pattern="result"),\
									critical=False,verbose=verbose)
	try:
		res=pd.read_csv(processed_file).to_dict('records')[0]
	except:
		try:
			if LOG_LEVEL<=LOG_WARNING:
				print("\n[WARNING]Processed file has no columns\n")
			val=float(pd.read_csv(processed_file).columns[1])
			res=pd.DataFrame(index=pd.RangeIndex(1),columns=["fitness","uid"])
			res["fitness"]=val
			if LOG_LEVEL<=LOG_WARNING:
				print("\n[WARNING]Extracted results\n",res)

		except Exception as e:
			raise e

	meta_file=fu.assert_one_dim(fu.file_list(ind_path,file_format=".yaml",pattern="meta"),\
								critical=True,verbose=verbose)
	dict_meta=yaml.load(open(meta_file))
	try:
		res["uid"]=dict_meta["uid"]
	except KeyError:
		print("\n[ERROR] Missing uid in",meta_file,"\nkeys are\t",dict_meta.keys())
		raise RuntimeError
	return res

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
