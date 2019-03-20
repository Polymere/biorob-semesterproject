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
import utils.file_utils as fu
import matplotlib.pyplot as plt

TIME_STEP=1.0 #ms

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
def plot_versus_ref(ref_file,raw_files,metric,what="max_value"):
	df_ref=compute_df(ref_file,process_params=metric)
	fig=plt.axes()
	if what=="max_value":
		fig_count=1
		fig.bar(fig_count,max(df_ref[metric]))
		for raw_file in raw_files:
			fig_count+=1
			df_file=compute_df(raw_file,process_params=metric)
			fig.bar(fig_count,max(df_file[metric]))
	elif what=="mean_value":
		fig_count=1
		fig.bar(fig_count,mean(df_ref[metric]))
		for raw_file in raw_files:
			fig_count+=1
			df_file=compute_df(raw_file,process_params=metric)
			fig.bar(fig_count,mean(df_file[metric]))
	elif what=="mean_std":
		fig_count=1
		fig.bar(fig_count,mean(df_ref[metric]),yerr=df_ref) ## WIP
		for raw_file in raw_files:
			fig_count+=1
			df_file=compute_df(raw_file,process_params=metric)
			fig.bar(fig_count,mean(df_file[metric]))

	else:
		print("Comparison:\t", what,"\tnot implemented")
	plt.show()


if __name__ == '__main__':
	if len(sys.argv)==4:

		process_params=sys.argv[3:]
	elif len(sys.argv)==3:
		process_params="all"
		
	else:
		print("Incorrect args",sys.argv)
	raw_path=sys.argv[1]
	#python process_run.py /data/prevel/runs/078_17:26/result
	raw_files=fu.file_list(raw_path,file_format=".csv",recursive=True)
	plot_versus_ref("/data/prevel/repos/biorob-semesterproject/data/raw_reference.csv",raw_files,"time",what="max_value")
	#for raw in raw_files:
	#	df=compute_df(raw,process_params)
		#export_file(df,out_file)
		