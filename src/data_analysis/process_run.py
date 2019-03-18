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


def compute_df(raw_file,process_params):

	data_in=pd.read_csv(open(raw_file))
	data_out=pd.DataFrame(data_in.index)
	print (data_in)
	print(data_out)
if __name__ == '__main__':
	if len(sys.argv)>=4:
		raw_path=sys.argv[1]

		out_path=sys.argv[2]
		raw_files=fu.file_list(raw_path,file_format=".csv")
		process_params=sys.argv[3:]
		for raw in raw_files:
			df=compute_df(raw,process_params)
		#export_file(df,out_file)
		