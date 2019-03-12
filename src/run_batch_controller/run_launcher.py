"""
Launching a run with
Inputs : 
	Parameter file eg. '../../../modeling/configFiles/geyer-structured-ISB.yaml' or paramter files folder
	Number of folds for each parameter set
	Worlds to test (if we optimize of different terrain/slopes)
	Run result output directory
"""
import utils.file_utils as fu
import data_analysis.import_run as imp
import sys
from shutil import copy
import os
#def launch_run(fold_number,parameter_file,world_file):


CONTROLLERPATH="/data/prevel/human_2d/webots/controllers/GeyerReflex"
STATIC_OUTPUT_DIRPATH=CONTROLLERPATH+"/Raw_files"
if __name__ == '__main__':
	if len(sys.argv)==5:

		parameter_dirpath=sys.argv[1]
		nfolds=int(sys.argv[2])
		world_dirpath=sys.argv[3]
		result_dir=sys.argv[4]

		parameter_files=fu.file_list(parameter_dirpath,file_format='.yaml')
		world_files=fu.file_list(world_dirpath,file_format='.wbt')
		fu.assert_dir(result_dir)
		print("********************************************\n")
		print("Running \t",nfolds," folds\n")
		print("Parameter files :\n",parameter_files,"\n")
		print("Testing on worlds:\n",world_files,"\n")
		print("Outputs in :\t",result_dir,"\n")
		print("*********************************************")
		param_counter=1
		world_counter=1

		for parameter_file in parameter_files:
			param_filepath=os.path.join(parameter_dirpath,parameter_file)
			param_dir=os.path.join(result_dir,("param"+str(param_counter)))
			fu.assert_dir(param_dir)
			copy(param_filepath, param_dir)
			for world_file in world_files:
				world_filepath=os.path.join(world_dirpath,world_file)
				world_dir=os.path.join(param_dir,("world"+str(world_counter)))
				fu.assert_dir(world_dir)
				copy(world_filepath, world_dir)
				for fold in range(nfolds):
					#launch_run(fold,param_filepath,world_filepath)
					imp.import_run(STATIC_OUTPUT_DIRPATH,save_path=world_dir,save_name="run"+str(fold))
				world_counter=world_counter+1
			param_counter=param_counter+1




	else:
		print("Arguments should be param_file, nfolds,worlds,result_dir")
		

