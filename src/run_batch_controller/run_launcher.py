"""
Launching a run with


Inputs : 
	Parameter file eg. '../../../modeling/configFiles/geyer-structured-ISB.yaml' or paramter files folder
	Number of folds for each parameter set
	Worlds to test (if we optimize of different terrain/slopes)
	Run result output directory


	The result directory structure is as follows
	result_dir
		/param1
			param1_file.yaml
			/world1
				world1_file.wbt
				run1.csv
				run2.csv
				...
			/world2
				...
		/param2

		...

EXAMPLE:
python run_launcher.py /data/prevel/trial/param_folder \
4 \
/data/prevel/trial/worlds_folder \
/data/prevel/trial/result 

python run_launcher.py /data/prevel/params/completed \
1 \
/data/prevel/trial/worlds_folder \
/data/prevel/trial/result 
"""
import utils.file_utils as fu
import data_analysis.import_run as imp
import sys
from shutil import copy,copyfile
import os
import subprocess
"""

"""
H2D_SRC_DIR="/data/prevel/human_2d"

CONTROLLER_RPATH="webots/controllers/GeyerReflex"

PARAMFILE_RPATH="modeling/configFiles/current.yaml"
WORLD_RPATH="webots/worlds/current.wbt"

CONTROLLER_ABSPATH=os.path.join(H2D_SRC_DIR,CONTROLLER_RPATH)
PARAMFILE_ABSPATH=os.path.join(H2D_SRC_DIR,PARAMFILE_RPATH)
WORLD_ABSPATH=os.path.join(H2D_SRC_DIR,WORLD_RPATH)
SIM_OUTPUTDIR_RPATH=os.path.join(CONTROLLER_ABSPATH,"Raw_files")

def launch_run(parameter_file=None,world_file=None):

	subprocess.run(["webots", "--mode=fast", "--batch", WORLD_ABSPATH])

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
			# save current param file in result/param folder
			copyfile(param_filepath,PARAMFILE_ABSPATH) 
			# copy current  param file to PARAMFILE_ABSPATH (replace the file to change current run parameters)
			for world_file in world_files:
				world_filepath=os.path.join(world_dirpath,world_file)
				world_dir=os.path.join(param_dir,("world"+str(world_counter)))
				fu.assert_dir(world_dir)
				copy(world_filepath, world_dir)
				# save current world file in result/param/world folder
				copyfile(world_filepath,WORLD_ABSPATH)
				# copy current  world file to WORLD_ABSPATH (replace the file to change current run world)
				for fold in range(nfolds):
					launch_run(param_filepath,world_filepath)
					imp.import_run(SIM_OUTPUTDIR_RPATH,save_path=world_dir,save_name="run"+str(fold+1))
				world_counter=world_counter+1
			param_counter=param_counter+1




	else:
		launch_run()
		print("Arguments should be param_file, nfolds,worlds,result_dir")
		

