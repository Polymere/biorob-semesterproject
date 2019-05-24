ENVS=$(conda env list | awk '{print $1}' )
if [[ $ENVS = *"bio"* ]]; then
  echo "Activating conda env (bio)"
   source activate bio
else 
   echo "Installing conda env "
	conda env create -f biorob_proj_config.yml
	source activate bio
   exit
fi;

export PYTHONPATH="$PYTHONPATH:../src"
