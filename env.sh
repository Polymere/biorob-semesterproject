ENVS=$(conda env list | awk '{print $1}' )
if [[ $ENVS = *"bio"* ]]; then
	echo "Activating conda env (bio)"
	conda activate bio
else 
	echo "Installing conda env "
	conda create --name bio --file setup/biorob_proj_config.yml
	conda activate bio
	#exit
fi;

export PYTHONPATH="$PYTHONPATH:$(pwd)/src"
