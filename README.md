# Repository for semester project : TITLE

## Milestones

### 1. Setting up the environment for desktop + laptop
- Webots -> OK
- Python  -> OK, see ./config

### 2. Parameter influence
 Evaluate the influence of the tunable parameters on the gait, both for the muscle reflex model and CPG+reflex model.

1. Determine which parameters to study (literature)

	- Muscle reflex model :

	<!--![Muscle reflex model parameters [@Geyer2010AMM]](./figures/reflex_params.png)-->
	![Muscle reflex model parameters interpretation [@Geyer2010AMM]](./figures/table_param_muscle.svg)

	- CPG + muscle reflex :

	![Muscle reflex + CPG model parameters](./figures/reflex_cpg_params.png)



2. Simulate the model with these parameters set to zero (or minimal bound value)

	As the minimum bound value is not a mathematical constraint, it would be safe to test values below (test zeros), to see how the model behaves

	> See how/which data is saved, and how I can import it with python

	
3. Compare with the optimized model, both visually (is it still working/stable) and numerically (similitudes on joint angles/position, muscle activations...) -> see what is applicable
	- Energy (cost of travel)
	- Similarity in angles, torques
	- Stride lenght
	- Stability (just binary or is there a metric ?)
	- Speed
	- Impact of terrain

4. Implement result analysis/plotting functions

### 3. Fitting model to experimental data

1. Select data (MOCAP) from a real subject (healthy/pathological)
2. Pre-process (normalize stride lenght, maybe mean values)
3. Fitness function
	- Multi-objective optimization
	 	- similarity in angles/torques
	 	- maximize speed
	 	- minimize energy
	 	- maximize stability (worst measure with different terrains)
4. Which GA (explore DEAP library (python))
	- Multi-objective optimization algorithm (NSGA-II ?)
5. Melt 3 clusters until good fit

### 3b. Alternative : Optimize model with pathological constraints

Idea would be to remove some components of the model in order to characterize a pathology (eg. ankle angle feedback for diabetes), and evolve the model from there.
The goal would be to see if we observe a similar gait (eg. flat footed for diabetes).
For the muscle-reflex model (FBL), this could correspond (?) of setting $G_{TA}$ to zero  (gain for TA length feedback)

### 4. Evaluate

Once the best set of parameters to fit the data is found, evaluate :

1. Gait similarity (parameters not considered in fitness function)
2. EMG values
3. Biological interpretation

![](./figures/biblio.svg)

## Progress

- Weeks 1-3 : litterature review, project organisation and environment setup
- Week 4 : 
	- Import a run result from raw files DONE (maybe refactor a bit the logger to write in a single csv file) (not done for now)
	- Launch a batch of runs DONE
	- Generated different parameters files, by modifying single parameters DONE
	- Improve param file generations (bound parameters, aliases) DONE
	- Add reverse mapping (in order to have a cleaner param file to read) -> W5

	
- Week 5 
	- Write higher level script for param file generation DONE
	- Select parameters and values DONE
	- Run (around 50 parameters sets, runtime 1h) DONE
	- Reverse mapping parameter file DONE
	- Evaluate the results
		- Metric selection DONE (extract metrics computed in objectives.py -> W6)
		- plot utils (versus reference run) DONE
- Week 6 
	- Runs with modrange (2 linespace for each parameters, between min to ref, and ref to max) DONE
	- Extracted metrics in objectives.py DONE
	- Run (20 parameters, 31 values) -> did not run for all different prestim DONE
	- Save metrics 	+ objectives in a single file DONE
	- Add metadata to each run (studied parameter, value, labels ...) DONE
	- Plot with discriminating parameter (ex: plot speed vs param when simulation time == full time) DONE
- Week 7 
	- Event detection in stride for additional metrics (stride length, frequency) DONE 
	- Split kinematics by stride to compare with real data (winter) DONE
	
- Week 8
	- Runs with 2 parameters (GGAS/GSOL) varying DONE
	 
 	- correlations plots/comparison with real data DONE
 	- Plots in 2D parameter space DONE
	- Improved simulation controller (switch parameters when leg is in swing) DONE
	- Correlations in kinematics with ref  (midterm) DONE

- MIDTERM PRESENTATION
 	1. Ankle/knee does not match real data, and leads to bad stability -> wait for better parameters or optimize myself ?
 	2. Sensitivity analysis with correlation to reference for kine (see how the gait changes with parameter)DONE
 	3. Better explore the "biological" range for dual GSOL/GGAS




## Running the code

### Dependencies

Additional python packages dependencies are :

- pandas (handling the data)
- jupyter notebook (not required, used for prototyping)

These dependencies where installed in a conda environment, which can be installed with:
  
`conda create --name <env> --file config/biorob_proj.txt`

### Environment

The src folder should be added to the PYTHON_PATH

`export PYTHONPATH=$PYTHONPATH:path/to/repo/src`

### Scripts
TODO
