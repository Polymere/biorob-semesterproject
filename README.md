# Repository for semester project : TITLE

## Milestones

### 1. Setting up the environment for desktop + laptop
- Webots -> OK
- Matlab ?
- Python 
- Other ?

### 2. Parameter influence
 Evaluate the influence of the tunable parameters on the gait, both for the muscle reflex model and CPG+reflex model.

1. Determine which parameters to study (literature)

	- Muscle reflex model :


	![Muscle reflex model parameters [@Geyer2010AMM]](./figures/reflex_params.png)



	- CPG + muscle reflex :


	![Muscle reflex + CPG model parameters](./figures/reflex_cpg_params.png)




2. Simulate the model with these parameters set to zero (or minimal bound value)

	> Is the minimal value a mathematical constraint or a physiological constraint ?

	In the first case, we should take it into account (avoid singularity), and use the minimal value instead of 0. In the second case, 
	we should test both.
	
3. Compare with the optimized model, both visually (is it still working/stable) and numerically (similitudes on joint angles/position, muscle activations...) -> see what is applicable

### 3. Fitting model to experimental data

1. Select data (MOCAP) from a real subject (healthy/pathological)
2. Pre-process (normalize stride lenght, maybe mean values)
3. Fitness function (start with RMS error)
	- What are we trying to fit ?
		- Joint angles
		- Joint torques 
		- stride
4. Which GA (explore DEAP library (python))
5. Melt 3 clusters until good fit


### 4. Evaluate

Once the best set of parameters to fit the data is found, evaluate :

1. Gait similarity (parameters not considered in fitness function)
2. EMG values
3. Biological interpretation
