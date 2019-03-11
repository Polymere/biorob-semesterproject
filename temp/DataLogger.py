import logging
import numpy as np
import os

class DataLogger():
    """Generate Muscle Models for the the animal.
    """

    def __init__(self,body_communication,jointopts = ["angles", "velocities", "moments"], muscleopts
        = ["act", "stim", "l_ce", "f_se", "v_ce", "f_ce"], grfopts =  ["LGRF", "RGRF"]):

        # Define what should be logged
        self.joint_log = jointopts
        self.muscle_log = muscleopts
        self.grf_log = grfopts


        self.create_files(jointopts, body_communication.joints)
        self.create_files(muscleopts, body_communication.muscles)
        self.create_files(grfopts)

        self.angles = []
        self.velocities = []
        self.moments = []
        self.act = []
        self.stim = []
        self.f_se = []
        self.l_ce = []
        self.f_ce = []
        self.v_ce = []
        self.l_mtc = []
        self.delta_length = []
        self.LGRF = []
        self.RGRF = []
        self.counter = 0

        self.angles = {}
        self.velocities = {}
        self.moments = {}
        self.act = {}
        self.stim = {}
        self.f_se = {}
        self.l_ce = {}
        self.f_ce = {}
        self.v_ce = {}
        self.RF={}
        self.first_call=True

    def create_files(self,log,system = None):
        # Create files and headers

        #Check if directory exists else create one
        if not os.path.isdir('./Raw_files'):
            os.mkdir('./Raw_files')
        for iname in range(len(log)):
            if system is None:
                with open("Raw_files/" + str(log[iname]) + ".txt", "w") as log_file:
                    log_file.write(str("GRFx\tGRFy"))
                    log_file.write("\n")
            else:
                # open file
                with open("Raw_files/" + str(log[iname]) + ".txt", "w") as log_file:
                    for name in system.iterkeys():
                    # write header -> muscle or joint name
                        log_file.write(str(system[name].name))
                        log_file.write("\t")
                    log_file.write("\n")

    def write_to_files(self, system):
        with open("Raw_files/velocities.txt", "a") as log_file:
            log_file.write(str(self.velocities))
        
        with open("Raw_files/moments.txt", "a") as log_file:
            log_file.write(str(self.moments))
                
        with open("Raw_files/act.txt", "a") as log_file:
            log_file.write(str(self.act))
    
        with open("Raw_files/stim.txt", "a") as log_file:
            log_file.write(str(self.stim))

        with open("Raw_files/l_ce.txt", "a") as log_file:
            log_file.write(str(self.l_ce))
    
        with open("Raw_files/v_ce.txt", "a") as log_file:
            log_file.write(str(self.v_ce))

        with open("Raw_files/f_ce.txt", "a") as log_file:
            log_file.write(str(self.f_ce))
    
        with open("Raw_files/f_se.txt", "a") as log_file:
            log_file.write(str(self.f_se))

        with open("Raw_files/angles.txt", "a") as log_file:
            log_file.write(str(self.angles))
    
        with open("Raw_files/LGRF.txt", "a") as log_file:
            log_file.write(str(self.LGRF))

        with open("Raw_files/RGRF.txt", "a") as log_file:
            log_file.write(str(self.RGRF))

    def write_to_single_file(self,file_path="./",run_id="run_default"):
    	run_id="moments" # BANDAID
    	with open(os.path.join(file_path,run_id,".csv"),'w') as log_file:
    		writer=csv.DictWriter(log_file,self.moments.keys())
    		writer.writeheader()
    		writer.writerow(self.moments)


    def dict_step(self, body_communication, ground_reaction_force = None):
    	"""
    	Saves the values of the current simulation step to dictionnaries, with the key being
    	the muscle or joint name
    	"""
    	if self.first_call:
    		print ("Initializing dictionnaries \n")
			for joint_name in body_communication.joints:
	    		self.angles[joint_name]=[]
	    		self.velocities[joint_name]=[]
	    		self.moments[joint_name]=[]
	    	for muscle_name in body_communication.muscles:
	    		self.act[muscle_name]=[]
		        self.stim[muscle_name]=[]
		        self.f_se[muscle_name]=[]
		        self.l_ce[muscle_name]=[]
		        self.f_ce[muscle_name]=[]
		        self.v_ce[muscle_name]=[]
		    self.first_call=False    		
    	for joint_name in body_communication.joints:
    		self.angles[joint_name].append(body_communication.joint_positions[joint_name])
    		self.velocities[joint_name].append(body_communication.joint_velocities[joint_name])
    		self.moments[joint_name].append(body_communication.new_torque[joint_name])
    	for muscle_name in body_communication.muscles:
    		self.act[muscle_name].append(body_communication.act[muscle_name])
	        self.stim[muscle_name].append(body_communication.stim[muscle_name])
	        self.f_se[muscle_name].append(body_communication.f_se[muscle_name])
	        self.l_ce[muscle_name].append(body_communication.l_ce[muscle_name])
	        self.f_ce[muscle_name].append(body_communication.f_ce[muscle_name])
	        self.v_ce[muscle_name].append(body_communication.v_ce[muscle_name])
	    if ground_reaction_force is not None:
	    	for reaction_force_key in ground_reaction_force.keys():
	    		self.RF[reaction_force_key].append(ground_reaction_force[reaction_force_key])


    
    def step(self, body_communication, ground_reaction_force = None):
        # make row that should be appeded
        rowangles = []
        rowvelocities = []
        rowmoments = []
        for name in body_communication.joints:
            rowangles.append(body_communication.joint_positions[name])
            rowvelocities.append(body_communication.joint_velocities[name])
            rowmoments.append(body_communication.new_torque[name])
            
        self.angles.append(rowangles)
        self.velocities.append(rowvelocities)
        self.moments.append(rowmoments)

        rowact = []
        rowstim = []
        rowf_se = []
        rowl_ce = []
        rowf_ce = []
        rowv_ce = []
        rowl_mtc = []
        rowdelta_length = []

        for name in body_communication.muscles:
            rowact.append(body_communication.act[name])
            rowstim.append(body_communication.stim[name])
            rowf_se.append(body_communication.f_se[name])
            rowl_ce.append(body_communication.l_ce[name])
            rowf_ce.append(body_communication.f_ce[name])
            rowv_ce.append(body_communication.v_ce[name])

        self.act.append(rowact)
        self.stim.append(rowstim)
        self.f_se.append(rowf_se)
        self.l_ce.append(rowl_ce)
        self.f_ce.append(rowf_ce)
        self.v_ce.append(rowv_ce)
        
        if ground_reaction_force != None:
            rowlgrf = []
            rowlgrf.append(ground_reaction_force['Lx'])
            rowlgrf.append(ground_reaction_force['Ly'])
            self.LGRF.append(rowlgrf)

            rowrgrf = []
            rowrgrf.append(ground_reaction_force['Rx'])
            rowrgrf.append(ground_reaction_force['Ry'])
            self.RGRF.append(rowrgrf)
     
    def add_to_logger(self, body_communication, ground_reaction_force = None, time_step = 10):
        self.counter += 1
        if self.counter >= time_step:
            self.step(body_communication, ground_reaction_force)
            self.counter = 0
            
