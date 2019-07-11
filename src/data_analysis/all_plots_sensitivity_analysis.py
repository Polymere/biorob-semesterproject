import pandas as pd
import utils.file_utils as fu
import matplotlib.pyplot as plt
from data_analysis.process_run import CppRunProcess
import seaborn as sns
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore') # pandas warning a utter trash
ASYM=False
MAP_PARAM_TEX={"solsol_wf": "$G_{SOL}$",
"solta_wf": "$G_{SOL,TA}$",
"gasgas_wf": "$G_{GAS}$",
"vasvas_wf": "$G_{VAS}$",
"gluglu_wf": "$G_{GLU}$",
"hamham_wf": "$G_{HAM}$",
"tata_wl": "$G_{TA}$",
"hfhf_wl": "$G_{HF}$",
"hamhf_wl": "$G_{HAM,HF}$",
"ta_bl": "$l_{off,TA}$",
"hf_bl": "$l_{off,HF}$",
"ham_bl": "$l_{off,HAM}$",
"deltas": "$\Delta S$",
"sol_activitybasal": "$S_{0,SOL}$",
"ta_activitybasal": "$S_{0,TA}$",
"gas_activitybasal": "$S_{0,GAS}$",
"vas_activitybasal": "$S_{0,VAS}$",
"ham_activitybasal": "$S_{0,HAM}$",
"glu_activitybasal": "$S_{0,GLU}$",
"hf_activitybasal": "$S_{0,HF}$",
"kbodyweight": "$k_{bw}$",
"kp": "$k_p$",
"kd": "$k_d$",
"kref": r'$\theta_{ref}$',
"klean": "$k_{lean}$"}


def plt_maxtime(fit_df,param):
	if ASYM:
		tmp_param=param+"_left"
	else:
		tmp_param=param
	fit_df=fit_df[fit_df[tmp_param].notnull()]
	sub=fit_df.maxtime.to_frame()
	plt.rc('text', usetex=False)
	plt.rc('font', family='serif')
	fig, ax = plt.subplots()
	cmap = sns.cubehelix_palette(dark=0.4, light=.85, gamma=0.9,rot=.3,start=1.2,\
						n_colors=2,hue=1.,as_cmap=True,reverse=False)
	sub.index=fit_df[tmp_param]
	sub.sort_index(inplace=True)
	sns.heatmap(sub.T,square=True,cmap=cmap,linewidths=.1,linecolor='k',vmin=0,vmax=20,xticklabels=4,cbar=False)
	plt.xticks(rotation=0)
	plt.xlabel("")
	plt.yticks([])
	labels = [item.get_text() for item in ax.get_xticklabels()]
	ax.set_xticklabels([str(round(float(label), 2)) for label in labels])
	plt.ylabel(MAP_PARAM_TEX[param],rotation=0,usetex=True,ha='right',va='center');
	plt.savefig("./figure_sym/maxtime_"+param+".pdf", dpi=None, facecolor='w', edgecolor='k',
		orientation='portrait', papertype=None, format="pdf",
		transparent=False, bbox_inches=None, pad_inches=0.1,
		frameon=True, metadata=None)

def plt_energytodist(fit_df,param,bounds):
	if ASYM:
		tmp_param=param+"_left"
	else:
		tmp_param=param
	fit_df=fit_df[fit_df[tmp_param].notnull()]
	nostab=fit_df.fit_stable==-1000
	zeroed_not_stab=fit_df
	zeroed_not_stab.loc[nostab,"energy_to_dist"]=np.nan
	sub=zeroed_not_stab.energy_to_dist.to_frame()
	plt.rc('text', usetex=False)
	#plt.rc('font', family='serif')
	cmap = sns.cubehelix_palette(dark=0.4, light=.85, gamma=2.9,rot=.3,start=2.0,\
							n_colors=2,hue=1.,as_cmap=True,reverse=True)

	cmap=sns.color_palette("Purples_r")
	fig, ax = plt.subplots()
	sub.index=zeroed_not_stab[tmp_param]
	sub.sort_index(inplace=True)
	sns.heatmap(sub.T,square=True,cmap=cmap,linewidths=.1,linecolor='k',vmin=bounds[0],vmax=bounds[1],xticklabels=4,cbar=True,mask=sub.T.isnull())
	plt.xticks(rotation=0)
	plt.xlabel("")
	plt.yticks([])
	labels = [item.get_text() for item in ax.get_xticklabels()]
	ax.set_xticklabels([str(round(float(label), 2)) for label in labels])
	plt.ylabel(MAP_PARAM_TEX[param],rotation=0,usetex=True,ha='right',va='center');
	plt.savefig("./figure_sym/energy_"+param+".pdf", dpi=None, facecolor='w', edgecolor='k',
		orientation='portrait', papertype=None, format="pdf",
		transparent=False, bbox_inches=None, pad_inches=0.1,
		frameon=True, metadata=None)

def plt_similarity(fit_df,param,which,bounds):

	if ASYM:
		tmp_param=param+"_left"
	else:
		tmp_param=param
	fit_df=fit_df[fit_df[tmp_param].notnull()]
	nostab=fit_df.fit_stable==-1000
	zeroed_not_stab=fit_df
	zeroed_not_stab.loc[nostab,which]=np.nan
	sub=zeroed_not_stab[which].to_frame()
	plt.rc('text', usetex=False)
	#plt.rc('font', family='serif')
	cmap = sns.cubehelix_palette(dark=0.4, light=.85, gamma=2.9,rot=.3,start=2.0,\
							n_colors=2,hue=1.,as_cmap=True,reverse=False)
	cmap=sns.color_palette("Reds")
	fig, ax = plt.subplots()
	sub.index=zeroed_not_stab[tmp_param]
	sub.sort_index(inplace=True)
	sns.heatmap(sub.T,square=True,cmap=cmap,linewidths=.1,linecolor='k',vmin=bounds[0],vmax=bounds[1],xticklabels=4,cbar=True,mask=sub.T.isnull())
	plt.xticks(rotation=0)
	plt.xlabel("")
	plt.yticks([])
	labels = [item.get_text() for item in ax.get_xticklabels()]
	ax.set_xticklabels([str(round(float(label), 2)) for label in labels])
	plt.ylabel(MAP_PARAM_TEX[param],rotation=0,usetex=True,ha='right',va='center');
	plt.savefig("./figure_sym/"+which+"_"+param+".pdf", dpi=None, facecolor='w', edgecolor='k',
		orientation='portrait', papertype=None, format="pdf",
		transparent=False, bbox_inches=None, pad_inches=0.1,
		frameon=True, metadata=None)
if __name__ == '__main__':


	"""proc=CppRunProcess(({"kinematics_compare_kind" :"winter_to_cpp",
					"kinematics_compare_file" : "../../data/winter_data/data_normal.csv",
					"do_plot": False,
					"split_how":"strike_to_strike"}))
	drop_cols=['Unnamed: 0', 'hip_left', 'joints_angle1_ANGLE_HIPCOR_LEFT',
	   'knee_left', 'ankle_left', 'hip_right',
	   'joints_angle1_ANGLE_HIPCOR_RIGHT', 'knee_right', 'ankle_right',
	   'energy1_energy', 'footfall1_left', 'footfall1_right',
	   'distance1_distance']

	fit_df=None


	all_dirs=fu.dir_list("sensi_sym",pattern="")
	param_names=[]
	for cdir in all_dirs:
		param_names.append(os.path.split(cdir)[-1])
		for raw_file in fu.file_list(cdir,file_format=".csv"):
			df=pd.read_csv(raw_file)
			mini=df.drop(drop_cols,axis=1)
			fit=proc.get_fitness(df,keep_metrics=True)
			nrow=pd.concat([mini.iloc[0,:].to_frame().T,fit],axis=1)
			if fit_df is not None:
				fit_df=pd.concat([fit_df,nrow])
			else:
				fit_df=nrow
	fit_df.index=fit_df.uid
	fit_df.to_csv("sym_vs_geyer.csv")"""
	fit_df=pd.read_csv("sym_vs_geyer.csv")
	all_dirs=fu.dir_list("sensi_sym",pattern="")
	param_names=[]
	for cdir in all_dirs:
		param_names.append(os.path.split(cdir)[-1])
	stab=fit_df.fit_stable==1
	max_ene=fit_df[stab].energy_to_dist.quantile(0.75)
	min_ene=fit_df[stab].energy_to_dist.quantile(0.25)
	bounds_ene=(min_ene,max_ene)

	max_ankle=fit_df[stab].fit_corankle.quantile(0.75)
	min_ankle=fit_df[stab].fit_corankle.quantile(0.25)
	bounds_ankle=(min_ankle,max_ankle)

	max_hip=fit_df[stab].fit_corhip.quantile(0.75)
	min_hip=fit_df[stab].fit_corhip.quantile(0.25)
	bounds_hip=(min_hip,max_hip)

	max_knee=fit_df[stab].fit_corknee.quantile(0.75)
	min_knee=fit_df[stab].fit_corknee.quantile(0.25)
	bounds_knee=(min_knee,max_knee)

	print(bounds_ankle)
	for param in param_names:
		#plt_maxtime(fit_df, param)
		plt_energytodist(fit_df, param, bounds_ene)

		plt_similarity(fit_df, param,"fit_rmsankle",bounds_ankle)
		plt_similarity(fit_df, param,"fit_rmship",bounds_hip)
		plt_similarity(fit_df, param,"fit_corknee",bounds_knee)

