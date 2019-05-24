import utils.file_utils as fu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema



def add_strike_lift(df,correction=True,smooth=True):
	fy=df.copy()
	fy=df.filter(like="Fy",axis=1)
	try:
		fy.columns=['left','right']
	except:
		print(df)
		return None
	smooth_size=10
	offset_lift=0
	offset_strike=0
	if smooth:
		offset_strike=smooth_size
		r=fy.rolling(smooth_size) #smooting (sum with 10 neighbours)
		fy=r.sum()  
	n=50 # number of points to be checked before and after 
	left_swing = argrelextrema(fy.left.values, np.less_equal, order=n)
	s=left_swing[0]
	offset_lift=0
	offset_strike=0
	
	s_filter=[s[idx+1]-offset_lift for idx in range(len(s)-1) if s[idx+1]-s[idx]>10 ]

	fy['leftlift']=fy.iloc[s_filter]['left']
	fy['leftoff']=fy.iloc[s]['left']
	
	s_filter=[s[idx] for idx in range(len(s)-1) if s[idx+1]-s[idx]>10 ]
	fy["leftstrike"]=fy.iloc[s_filter]['left']
	
	
	
	right_swing=argrelextrema(fy.right.values, np.less_equal, order=n)#[0]['right']
	s=right_swing[0]
	s_filter=[s[idx+1] for idx in range(len(s)-1) if s[idx+1]-s[idx]>10 ]
	fy['rightlift']=fy.iloc[s_filter]['right']
	fy['rightoff']=fy.iloc[s]['right']
	s_filter=[s[idx] for idx in range(len(s)-1) if s[idx+1]-s[idx]>10 ]
	fy['rightstrike']=fy.iloc[s_filter]['right']
	
	#fy['r_strike'] = fy.iloc[argrelextrema(fy.right.values, np.greater_equal, order=n)[0]-smooth_size]['right']
	#fy['l_strike'] = fy.iloc[argrelextrema(fy.left.values, np.greater_equal, order=n)[0]-smooth_size]['left']
	
	
	if correction:
		return fp_correction(fy)
	else:
		return fy

def fp_correction(fy_df,side='left',what="strike"):
	corr_df=fy_df.copy()

	idx_lift=fy_df[fy_df[side+'lift'].notnull()].index.to_list()
	idx_strike=fy_df[fy_df[side+'strike'].notnull()].index.to_list()
	#print(idx_lift)
	#print(idx_strike)
	if what=="strike":
		"""
		Checks that there is only one strike before the next lift detection.
		If there is two, only consider the first one
		"""
		for n in range(len(idx_strike)-1):
			if n>=len(idx_strike)-1:
				print("Breaking")
				break
			else:
				c_strike=idx_strike[n]
				n_strike=idx_strike[n+1]
				c_lift=idx_lift[n]
				#print("C_strike",c_strike,"n_strike",n_strike,"c_lift",c_lift,"\n")
				if n_strike<c_lift:
					strike=idx_strike.pop(n+1)
					print("Removing strike",strike)
					corr_df[side+'strike'][strike]=np.nan
					n=n+1
		return fp_correction(corr_df,side=side,what="lift")
	elif what=="lift":
		"""
		Checks that there is only one lift before the next strike detection.
		If there is two, only consider the first one
		"""
		for n in range(len(idx_lift)-1):
			if n>=len(idx_lift)-1:
				print("Breaking")
				break
			else:
				c_lift=idx_lift[n]
				n_lift=idx_lift[n+1]
				c_strike=idx_strike[n]
				#print("C_strike",c_strike,"n_strike",n_strike,"c_lift",c_lift,"\n")
				if n_lift<c_strike:
					lift=idx_lift.pop(n+1)
					print("Removing lift",lift)
					corr_df[side+'lift'][lift]=np.nan
					n=n+1
		return corr_df

def get_max_event_duration(idx_event1,idx_event2):
	dur=[]
	for idx in range(len(idx_event1)-1):
		idx1=idx_event1[idx]
		idx2=idx_event2[idx]
		if idx2<idx1:
			idx2=idx_event2[idx+1]
		dur.append(idx2-idx1)
	return max(dur)

def split_stride(df,fy_df,column,side='smart',how="strike_to_strike",with_timestamps=False):
	if side=='smart':
		lwr=column.lower()
		if 'left' in lwr or lwr[-2:]=='_l':
			side='left'
		elif 'right' in lwr or lwr[-2:]=='_r':
			side='right'
		else:
			print("No side found in ",column,'taking right as default')
			side='right'
	if how=="strike_to_strike":
		idx_event1=fy_df[fy_df[side+'strike'].notnull()].index.to_list()
		idx_event2=idx_event1[1:]
	elif how=="liftoff_to_liftoff":
		idx_event1=fy_df[fy_df[side+'lift'].notnull()].index.to_list()
		idx_event2=idx_event1[1:]
	elif how=="liftoff_to_strike":
		idx_event1=fy_df[fy_df[side+'lift'].notnull()].index.to_list()
		idx_event2=fy_df[fy_df[side+'strike'].notnull()].index.to_list()
	elif how=="strike_to_liftoff":
		idx_event1=fy_df[fy_df[side+'strike'].notnull()].index.to_list()
		idx_event2=fy_df[fy_df[side+'lift'].notnull()].index.to_list()
	
	stride_index=pd.RangeIndex(0,int(get_max_event_duration(idx_event1,idx_event2)*100.0))
	df_out=pd.DataFrame(index=stride_index)
	time_stamps={}
	for idx in range(len(idx_event1)-1):
		idx1=idx_event1[idx]
		idx2=idx_event2[idx]
		if idx2<idx1:
			idx2=idx_event2[idx+1]
		st=df[column][idx1:idx2]
		st.index=pd.RangeIndex(0,len(st))
		st_newindex=st.reindex(stride_index)
		st_name="stride"+str(idx)
		time_stamps[st_name]=(idx1,idx2)
		df_out[st_name]=st_newindex    
	if not with_timestamps:
		return df_out
	else:
		return df_out,time_stamps

def interp_gaitprcent(s,n_goal):
	if s is None:
		print("Empty stride interpolate")
		return pd.Series(np.zeros(n_goal))
	r=s.dropna()
	n_init=len(r)

	if n_init>n_goal:
		downsample_idx=np.linspace(0,n_init,n_goal,dtype=int)
		vals=s[downsample_idx]
		fr=vals.dropna()
		fr.index=range(n_goal-1)
	elif n_init<n_goal:
		subsamble_idx=np.linspace(0,n_goal-1,n_init,dtype=int)
		r.index=subsamble_idx
		nr=pd.Series(index=np.linspace(0,n_goal-1,n_goal,dtype=int))
		nr[subsamble_idx]=r
		fr=nr.interpolate()
	else:
		fr=s
	return fr
def get_all_stride(full_df,metric,interp=True,how="strike_to_strike",with_timestamps=False):

	 # event detection
	fy_df=add_strike_lift(full_df)
	if with_timestamps:
		spl_stride,ts=split_stride(full_df,fy_df,metric,how,with_timestamps=with_timestamps) # split by stride
	else:
		spl_stride=split_stride(full_df,fy_df,metric,how,with_timestamps=with_timestamps)
	if interp:
		for stride in spl_stride.columns:
			spl_stride[stride]=interp_gaitprcent(spl_stride[stride],100)
		spl_stride=spl_stride.dropna()
		
	if with_timestamps:
		return spl_stride,ts# split by stride
	else:
		return spl_stride

def get_mean_std_stride(full_df,metric,fy_df=None,interp=True,how="strike_to_strike",stride_choice="repmax"):
	if fy_df is None:
		fy_df=add_strike_lift(full_df)
	
	spl_stride=split_stride(full_df,fy_df,metric,how=how) # split by stride
	for stride in spl_stride.columns:
		spl_stride[stride]=interp_gaitprcent(spl_stride[stride],100)
	
	spl_stride=spl_stride.dropna()
	cor=spl_stride.corr() # computes correlation
	
	su=cor.sum(axis=1)
	rep_index=su[su.values>su.mean()].index
	rep_strides=spl_stride.filter(items=rep_index)
	
	var=spl_stride.std(axis=1)
	if stride_choice=="repmax":
		rep_max_idx=su[su.values==su.max()].index
		#print("repmax:",rep_max_idx)
		y=rep_strides[rep_max_idx].iloc[:,0]
	elif stride_choice=="mean":
		 y=rep_strides.mean(axis=1)
	elif stride_choice in spl_stride.columns:
		y=spl_stride[stride_choice]
	else:
		raise ValueError('stride_choice',stride_choice)
	if interp:
		var=interp_gaitprcent(var,100)
		y=interp_gaitprcent(y,100)
	return y,var
# ALTERNATES TO HANDLE FLORIN FORMAT
def split_stride_contact(contact):
	fy=contact.copy()
	s=contact[contact.left==0].index

	s_filter=[s[idx+1] for idx in range(len(s)-1) if s[idx+1]-s[idx]>10 ]
	fy['leftlift']=contact.iloc[s_filter]['left']
	s_filter=[s[idx] for idx in range(len(s)-1) if s[idx+1]-s[idx]>10 ]
	fy['leftstrike']=contact.iloc[s_filter]['left']
	
	s=contact[contact.right==0].index

	s_filter=[s[idx+1] for idx in range(len(s)-1) if s[idx+1]-s[idx]>10 ]
	fy['rightlift']=contact.iloc[s_filter]['right']
	s_filter=[s[idx] for idx in range(len(s)-1) if s[idx+1]-s[idx]>10 ]
	fy['rightstrike']=contact.iloc[s_filter]['right']
	
	return fp_correction(fy)

def get_rep_var_from_contact(contact,metric,joints,drop_n_first_strides=3):
	if metric not in joints.columns:
		print('\n Metric:\t',metric)
		print('\n Columns:\t',joints.columns())
		raise('KeyError')
	stride_choice='repmax'
	interp=True
	fy_df=split_stride_contact(contact)
	spl_stride=split_stride(joints,fy_df,metric)
	spl_stride.drop(spl_stride.iloc[:,0:drop_n_first_strides], axis=1, inplace=True)
	for stride in spl_stride.columns:
		spl_stride[stride]=interp_gaitprcent(spl_stride[stride],100)
	
	spl_stride=spl_stride.dropna()

	cor=spl_stride.corr() # computes correlation
	su=cor.sum(axis=1)

	var=spl_stride.std(axis=1)

	if stride_choice=="repmax":
		rep_max_idx=su[su.values==su.max()].index
		try:
			y=spl_stride[rep_max_idx[0]]#.iloc[:,0]
		except IndexError:
			y=None 
	elif stride_choice=="mean":
		rep_index=su[su.values>su.mean()].index
		rep_strides=spl_stride.filter(items=rep_index)
		y=rep_strides.mean(axis=1)
	elif stride_choice in spl_stride.columns:
		y=spl_stride[stride_choice]
	else:
		raise ValueError('stride_choice',stride_choice)
	if interp:
		var=interp_gaitprcent(var,100)
		y=interp_gaitprcent(y,100)
		
	return y,var


if __name__ == '__main__':
	WINTER_PATH="../../data/winter_data/"
	win_df_data=pd.read_csv(WINTER_PATH+"data_normal.csv")
	win_df_std=pd.read_csv(WINTER_PATH+"std_normal.csv")
	what="ankle"
	RAW_REFERENCE_PATH="../../data/raw_reference.csv"

	df=pd.read_csv(open(RAW_REFERENCE_PATH))

	sub_hip=df.filter(like="hip",axis=1)
	df[sub_hip.columns]=-df[sub_hip.columns]
	mean_win=interp_gaitprcent(win_df_data[what],100)
	std_win=interp_gaitprcent(win_df_std[what],100)

	mean_exp,std_exp=get_mean_std_stride(df,"angles_"+what+"_r",interp=True,stride_choice="stride24")
	plt.plot(mean_exp)
	plt.show()