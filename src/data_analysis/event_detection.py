#!/usr/bin/env python
""" @package event_detection
Detects gait events and split accordingly
"""
import utils.file_utils as fu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

import time

LOG_DEBUG=1
LOG_INFO=2
LOG_WARNING=3
LOG_ERROR=4
LOG_LEVEL=LOG_INFO
def events_from_grf(fy,smooth=True):
	""" Processing of GRF data to detect strike and lift off timestamps

		Moving average smoothing, local extrema detection and false positive 
		removal (based on expected event succession order)
		
		Note :
		Equivalent to events_from_contact for the python implementation
		of the reflex controller (different available data)""
	"""


	if smooth:
		smooth_size=10
		r=fy.rolling(smooth_size) #smoothing (sum with 10 neighbours)
		fy=r.sum()  
	# number of points to check before and after for local extremas detection
	n=50 


	left_swing = argrelextrema(fy.left.values, np.less_equal, order=n)
	s=left_swing[0]
	s_filter=[s[idx+1] for idx in range(len(s)-1) if s[idx+1]-s[idx]>10 ]
	fy['leftlift']=fy.iloc[s_filter]['left']
	fy['leftoff']=fy.iloc[s]['left']
	s_filter=[s[idx] for idx in range(len(s)-1) if s[idx+1]-s[idx]>10 ]
	fy["leftstrike"]=fy.iloc[s_filter]['left']

	right_swing=argrelextrema(fy.right.values, np.less_equal, order=n)
	s=right_swing[0]
	s_filter=[s[idx+1] for idx in range(len(s)-1) if s[idx+1]-s[idx]>10 ]
	fy['rightlift']=fy.iloc[s_filter]['right']
	fy['rightoff']=fy.iloc[s]['right']
	s_filter=[s[idx] for idx in range(len(s)-1) if s[idx+1]-s[idx]>10 ]
	fy['rightstrike']=fy.iloc[s_filter]['right']

	return fp_correction(fy)

def events_from_contact(contact):
	"""	Processing of contact data to extract strike and lift off timestamps
		
		Note :
		Equivalent to events_from_grf for the cpp implementation
		of the reflex controller (different available data)"""
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

def fp_correction(event_df,side='left',what="strike"):
	""" False positive correction

		Dropping first all the strikes that are not followed by a lift off, 
		then all the lift off that are not followed by a strike
		Needed when using force feedback on the feet to split strides 
		(python controller) as the signal may be noisy
	"""
	corr_df=event_df.copy()

	idx_lift=event_df[event_df[side+'lift'].notnull()].index.to_list()
	idx_strike=event_df[event_df[side+'strike'].notnull()].index.to_list()

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
				if n_lift<c_strike:
					lift=idx_lift.pop(n+1)
					print("Removing lift",lift)
					corr_df[side+'lift'][lift]=np.nan
					n=n+1
		return corr_df

def get_max_event_duration(idx_event1,idx_event2):
	""" Maximum stride size for preallocation """
	dur=[]
	for idx in range(len(idx_event1)-1):
		idx1=idx_event1[idx]
		idx2=idx_event2[idx]
		if idx2<idx1:
			idx2=idx_event2[idx+1]
		dur.append(idx2-idx1)
	return max(dur)

def split_stride(data_df,event_df,metric,side='smart',how="strike_to_strike",
	with_timestamps=False):
	""" Splits data according to gait events 

			Input :
			data_df -- trial dataframe, containing metric as one of the columns
				(typically kinematics data), see import_raw in process_run.py
			event_df -- Output of one of the event detection methods (grf or
				contact)
			metric -- column of data_df to be considered
			side -- leg to consider for the gait events. Can be extracted from 
				metric string in our implementation (side = 'smart',default 
				value). This works if the metric contains 'left' or 'right' 
				(case insensitive) or ends with '_r' / '_l'. 
				Otherwise 'right' or 'left'
			how -- start and end event to consider, either 'strike_to_strike' 
				(default), 'strike_to_liftoff','liftoff_to_liftoff' or 
				'liftoff_to_strike'
			with_timestamps -- Include or not selected events timestamps to the 
				output (False by default)

			Output :
			stride_df -- dataframe, with each column containing a different
				stride. Temporal relations are kept.
			(time_stamps) --  Optional, dict of tuples 
				{stride#:(start_idx,end_idx)...} containing the index for
				each stride. Depends on sampling rate

	"""
	if side=='smart':
		lwr=metric.lower()
		if 'left' in lwr or lwr[-2:]=='_l':
			side='left'
		elif 'right' in lwr or lwr[-2:]=='_r':
			side='right'
		else:
			print("No side found in ",metric,'taking right as default')
			side='right'
	if how=="strike_to_strike":
		idx_event1=event_df[event_df[side+'strike'].notnull()].index.to_list()
		idx_event2=idx_event1[1:]
	elif how=="liftoff_to_liftoff":
		idx_event1=event_df[event_df[side+'lift'].notnull()].index.to_list()
		idx_event2=idx_event1[1:]
	elif how=="liftoff_to_strike":
		idx_event1=event_df[event_df[side+'lift'].notnull()].index.to_list()
		idx_event2=event_df[event_df[side+'strike'].notnull()].index.to_list()
	elif how=="strike_to_liftoff":
		idx_event1=event_df[event_df[side+'strike'].notnull()].index.to_list()
		idx_event2=event_df[event_df[side+'lift'].notnull()].index.to_list()
	
	stride_index=pd.RangeIndex(0,int(get_max_event_duration(idx_event1,idx_event2)*100.0))
	stride_df=pd.DataFrame(index=stride_index)
	time_stamps={}
	for idx in range(len(idx_event1)-1):
		idx1=idx_event1[idx]
		idx2=idx_event2[idx]
		if idx2<idx1:
			idx2=idx_event2[idx+1]
		st=data_df[metric][idx1:idx2]
		st.index=pd.RangeIndex(0,len(st))
		st_newindex=st.reindex(stride_index)
		st_name="stride"+str(idx)
		time_stamps[st_name]=(idx1,idx2)
		stride_df[st_name]=st_newindex    
	if not with_timestamps:
		return stride_df
	else:
		return stride_df,time_stamps

def interp_gaitprcent(s,n_goal):
	""" Interpolates  serie s to lenght n_goal, either by subsambling or 
	downsampling """
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



def get_repr_from_grf(data_df,metric,stride_choice="repmax",
						drop_n_first_strides=0,how="strike_to_strike"):
	""" Returns value and variance for metric in data according to stride_choice

		Gets the trial events thanks to contact data and splits the values of
		metric in joints data according to these events.
		Gets and returns the representative values and std of metric over the 
		splitted strides

		Input :
			data_df -- Dataframe containing both ground reaction force (GRFy)
				and kinematics information, typically output of get_raw in 
				process_run.py
			metric -- column of data_df to be considered
			stride_choice -- method used to compute the representative stride
				- repmax (default) : most representative stride for metric in 
				 the trial (highest correlation with other strides)
				- mean : mean over 'good strides' (above average correlation 
				with other strides) 
				- strideX : single stride, with X being the stride number
			drop_n_first_strides -- number of initial strides to be dropped when
			 	we use a launching gait, 0 by default
			how -- start and end event to consider, either 'strike_to_strike' 
				(default), 'strike_to_liftoff','liftoff_to_liftoff' or 
				'liftoff_to_strike'
		
		Output :
			y -- value of metric for the chosen representative stride
			var -- variance of metric over all the strides of the trial 
				(excluding dropped initial strides)


		Note : 
		Equivalent to get_repr_from_contact for the cpp implementation
		of the reflex controller (different available data)"""
	if metric not in data_df.columns:
		if LOG_LEVEL<=LOG_ERROR:
			print("\n[ERROR] Desired metric (",metric,") is not available \
				in input data\n",data_df.columns)
		raise KeyError 
	try:
		grf_df=data_df.filter(like="Fy",axis=1)
		grf_df.columns=['left','right']
	except ValueError:
		if LOG_LEVEL<=LOG_ERROR:
			print("[ERROR]Missing grf data:\n",data_df.columns)
		raise ValueError
	event_df=events_from_grf(grf_df)
	
	spl_stride=split_stride(data_df,event_df,metric,how=how) # split by stride
	return _get_repr_std(spl_stride,stride_choice,drop_n_first_strides)
	


def get_repr_from_contact(data_df,metric,stride_choice="repmax",
							drop_n_first_strides=3,how="strike_to_strike"):
	""" Returns value and variance for metric in data according to stride_choice

		Gets the trial events thanks to contact data and splits the values of
		metric in joints data according to these events.
		Gets and returns the representative values and std of metric over the 
		splitted strides

		Input :
			data_df -- Dataframe containing both contact information (footfall)
				and kinematics information, typically output of get_raw in 
				process_run.py
			metric -- column of data_df to be considered
			stride_choice -- method used to compute the representative stride
				- repmax (default) : most representative stride for metric in 
				 the trial (highest correlation with other strides)
				- mean : mean over 'good strides' (above average correlation with 
				other strides) 
				- strideX : single stride, with X being the stride number
			drop_n_first_strides -- number of initial strides to be dropped when
			 	we use a launching gait, 3 by default 
				see settings.xml file used for the simulation (in humanWebotsNmm/config)
			how -- start and end event to consider, either 'strike_to_strike' 
				(default), 'strike_to_liftoff','liftoff_to_liftoff' or 
				'liftoff_to_strike'
		
		Output :
			y -- value of metric for the chosen representative stride
			var -- variance of metric over all the strides of the trial 
				(excluding dropped initial strides)


		Note : 
		Equivalent to get_repr_from_grf for the python implementation
		of the reflex controller (different available data)"""
	if metric not in data_df.columns:
		if LOG_LEVEL<=LOG_ERROR:
			print("\n[ERROR] Desired metric (",metric,") is not available \
				in input data\n",data_df.columns)
		raise KeyError 
	try:
		contact_df=data_df.filter(like="footfall1")
		contact_df.columns=["left","right"]
	except ValueError:
		if LOG_LEVEL<=LOG_ERROR:
			print("[ERROR]Missing contact data:\n",data_df.columns)
		raise ValueError

	event_df=events_from_contact(contact_df)
	spl_stride=split_stride(data_df,event_df,metric,how=how)
	return _get_repr_std(spl_stride,stride_choice,drop_n_first_strides)
	

def _get_repr_std(spl_stride,stride_choice,drop_n_first_strides):
	""" Computes value and variance for input strides

		Input :
			spl_stride -- dataframe, with each column corresponding to a 
				different stride

			stride_choice -- method used to compute the representative stride
				- repmax : most representative stride for metric in 
				 the trial (highest correlation with other strides)
				- mean : mean over 'good strides' (above average correlation with 
				other strides) 
				- strideX : single stride, with X being the stride number
			drop_n_first_strides -- number of initial strides to be dropped when
			 	we use a launching gait		
		Output :
			y -- value of metric for the chosen representative stride
			var -- variance of metric over all the strides of the trial 
				(excluding dropped initial strides)	"""
	spl_stride.drop(spl_stride.iloc[:,0:drop_n_first_strides-1], axis=1, 
		inplace=True)

	for stride in spl_stride.columns:
		spl_stride[stride]=interp_gaitprcent(spl_stride[stride],101)
	spl_stride=spl_stride.dropna()

	cor=spl_stride.corr() # computes correlation
	su=cor.sum(axis=1) # summed correlation with all the other strides

	var=spl_stride.std(axis=1)

	if stride_choice=="repmax":
		rep_max_idx=su[su.values==su.max()].index
		try:
			y=spl_stride[rep_max_idx[0]]
		except IndexError:
			if LOG_LEVEL<=LOG_WARNING:
				print("[WARNING]No representative stride, \
					setting values to zero")
			y=pd.Series(np.zeros(100)) 
	elif stride_choice=="mean":
		rep_index=su[su.values>su.mean()].index 
		""" mean value over the "good strides" (correlation greater than mean 
		correlation) to avoid artifacts due to a "bad stride" (typically 
		triping) """
		rep_strides=spl_stride.filter(items=rep_index)
		y=rep_strides.mean(axis=1)
	elif stride_choice in spl_stride.columns:
		y=spl_stride[stride_choice]
	else:
		if LOG_LEVEL<=LOG_ERROR:
			print("[ERROR]Stride choice:\t",stride_choice,
				"\n Available strides:",spl_stride.columns)
		raise ValueError
	return y,var