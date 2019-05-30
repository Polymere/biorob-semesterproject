import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def plot_correlation_window(s1,s2,nwind,save=False):
	if len(s1)!=len(s2):
		print("Series don't have the same size")
		s=min((len(s1),len(s2)))
		raise('RuntimeWarning')
	else:
		s=len(s1)
	wind_size=int(s/nwind)
	corr_df=pd.DataFrame(index=pd.RangeIndex(nwind),columns=["prcent","corre"])
	for wind in range(nwind):
		start=wind*wind_size
		stop=(wind+1)*wind_size
		r=s1[start:stop].corr(s2[start:stop])
		corr_df.corre.loc[wind]=r
		corr_df.prcent.loc[wind]=(start+stop)/2
		if r<0:
			c='r'
		else:
			c='g'
		a=abs(r)
		if a<0.25:
			a=0.5
	plt.bar((start+stop)/2,r,wind_size*0.95,color=c,alpha=a)
	plt.show()

def plot_mean_std_fill(mean,std,color,ax=None,ratio=1):
	if ax is None:
		ax=plt.axes()
	if ratio>1:
		ratio=1

	max_idx=int(len(mean)*ratio)
	x_range=range(max_idx)
	mean_smp=mean[0:max_idx]
	if std is None:
		std=np.zeros_like(mean_smp)
	std_smp=std[0:max_idx]
	ax.plot(x_range,mean_smp,color=color)
	ax.fill_between(x_range,mean_smp-std_smp,mean_smp+std_smp,alpha=0.4,color=color)
	return ax

