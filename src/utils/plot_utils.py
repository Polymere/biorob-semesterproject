import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import file_utils as fu

class barplot():
	values=[]

	def __init__(self):
		self.fig=plt.axes()
	def add_bar(self,label,value,meta):
		self.values.append[(label,value,meta)]
	def plot(self,do_sort=True):
		if do_sort:
			"""Sort by ascending value in meta
			"""
			self.values=sorted(self.values,key=lambda val:val[2])
		plt_count=0
		for val in self.values:
			plt.bar(plot_count,val[1])
			label_lst.append(val[0])
			plt_count+=1
		xt=np.arange(1,plt_count)
		plt.xticks(xt,label_lst)



def import_plot_triplet(cdir,metric,expected_meta=None):
	meta_file=fu.file_list(cdir,pattern="meta",file_format=".yaml")
	if len(meta_file)>1:
		print("Multiple meta files in ",cdir)
		meta_file=meta_file[0]
	proc_files=fu.file_list(cdir,pattern="processed",file_format=".csv",recursive=True)
	meta=yaml.load(open(meta_file,'r'))
	if "label" in meta.keys():
		label=meta["label"]
	else:
		if len(meta.keys())>1:
			print(" \n Multiple non label values in meta for ",cdir)
		for key,srt_val in meta.items():
			k=key
			label=round(srt_val, 3)
	if expected_meta is not None and k !=expected_meta:
		print("Meta param is",k,"should be",expected_meta)
	return


def barplot_vs_ref(gen_path,ref_path,metric,how="max"):
	ind_dirs=fu.dir_list(gen_path,pattern="ind")
	for ind in ind_dirs:
		proc_files=fu.file_list(ind,pattern="processed",file_format=".csv",recursive=True)
		meta_file=fu.file_list(ind,pattern="meta",file_format=".yaml")
		if len(meta_file)>1:
			print

	ref_file=fu.file_lsi	

if __name__=='__main__':
	mode=sys.argv[1]
	params=sys.argv[2:]