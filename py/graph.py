import copy
import numpy as np

rng=np.random.default_rng()

class site:
	def __init__(self):
		self.vol=1 #volume is initially 1
	def __repr__(self):
		return str(self.vol)

class bond:
	def __init__(self,v1,v2,j):
		self.v1=v1 #first vertex
		self.v2=v2 #second vertex
		self.j=j #weight
		self.order=None #iteration (0-indexed) that the bond was processed. if None, flag for contraction
	def __repr__(self):
		return str([self.v1,self.v2,self.j,self.order])

def gen_hypercubic(d,l,dist,pbc=True,**kwargs):
	if type(l) in [list,tuple]:
		assert d==len(l)
	else:
		l=[l for i in range(d)]
	assert dist in ['gaussian','bimodal']
	if dist=='gaussian':
		if 'p' in kwargs.keys():
			print("Ignoring 'p' kwarg in gen_hypercubic...")
		mean=kwargs['mean'] if 'mean' in kwargs.keys() else 0
		std=kwargs['std'] if 'std' in kwargs.keys() else 1
		assert std>0
	if dist=='bimodal':
		if 'mean' in kwargs.keys():
			print("Ignoring 'mean' kwarg in gen_hypercubic...")
		if 'std' in kwargs.keys():
			print("Ignoring 'std' kwarg in gen_hypercubic...")
		p=kwargs['p'] if 'p' in kwargs.keys() else 0.5
		assert p>=0 and p<=1

	es=[]
	num_vs=np.prod(l)
	for v1 in range(num_vs):
		v1_idx=copy.deepcopy(v1)
		#identify position of site v1 on hypercubic lattice
		base_coords=[]
		for d_idx in range(d):
			base_coords.append(v1_idx%l[d_idx])
			v1_idx=v1_idx//l[d_idx]
		#add offset (+1 for nearest neighbor)
		for d_idx in range(d):
			y=copy.deepcopy(base_coords)
			#boundary condition
			if (not pbc) and (y[d_idx]+1)>=l[d_idx]:
				continue
			y[d_idx]=(y[d_idx]+1)%l[d_idx]
			#compute idx of v2
			v2=0
			for d_idx2 in range(d-1,-1,-1):
				v2=(v2*l[d_idx2])+y[d_idx2]
			if dist=='gaussian':
				j=rng.normal(loc=mean,scale=std)
			elif dist=='bimodal':
				j=rng.choice([1,-1],p=[p,1-p])
			if v2<v1:
				temp=v1
				v1=v2
				v2=temp
			es.append([v1,v2,int(j*1e5)])
	return num_vs,es

def save_graph(fn,num_vs,es):
	with open(fn,"w") as f:
		f.write("%d %d"%(num_vs,len(es)))
		for e in es:
			f.write("\n")
			f.write("%d %d %d"%tuple(e))

def load_graph(fn):
	with open(fn,"r") as f:
		lines=f.readlines()
	num_vs=int(lines[0].split(" ")[0])
	es=[] #vertex1, vertex2, weight
	for n in range(len(lines[1:])):
		v1,v2,j=(int(i) for i in lines[n+1].split(" "))
		v1-=1 #TODO: munge rudy output to 0-indexing
		v2-=1
		if v2<v1:
			temp=v1
			v1=v2
			v2=temp
		es.append([v1,v2,j/1e5])
	return num_vs,es