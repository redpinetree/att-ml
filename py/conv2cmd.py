import numpy as np
import sys,copy,os,time
import argparse

sys.path.append("./py_cmd")

def conv2cmd(q,old_bonds,old_sites):
	# print(old_bonds)
	# print(old_sites)
	old_n_sites=len(old_sites)
	sites=[]
	for i in range(len(old_sites)):
		sites.append(graph_cmd.site())
		sites[-1].vol=old_sites[i].vol
	# print(sites)
	d={i:i for (i,i) in ((i,i) for i in range(old_n_sites))}
	old_bonds.sort(key=lambda e: e.order,reverse=True)
	bonds=[]
	for i in range(len(old_bonds)):
		old_v1=old_bonds[-1-i].v1
		old_v2=old_bonds[-1-i].v2
		j=old_bonds[-1-i].j
		if d[old_v1]<d[old_v2]:
			sites.append(graph_cmd.virtual_site(d[old_v1],d[old_v2],sites))
		else:
			sites.append(graph_cmd.virtual_site(d[old_v2],d[old_v1],sites))
		sites[-1].vol=sites[d[old_v1]].vol if sites[d[old_v1]].vol>sites[d[old_v2]].vol else sites[d[old_v2]].vol
		w=np.zeros((q,q))
		for k in range(w.shape[0]):
			for l in range(w.shape[1]):
				# w[k,l]=0.25*(1+np.tanh(beta*j)*((2*(k==l))-1)) #ising spin glass
				# w[k,l]=0.25 #no coupling
				# w[k,l]=0.25*(1+((2*(k==l))-1)) #perfect correlation
				# if k==l: #ising spin glass
					# w[k,l]=1/(q+(q*(q-1)*np.exp(-2*beta*j)))
				# else:
					# w[k,l]=1/((q*np.exp(2*beta*j))+(q*(q-1)))
				if k==l: #potts spin glass
					w[k,l]=1/(q+(q*(q-1)*np.exp(-j)))
				else:
					w[k,l]=1/((q*np.exp(j))+(q*(q-1)))
		new_bond_v1=d[old_v1]
		new_bond_v2=d[old_v2]
		if new_bond_v1>new_bond_v2:
			temp=new_bond_v1
			new_bond_v1=new_bond_v2
			new_bond_v2=temp
		bonds.append(graph_cmd.bond(new_bond_v1,new_bond_v2,w))
		bonds[-1].up=len(sites)-1
		d[old_v1]=len(sites)-1
		d[old_v2]=len(sites)-1
		bonds[-1].calc_bmi()
	bonds.reverse()
	# print(bonds)
	# print(sites)
	return bonds,sites

parser=argparse.ArgumentParser()
parser.add_argument("q",type=int)
parser.add_argument("input")
parser.add_argument("output")
args=parser.parse_args()
q=args.q
file=args.input
data=np.load(file,allow_pickle=True)
snapshots=[]
for file in data.files:
	#old scheme for algorithm 1
	import graph
	beta=data[file][0]
	old_bonds=data[file][1]
	old_sites=data[file][2]
	
	#new scheme for cmd, assumes simple cmd function!
	import graph_cmd
	# class site:
		# def __init__(self):
			# self.vol=1 #volume is initially 1
			# self.virtual_flag=False
		# def __repr__(self):
			# return str(self.vol)

	# class virtual_site(site): #extends site by including parent data
		# def __init__(self,p1,p2,sites):
			# super().__init__()
			# self.parents=(p1,p2)
			# self.virtual_flag=True
		# def __repr__(self):
			# return str([self.vol,self.parents])

	# class bond:
		# def __init__(self,v1,v2,w):
			# self.v1=v1 #first vertex
			# self.v2=v2 #second vertex
			# self.w=w #weight matrix
			# self.up=True #flag for contraction
			# self.bmi=None
		# def __repr__(self):
			# # j=0.5*np.log(self.w[0,0]/self.w[0,1])
			# j=-np.log(((1/self.w[0,0])-self.w.shape[0])/(self.w.shape[0]*(self.w.shape[0]-1)))
			# # return str([self.v1,self.v2,self.w,self.bmi,self.up])
			# return str([self.v1,self.v2,j,self.bmi,self.up])
		# def calc_bmi(self):
			# #normalize w
			# #note: this doesn't seem to be necessary for the initialization step
			# norm_check_ax0=np.allclose(np.sum(self.w,axis=0),self.w.shape[0]**-1,1e-10)
			# norm_check_ax1=np.allclose(np.sum(self.w,axis=1),self.w.shape[1]**-1,1e-10)
			# if not norm_check_ax0 or not norm_check_ax1:
				# # print(np.sum(self.w,axis=0),np.sum(self.w,axis=1))
			# # if True:
				# # print("x:")
				# # x=np.random.rand(self.w.shape[0])
				# x=np.ones(self.w.shape[0])/self.w.shape[0] #seems to work better?
				# x_old=np.zeros(self.w.shape[0])
				# # print(x)
				# while not np.allclose(x,x_old,rtol=1e-10,atol=1e-10):
					# x_old=copy.deepcopy(x)
					# x=(self.w.shape[1]/self.w.shape[0])*((self.w@((self.w.T@x)**-1))**-1)
					# # print(x)
				# # print("y:")
				# # y=np.random.rand(self.w.shape[1])
				# y=np.ones(self.w.shape[1])/self.w.shape[1]
				# y_old=np.zeros(self.w.shape[1])
				# # print(y)
				# while not np.allclose(y,y_old,rtol=1e-10,atol=1e-10):
					# y_old=copy.deepcopy(y)
					# y=(self.w.shape[0]/self.w.shape[1])*((self.w.T@((self.w@y)**-1))**-1)
					# # print(y)
				# # print("original w:",w)
				# p_ij=np.diag(x)@self.w@np.diag(y)
				# # print("new w:",p_ij)
				# # print(np.sum(p_ij,axis=0),np.sum(p_ij,axis=1))
				# p_ij/=np.sum(p_ij)
				# # print("p_ij:",p_ij)
			# else: #skip if w is already normalized properly
				# p_ij=copy.deepcopy(self.w)
			# # p_ij=copy.deepcopy(self.w)
			# p_i=np.sum(p_ij,axis=0) #marginals
			# p_j=np.sum(p_ij,axis=1)
			# with np.errstate(divide='ignore',invalid='ignore'): #if we have 0*ln(0), it is set to 0
				# S_ij=-np.sum(np.nan_to_num(p_ij*np.log(p_ij)))
				# S_i=-np.sum(np.nan_to_num(p_i*np.log(p_i)))
				# S_j=-np.sum(np.nan_to_num(p_j*np.log(p_j)))
			# self.bmi=S_i+S_j-S_ij
			# # print(self,p_ij,p_i,p_j,S_ij,S_i,S_j)

	bonds,sites=conv2cmd(q,old_bonds,old_sites)
	snapshots.append([beta,bonds,sites])

if args.output is not None:
	path=os.path.split(args.output)[0]
	if path!="" and not os.path.exists(path):
		os.mkdir(os.path.split(args.output)[0])
	fn=args.output
else:
	fn=os.path.splitext(os.path.basename(args.input))[0]
snapshots=np.asarray(snapshots,dtype=object)
np.savez(fn,*snapshots)