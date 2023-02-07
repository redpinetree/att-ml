import copy
import numpy as np

class site:
	def __init__(self):
		self.vol=1 #volume is initially 1
		self.virtual_flag=False
	def __repr__(self):
		return str(self.vol)

class virtual_site(site): #extends site by including parent data
	def __init__(self,p1,p2,sites):
		super().__init__()
		self.parents=(p1,p2)
		self.virtual_flag=True
	def __repr__(self):
		return str([self.vol,self.parents])

class bond:
	def __init__(self,v1,v2,w):
		self.v1=v1 #first vertex
		self.v2=v2 #second vertex
		self.orig_v1=v1 #original first vertex, for sort order in cmd approximation
		self.orig_v2=v2 #original second vertex, for sort order in cmd approximation
		self.w=w #weight matrix
		self.up=None #upstream site. if None, flag for contraction
		self.bmi=None
	def __repr__(self):
		# j=0.5*np.log(self.w[0,0]/self.w[0,1])
		j=-np.log(((1/self.w[0,0])-self.w.shape[0])/(self.w.shape[0]*(self.w.shape[0]-1)))
		# return str([self.v1,self.v2,self.w,self.bmi,self.up])
		return str([self.v1,self.v2,self.orig_v1,self.orig_v2,j,np.round(self.bmi,14),self.up])
	def calc_bmi(self):
		#normalize w
		#note: this doesn't seem to be necessary for the initialization step
		norm_check_ax0=np.allclose(np.sum(self.w,axis=0),self.w.shape[0]**-1,1e-10)
		norm_check_ax1=np.allclose(np.sum(self.w,axis=1),self.w.shape[1]**-1,1e-10)
		if not norm_check_ax0 or not norm_check_ax1:
			# print(np.sum(self.w,axis=0),np.sum(self.w,axis=1))
		# if True:
			# print("x:")
			# x=np.random.rand(self.w.shape[0])
			x=np.ones(self.w.shape[0])/self.w.shape[0] #seems to work better?
			x_old=np.zeros(self.w.shape[0])
			# print(x)
			while not np.allclose(x,x_old,rtol=1e-10,atol=1e-10):
				x_old=copy.deepcopy(x)
				x=(self.w.shape[1]/self.w.shape[0])*((self.w@((self.w.T@x)**-1))**-1)
				# print(x)
			# print("y:")
			# y=np.random.rand(self.w.shape[1])
			y=np.ones(self.w.shape[1])/self.w.shape[1]
			y_old=np.zeros(self.w.shape[1])
			# print(y)
			while not np.allclose(y,y_old,rtol=1e-10,atol=1e-10):
				y_old=copy.deepcopy(y)
				y=(self.w.shape[0]/self.w.shape[1])*((self.w.T@((self.w@y)**-1))**-1)
				# print(y)
			# print("original w:",self.w)
			# print(np.sum(self.w,axis=0),np.sum(self.w,axis=1))
			p_ij=np.diag(x)@self.w@np.diag(y)
			# print("new w:",p_ij)
			# print(np.sum(p_ij,axis=0),np.sum(p_ij,axis=1))
			p_ij/=np.sum(p_ij)
			# print(np.sum(p_ij,axis=0),np.sum(p_ij,axis=1))
			# print("p_ij:",p_ij)
			# print("")
		else: #skip if w is already normalized properly
			p_ij=copy.deepcopy(self.w)
		# p_ij=copy.deepcopy(self.w)
		p_i=np.sum(p_ij,axis=0) #marginals
		p_j=np.sum(p_ij,axis=1)
		with np.errstate(divide='ignore',invalid='ignore'): #if we have 0*ln(0), it is set to 0
			S_ij=-np.sum(np.nan_to_num(p_ij*np.log(p_ij)))
			S_i=-np.sum(np.nan_to_num(p_i*np.log(p_i)))
			S_j=-np.sum(np.nan_to_num(p_j*np.log(p_j)))
		self.bmi=S_i+S_j-S_ij
		# print(self,p_ij,p_i,p_j,S_ij,S_i,S_j)