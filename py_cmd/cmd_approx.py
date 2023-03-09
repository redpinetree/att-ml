import numpy as np
import sys,copy,os,time
import argparse
import graphviz

import graph_cmd

import struct
def to_bin(x):
	return ''.join('{:0>8b}'.format(c) for c in struct.pack('!d',x))

def make_graph(sites,bonds,name):
	g=graphviz.Graph(name=name,format='png',engine="dot")
	g.attr(overlap="false")
	for i in range(len(sites)):
		g.node("t"+str(i),str(sites[i].vol),shape="circle")
	for i in range(len(bonds)):
		g.edge("t"+str(bonds[i].v1),"t"+str(bonds[i].v2),label=str(bonds[i].j)[:5],labelfontsize="2")
	g.render()
img_count=0

def f(site1,s1,site2,s2): #DEFAULT confluent mapping function reducing to old method
	#s1,s2 are 0,1
	return s1 if site1.vol>=site2.vol else s2

parser=argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("q",type=int) #control for potts spin glass (q=2,3,4)
parser.add_argument("min_beta",type=float)
parser.add_argument("max_beta",type=float)
parser.add_argument("step_beta",type=float)
parser.add_argument("-o","--output")
parser.add_argument("--sort-by-j",action='store_true')
args=parser.parse_args()

assert args.min_beta<=args.max_beta
beta=args.min_beta
q=args.q

snapshots=[]
times=[]
while beta<=args.max_beta:
	print("beta=%g"%beta)
	#file input
	#TODO: avoid re-reading file for multiple betas
	# print("----initialization----")
	with open(args.input,"r") as file:
		lines=file.readlines()
	n_sites=int(lines[0].split(" ")[0])
	sites=[graph_cmd.site() for i in range(n_sites)] #TODO: figure out what sites contain if anything
	bonds=[] #vertex1, vertex2, weight, todo
	for n in range(len(lines[1:])):
		v1,v2,j=(int(i) for i in lines[n+1].split(" "))
		v1-=1
		v2-=1
		if v2<v1:
			temp=v1
			v1=v2
			v2=temp
		j/=1e5
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
					w[k,l]=1/(q+(q*(q-1)*np.exp(-beta*j)))
				else:
					w[k,l]=1/((q*np.exp(beta*j))+(q*(q-1)))
		bonds.append(graph_cmd.bond(v1,v2,w))
	# print(sites)
	# print(bonds)
	for b in bonds:
		b.calc_bmi()
	# print(sites)
	# print(bonds)
	start=time.time()
	#bonds[-1] is the heaviest (bmi) bond that is yet to be contracted
	if args.sort_by_j:
		bonds.sort(key=lambda b: (b.up is None,b.up,np.abs(0.5*np.log(b.w[0,0]/b.w[0,1])),2-(sites[b.v1].virtual_flag+sites[b.v2].virtual_flag),b.orig_v1,b.orig_v2))
	else:
		bonds.sort(key=lambda b: (b.up is None,b.up,np.round(b.bmi,14),2-(sites[b.v1].virtual_flag+sites[b.v2].virtual_flag),b.orig_v1,b.orig_v2))
	# print(bonds)
	
	#graph deformation
	# print("----graph deformation----")
	while bonds[-1].up is None:
		# make_graph(sites,bonds,"tree%d"%img_count)
		# img_count+=1
		#update master vertex volume for default cm func
		# (master,slave)=(bonds[-1].v2,bonds[-1].v1) if sites[bonds[-1].v1].vol<sites[bonds[-1].v2].vol else (bonds[-1].v1,bonds[-1].v2)
		#EXP: (master,slave) determined by traversing the tree until a physical spin is obtained
		v1_vol=sites[bonds[-1].v1].vol
		v1_orig_idx=bonds[-1].orig_v1
		v2_vol=sites[bonds[-1].v2].vol
		v2_orig_idx=bonds[-1].orig_v2
		if v1_vol==v2_vol:
			(master,slave)=(bonds[-1].v1,bonds[-1].v2) if v1_orig_idx<=v2_orig_idx else (bonds[-1].v2,bonds[-1].v1)
		else:
			(master,slave)=(bonds[-1].v1,bonds[-1].v2) if v1_vol>=v2_vol else (bonds[-1].v2,bonds[-1].v1)
		sites[master].vol+=sites[slave].vol
		# print(bonds[-1])
		# print("v1_orig=%d,v2_orig=%d"%(v1_orig_idx,v2_orig_idx))
		# print("master=%d,slave=%d"%(master,slave))
		#create new virtual site
		sites.append(graph_cmd.virtual_site(bonds[-1].v1,bonds[-1].v2))
		sites[-1].vol=copy.deepcopy(sites[master].vol)
		# sites[-1].vol+=sites[slave].vol
		#update volume of physical site associated with new virtual site for algorithm 1
		current_site=master
		next_site=master
		while sites[current_site].virtual_flag is True:
			current_site=next_site
			if sites[current_site].virtual_flag:
				next_site=sites[current_site].parents[0] if sites[sites[current_site].parents[0]].vol>=sites[sites[current_site].parents[1]].vol else sites[current_site].parents[1]
				sites[next_site].vol=copy.deepcopy(sites[current_site].vol)
		virtual_idx=len(sites)-1
		# print(sites)
		
		#sort so that first elements are connected to bond vertices, with current bond at the end
		#the spin cluster is described by the first cluster_size bonds and the current bond
		# print(bonds)
		current=bonds.pop()
		bonds.sort(key=lambda e: (e.v1==current.v1 or e.v1==current.v2 or e.v2==current.v1 or e.v2==current.v2,e.up is None),reverse=True)
		bonds.append(current)
		# print(bonds)
		# print("current working bond:",bonds[-1])
		cluster_size=0
		bonds_group1=[] #idxs of bonds originially connected to bonds[-1].v1
		bonds_group2=[] #idxs of bonds originially connected to bonds[-1].v2
		for i in range(len(bonds)-1):
			if (bonds[i].v1==bonds[-1].v1 or bonds[i].v2==bonds[-1].v1) and bonds[i].up is None:
				cluster_size+=1
				bonds_group1.append(i)
			elif (bonds[i].v1==bonds[-1].v2 or bonds[i].v2==bonds[-1].v2) and bonds[i].up is None:
				cluster_size+=1
				bonds_group2.append(i)
			else: #since the list is sorted, break when the next bonds no longer contain the bond vertices or are all done
				break
		# print("bonds_group1:",[bonds[i] for i in bonds_group1])
		# print("bonds_group2:",[bonds[i] for i in bonds_group2])
		
		#confluent mapping decomposition
		#TODO: how to initialize new weights with r_k rank instead of r?
		for i in range(cluster_size):
			old=copy.deepcopy(bonds[-1])
			old2=copy.deepcopy(bonds[i])
			change=False
			# if (bonds[i].v1==bonds[-1].v1 or bonds[i].v1==bonds[-1].v2):
			if (bonds[i].v1==bonds[-1].v1 or bonds[i].v1==bonds[-1].v2) and bonds[i].up is None:
				bonds[i].orig_v1=bonds[-1].orig_v1 if bonds[-1].v1==master else bonds[-1].orig_v2
				#reconnect
				bonds[i].v1=bonds[i].v2 #move v2 to v1
				bonds[i].v2=virtual_idx #because bonds[i] vertices<virtual_idx always
				temp=bonds[i].orig_v1 #swap orig_v2 to orig_v1
				bonds[i].orig_v1=bonds[i].orig_v2
				bonds[i].orig_v2=temp
				#TAKE CARE OF ROW/COL ORDER!
				#update bond weight
				# bonds[i].j=np.arctanh(np.tanh(bonds[-1].j)*np.tanh(bonds[i].j)) #update bond weight
				change=True
			# elif (bonds[i].v2==bonds[-1].v1 or bonds[i].v2==bonds[-1].v2)==slave:
			elif (bonds[i].v2==bonds[-1].v1 or bonds[i].v2==bonds[-1].v2) and bonds[i].up is None:
				bonds[i].orig_v2=bonds[-1].orig_v1 if bonds[-1].v1==master else bonds[-1].orig_v2
				#reconnect
				bonds[i].v2=virtual_idx
				#update bond weight
				# bonds[i].j=np.arctanh(np.tanh(bonds[-1].j)*np.tanh(bonds[i].j)) #update bond weight
				change=True
			if change:
				# print(bonds[-1],old2,bonds[i])
				pass
		bonds[-1].up=len(sites)-1 #mark as done by setting to the upstream site
		# print(sites)
		# print(bonds)
		# print("")
		
		#calculate pair_{ij}
		pair_ij=np.empty_like(bonds[-1].w) #pair distribution of considered bond
		for i in range(pair_ij.shape[0]):
			for j in range(pair_ij.shape[1]):
				g1=np.prod([np.sum(bonds[a].w,axis=0 if bonds[a].v1==bonds[-1].v1 else 1)[i] for a in bonds_group1])
				g2=np.prod([np.sum(bonds[b].w,axis=0 if bonds[b].v1==bonds[-1].v2 else 1)[j] for b in bonds_group2])
				pair_ij[i,j]=bonds[-1].w[i,j]*g1*g2
				# print(i,[np.sum(bonds[k].w,axis=0 if bonds[k].v1==bonds[-1].v1 else 1)[i] for k in bonds_group1])il
				# print(j,[np.sum(bonds[k].w,axis=0 if bonds[k].v1==bonds[-1].v2 else 1)[i] for k in bonds_group2])
				# print(g1,g2)
		# print("pair_ij:",pair_ij)
		# print("pair_ij sum (Z):",np.sum(pair_ij))
		# pair_ij=0.5*(pair_ij+pair_ij.T)
		pair_ij/=np.sum(pair_ij)
		# print("normalized pair_ij:",pair_ij)
		
		#TODO: calculate pair'_{ij}^{env} (uses cm func), can be reused for pair'_{ki_mu}^{env}
		#TODO: merge with prev nested loop
		pair_ij_env=np.empty_like(bonds[-1].w) #pair distribution of considered bond based on S_c'
		for i in range(pair_ij_env.shape[0]):
			for j in range(pair_ij_env.shape[1]):
				k=f(sites[bonds[-1].v1],i,sites[bonds[-1].v2],j) #select where output is added to
				g1=np.prod([np.sum(bonds[a].w,axis=0)[k] for a in bonds_group1]) #new site is always v2 so sum over axis 0
				g2=np.prod([np.sum(bonds[b].w,axis=0)[k] for b in bonds_group2]) #new site is always v2 so sum over axis 0
				pair_ij_env[i,j]=g1*g2
				# print(i,[np.sum(bonds[k].w,axis=0 if bonds[k].v1==bonds[-1].v1 else 1)[i] for k in bonds_group1])
				# print(j,[np.sum(bonds[k].w,axis=0 if bonds[k].v1==bonds[-1].v2 else 1)[i] for k in bonds_group2])
				# print(g1,g2)
		# print("pair_ij_env:",pair_ij_env)
		# print("w_ij*pair_ij_env:",bonds[-1].w*pair_ij_env)
		# print("w_ij*pair_ij_env sum (Z'):",np.sum(bonds[-1].w*pair_ij_env))
		# pair_ij_env=0.5*(pair_ij_env+pair_ij_env.T)
		pair_ij_env/=np.sum(bonds[-1].w*pair_ij_env)
		# print("normalized pair_ij_env:",pair_ij_env)
		# print("w_ij:",bonds[-1].w)
		# print("w_ij (sum):",np.sum(bonds[-1].w))
		# print("w_ij (sums):",np.sum(bonds[-1].w,axis=0),np.sum(bonds[-1].w,axis=1))
		# print("next w'_ij:",pair_ij/pair_ij_env)
		bonds[-1].w=pair_ij/pair_ij_env
		# bonds[-1].w=0.5*(bonds[-1].w+bonds[-1].w.T)
		# print("w_ij:",bonds[-1].w)
		# print("w_ij (sum):",np.sum(bonds[-1].w))
		# print("w_ij (sums):",np.sum(bonds[-1].w,axis=0),np.sum(bonds[-1].w,axis=1))
		# print("")
		
		# f_arr=np.zeros([q,q])
		# for i in range(q):
			# for j in range(q):
				# f_arr[i,j]=f(sites[bonds[-1].v1],i,sites[bonds[-1].v2],j)
		# print(f_arr)
		
		#TODO: calculate pair_{k{i_\mu}}+env and pair_{k{j_\nu}}+env (uses cm func)
		pair_kis=[]
		pair_ki_envs=[]
		for m in bonds_group1:
			pair_ki=np.zeros(bonds[m].w.shape) #pair distribution of neighbor bond
			pair_ki_env=np.zeros(bonds[m].w.shape) #environment of neighbor bond
			for i in range(bonds[-1].w.shape[0]):
				for j in range(bonds[-1].w.shape[1]):
					k=f(sites[bonds[-1].v1],i,sites[bonds[-1].v2],j) #select where output is added to
					g1=np.prod([np.sum(bonds[a].w,axis=0 if bonds[a].v1==bonds[-1].v1 else 1)[i] for a in bonds_group1 if a!=m]) #exclude sum over m
					g2=np.prod([np.sum(bonds[b].w,axis=0 if bonds[b].v1==bonds[-1].v2 else 1)[j] for b in bonds_group2])
					# print(m,imu,i,j,k,bonds[-1].w[i,j],bonds[m].w[imu,i],g1,g2)
					for imu in range(pair_ki.shape[0]):
						pair_ki[imu,k]+=bonds[-1].w[i,j]*bonds[m].w[imu,i]*g1*g2 #w[imu,i] ASSUMES SYMMETRIC W
						pair_ki_env[imu,k]+=pair_ij_env[i,j]*bonds[-1].w[i,j]/np.sum(bonds[m].w,axis=0)[k]
					# print(i,[np.sum(bonds[a].w,axis=0 if bonds[a].v1==bonds[-1].v1 else 1)[i] for a in bonds_group1 if a!=m])
					# print(j,[np.sum(bonds[b].w,axis=0 if bonds[b].v1==bonds[-1].v2 else 1)[j] for b in bonds_group2])
					# print(g1,g2)
			if bonds[-1].v1==slave:
				test=bonds[m].w
				j_current=-np.log(((1/bonds[-1].w[0,0])-bonds[-1].w.shape[0])/(bonds[-1].w.shape[0]*(bonds[-1].w.shape[0]-1)))
				j_current_bond=-np.log(((1/bonds[m].w[0,0])-bonds[m].w.shape[0])/(bonds[m].w.shape[0]*(bonds[m].w.shape[0]-1)))
				num=np.exp(j_current_bond+j_current)+(q-1)
				denom=np.exp(j_current_bond)+np.exp(j_current)+(q-2)
				test_j=np.log(num/denom)
				for i in range(q):
					for j in range(q):
						test[i,j]=(1/(q+(q*(q-1)*np.exp(-test_j)))) if i==j else (1/((q*np.exp(test_j))+(q*(q-1))))
			else:
				test=bonds[m].w
			# print("rg_i:")
			# print(test)
			# print("opt_i:")
			# print(pair_ki/np.sum(pair_ki))
			a=pair_ki/np.sum(pair_ki)
			# print("%g %g"%(pair_ki[0,0]-pair_ki[1,1],pair_ki[0,1]-pair_ki[1,0]))
			# print("%g %g"%(pair_ki_env[0,0]-pair_ki_env[1,1],pair_ki_env[0,1]-pair_ki_env[1,0]))
			pair_kis.append(pair_ki)
			pair_ki_envs.append(pair_ki_env)
		# print("pair_kis:",pair_kis)
		# print("pair_ki sums:",[np.sum(pair_kis[m]) for m in range(len(bonds_group1))])
		# pair_kis=[0.5*(pair_kis[m]+pair_kis[m].T) for m in range(len(bonds_group1))]
		# pair_ki_envs=[0.5*(pair_ki_envs[m]+pair_ki_envs[m].T) for m in range(len(bonds_group1))]
		pair_kis=[pair_kis[m]/np.sum(pair_kis[m]) for m in range(len(bonds_group1))]
		# print("normalized pair_kis:",pair_kis)
		# print("normalized pair_ki_envs:",pair_ki_envs)
		# print("w_kis:",[bonds[m].w for m in bonds_group1])
		# print("next w'_kis:",[pair_kis[m]/pair_ki_envs[m] for m in range(len(bonds_group1))])
		for m in range(len(bonds_group1)):
			bonds[bonds_group1[m]].w=pair_kis[m]/pair_ki_envs[m]
		# print("")
		
		pair_kjs=[]
		pair_kj_envs=[]
		for n in bonds_group2:
			pair_kj=np.zeros(bonds[n].w.shape) #pair distribution of neighbor bond
			pair_kj_env=np.zeros(bonds[n].w.shape) #environment of neighbor bond
			for i in range(bonds[-1].w.shape[0]):
				for j in range(bonds[-1].w.shape[1]):
					k=f(sites[bonds[-1].v1],i,sites[bonds[-1].v2],j) #select where output is added to
					# for a in bonds_group1:
						# print("g1_contrib: %f"%np.sum(bonds[a].w,axis=0 if bonds[a].v1==bonds[-1].v1 else 1)[i])
					g1=np.prod([np.sum(bonds[a].w,axis=0 if bonds[a].v1==bonds[-1].v1 else 1)[i] for a in bonds_group1])
					# for b in bonds_group2:
						# if b==n:
							# continue
						# print("g2_contrib: %f"%np.sum(bonds[b].w,axis=0 if bonds[b].v1==bonds[-1].v2 else 1)[j])
					g2=np.prod([np.sum(bonds[b].w,axis=0 if bonds[b].v1==bonds[-1].v2 else 1)[j] for b in bonds_group2 if b!=n]) #exclude sum over n
					# print("g1*g2: %f"%(g1*g2))
					# print(n,jnu,i,j,k,bonds[-1].w[i,j],bonds[n].w[jnu,j],g1,g2)
					for jnu in range(pair_kj.shape[0]):
						# print("pair_kj[%d,%d] partial: %g"%(jnu,k,bonds[n].w[jnu,j]))
						# print("pair_kj[%d,%d] contrib: %g"%(jnu,k,bonds[-1].w[i,j]*bonds[n].w[jnu,j]*g1*g2))
						pair_kj[jnu,k]+=bonds[-1].w[i,j]*bonds[n].w[jnu,j]*g1*g2 #w[jnu,j] ASSUMES SYMMETRIC W
						pair_kj_env[jnu,k]+=pair_ij_env[i,j]*bonds[-1].w[i,j]/np.sum(bonds[n].w,axis=0)[k]
						# print(i,[np.sum(bonds[a].w,axis=0 if bonds[a].v1==bonds[-1].v1 else 1)[i] for a in bonds_group1 if a!=m])
						# print(j,[np.sum(bonds[b].w,axis=0 if bonds[b].v1==bonds[-1].v2 else 1)[j] for b in bonds_group2])
						# print(g1,g2)
			if bonds[-1].v2==slave:
				test=bonds[n].w
				j_current=-np.log(((1/bonds[-1].w[0,0])-bonds[-1].w.shape[0])/(bonds[-1].w.shape[0]*(bonds[-1].w.shape[0]-1)))
				j_current_bond=-np.log(((1/bonds[n].w[0,0])-bonds[n].w.shape[0])/(bonds[n].w.shape[0]*(bonds[n].w.shape[0]-1)))
				num=np.exp(j_current_bond+j_current)+(q-1)
				denom=np.exp(j_current_bond)+np.exp(j_current)+(q-2)
				test_j=np.log(num/denom)
				for i in range(q):
					for j in range(q):
						test[i,j]=(1/(q+(q*(q-1)*np.exp(-test_j)))) if i==j else (1/((q*np.exp(test_j))+(q*(q-1))))
			else:
				test=bonds[n].w
			# print("rg_j:")
			# print(test)
			# print("opt_j:")
			# print(pair_kj/np.sum(pair_kj))
			a=pair_kj/np.sum(pair_kj)
			# print("%g %g"%(pair_kj[0,0]-pair_kj[1,1],pair_kj[0,1]-pair_kj[1,0]))
			# print("%g %g"%(pair_kj_env[0,0]-pair_kj_env[1,1],pair_kj_env[0,1]-pair_kj_env[1,0]))
			pair_kjs.append(pair_kj)
			pair_kj_envs.append(pair_kj_env)
		# print("pair_kjs:",pair_kjs)
		# print("pair_kj sums:",[np.sum(pair_kjs[n]) for n in range(len(bonds_group2))])
		# pair_kjs=[0.5*(pair_kjs[n]+pair_kjs[n].T) for n in range(len(bonds_group2))]
		# pair_kj_envs=[0.5*(pair_kj_envs[n]+pair_kj_envs[n].T) for n in range(len(bonds_group2))]
		pair_kjs=[pair_kjs[n]/np.sum(pair_kjs[n]) for n in range(len(bonds_group2))]
		# print("normalized pair_kjs:",pair_kjs)
		# print("normalized pair_kj_envs:",pair_kj_envs)
		# print("w_kjs:",[bonds[n].w for n in bonds_group2])
		# print("next w'_kjs:",[pair_kjs[n]/pair_kj_envs[n] for n in range(len(bonds_group2))])
		for n in range(len(bonds_group2)):
			bonds[bonds_group2[n]].w=pair_kjs[n]/pair_kj_envs[n]
		# print("")
		
		#update bmis in cluster (this is done before merging since merges are not common and the merging process uses a different sorting order)
		bonds[-1].calc_bmi()
		for i in bonds_group1:
			bonds[i].calc_bmi()
		for i in bonds_group2:
			bonds[i].calc_bmi()
		#merge double bonds
		bonds.sort(key=lambda e: (e.v1,e.v2,e.orig_v1,e.orig_v2)) #sort by vertices
		remove_idxs=[]
		for i in range(len(bonds)):
			for j in range(i+1,len(bonds)):
				if bonds[i].v1==bonds[j].v1:
					if bonds[i].v2==bonds[j].v2:
						# print("merging %d to %d"%(j,i))
						# print(bonds[i],bonds[j])
						bonds[i].w*=bonds[j].w #elementwise product of ws on both bonds
						bonds[i].w/=np.sum(bonds[i].w) #proper normalization
						bonds[i].calc_bmi() #update bmi
						remove_idxs.append(j)
						# print(bonds[i])
				else: #since the list is sorted, go to next i coord when no matches appear in the i coord
					break
		remove_idxs.sort(reverse=True)
		for i in remove_idxs:
			del bonds[i]
			#also remove from bonds_group1 or bonds_group2 so no issues in bmi update
			if i in bonds_group1:
				bonds_group1.remove(i)
			elif i in bonds_group2:
				bonds_group2.remove(i)
		if args.sort_by_j:
			bonds.sort(key=lambda b: (b.up is None,b.up,np.abs(0.5*np.log(b.w[0,0]/b.w[0,1])),2-(sites[b.v1].virtual_flag+sites[b.v2].virtual_flag),b.orig_v1,b.orig_v2))
		else:
			bonds.sort(key=lambda b: (b.up is None,b.up,np.round(b.bmi,14),2-(sites[b.v1].virtual_flag+sites[b.v2].virtual_flag),b.orig_v1,b.orig_v2))
		# sys.exit(1)
		# print(bonds)
		# print(sites)
		# print("")

	print(bonds)
	print(sites)
	# # make_graph(sites,bonds,"tree%d"%img_count)
	# for b in bonds:
		# print(b.w[0,0]==b.w[1,1],b.w)

	snapshots.append([beta,bonds,sites])
	end=time.time()
	elapsed=end-start
	times.append(elapsed)
	print("Time elapsed for this beta: %f s"%times[-1])
	beta+=args.step_beta

print("Total time elapsed for this run: %f s"%np.sum(times))
print("Average time elapsed for this run per beta: %f+/-%f s"%(np.mean(times),np.std(times)))
if args.output is not None:
	path=os.path.split(args.output)[0]
	if path!="" and not os.path.exists(path):
		os.mkdir(os.path.split(args.output)[0])
	fn=args.output
else:
	fn=os.path.splitext(os.path.basename(args.input))[0]
snapshots=np.asarray(snapshots,dtype=object)
np.savez(fn,*snapshots)
