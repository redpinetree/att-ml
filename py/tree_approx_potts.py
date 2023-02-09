import numpy as np
import sys,copy,os,time
import argparse
import graphviz

import graph

def make_graph(sites,bonds,name):
	g=graphviz.Graph(name=name,format='png',engine="dot")
	g.attr(overlap="false")
	for i in range(len(sites)):
		g.node("t"+str(i),str(sites[i].vol),shape="circle")
	for i in range(len(bonds)):
		g.edge("t"+str(bonds[i].v1),"t"+str(bonds[i].v2),label=str(bonds[i].j)[:5],labelfontsize="2")
	g.render()
img_count=0

def bmi(q,k): #bond mutual information for q-state potts model
	z=(q*np.exp(k))+(q*(q-1))
	return (2*np.log(q))+(q*((np.exp(k)*(k-np.log(z)))-((q-1)*np.log(z)))/z)

parser=argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("q",type=int)
parser.add_argument("min_beta",type=float)
parser.add_argument("max_beta",type=float)
parser.add_argument("step_beta",type=float)
parser.add_argument("-o","--output")
parser.add_argument("--sort-by-j",action='store_true')
args=parser.parse_args()

q=args.q

assert args.min_beta<=args.max_beta
beta=args.min_beta

vs,es=graph.load_graph(args.input)
orig_sites=copy.deepcopy(vs) #volumes
orig_bonds=copy.deepcopy(es) #vertex1, vertex2, weight, order

snapshots=[]
times=[]
while beta<=args.max_beta:
	print("beta=%g"%beta)
	bonds=copy.deepcopy(orig_bonds)
	sites=copy.deepcopy(orig_sites)
	for i in range(len(bonds)):
		bonds[i].j*=beta
	start=time.time()
	#bonds[-1] is the heaviest bond that is yet to be contracted
	if args.sort_by_j:
		bonds.sort(key=lambda e: (e.order is None,np.abs(e.j),e.v1,e.v2))
	else:
		# bonds.sort(key=lambda e: (e.order is None,np.round(bmi(q,e.j),14),e.v1,e.v2))
		bonds.sort(key=lambda e: (e.order is None,bmi(q,e.j),e.v1,e.v2))
	# print(bonds)
	# print(sites)
	
	#if graph has 2 points connected by 2 bonds, merge them (only for weird/small graphs
	# bonds.sort(key=lambda e: (e.v1,e.v2)) #sort by vertices
	# remove_idxs=[]
	# for i in range(len(bonds)):
		# for j in range(i+1,len(bonds)):
			# if bonds[i].v1==bonds[j].v1:
				# if bonds[i].v2==bonds[j].v2:
					# # print("merging %d to %d"%(j,i))
					# # print(bonds[i],bonds[j])
					# bonds[i].j+=bonds[j].j
					# remove_idxs.append(j)
					# # print(bonds[i])
			# else: #since the list is sorted, go to next i coord when no matches appear in the i coord
				# break
	# remove_idxs.sort(reverse=True)
	# for i in remove_idxs:
		# del bonds[i]
	# print(bonds)
	# print(sites)
	
	#graph deformation
	iteration=0
	while bonds[-1].order is None:
		# make_graph(sites,bonds,"tree%d"%img_count)
		# img_count+=1
		(master,slave)=(bonds[-1].v2,bonds[-1].v1) if sites[bonds[-1].v1].vol<sites[bonds[-1].v2].vol else (bonds[-1].v1,bonds[-1].v2)
		sites[master].vol+=sites[slave].vol #update master vertex volume\
		# print("")
		# print(bonds[-1],master,slave)
		current=bonds.pop()
		bonds.sort(key=lambda e: (e.order is None,e.v1==slave or e.v2==slave),reverse=True)
		bonds.append(current)
		# print(bonds)
		for i in range(len(bonds)-1):
			old=copy.deepcopy(bonds[-1])
			old2=copy.deepcopy(bonds[i])
			change=False
			# if bonds[i].v1==slave:
			if bonds[i].v1==slave and bonds[i].order is None:
				bonds[i].v1=master #reconnect
				if bonds[i].v2<bonds[i].v1: #ordering
					temp=bonds[i].v1
					bonds[i].v1=bonds[i].v2
					bonds[i].v2=temp
				bonds[i].j=np.log((np.exp(bonds[-1].j+bonds[i].j)+(q-1))/(np.exp(bonds[-1].j)+np.exp(bonds[i].j)+(q-2))) #update bond weight
				change=True
			# elif bonds[i].v2==slave:
			elif bonds[i].v2==slave and bonds[i].order is None:
				bonds[i].v2=master
				if bonds[i].v2<bonds[i].v1:
					temp=bonds[i].v1
					bonds[i].v1=bonds[i].v2
					bonds[i].v2=temp
				bonds[i].j=np.log((np.exp(bonds[-1].j+bonds[i].j)+(q-1))/(np.exp(bonds[-1].j)+np.exp(bonds[i].j)+(q-2))) #update bond weight
				change=True
			else: #since the list is sorted, break when the bonds no longer contain the slave vertex or are all done
				break
			if change:
				# print(bonds[-1],old2,bonds[i])
				pass
		bonds[-1].order=iteration
		# print(bonds)
		# print(sites)
		#merge identical bonds
		bonds.sort(key=lambda e: (e.v1,e.v2)) #sort by vertices
		remove_idxs=[]
		for i in range(len(bonds)):
			for j in range(i+1,len(bonds)):
				if bonds[i].v1==bonds[j].v1:
					if bonds[i].v2==bonds[j].v2:
						# print("merging %d to %d"%(j,i))
						# print(bonds[i],bonds[j])
						bonds[i].j+=bonds[j].j
						remove_idxs.append(j)
						# print(bonds[i])
				else: #since the list is sorted, go to next i coord when no matches appear in the i coord
					break
		# print(bonds)
		# print(sites)
		remove_idxs.sort(reverse=True)
		for i in remove_idxs:
			del bonds[i]
		# sys.exit(1)
		if args.sort_by_j:
			bonds.sort(key=lambda e: (e.order is None,np.abs(e.j),e.v1,e.v2))
		else:
			bonds.sort(key=lambda e: (e.order is None,bmi(q,e.j),e.v1,e.v2))
		iteration+=1
		# print(sites)
		# print(bonds)
		# print("")
		# sys.exit(1)

	end=time.time()
	elapsed=end-start
	times.append(elapsed)
	
	# print(bonds)
	# print(sites)
	# print([np.round(bmi(q,e.j),14) for e in bonds])
	# make_graph(sites,bonds,"tree%d"%img_count)

	snapshots.append([beta,bonds,sites])
	print("Time elapsed for this beta: %f s"%times[-1])
	beta+=args.step_beta
	# sys.exit(1)

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
