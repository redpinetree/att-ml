import numpy as np
import sys,copy,os,time
import argparse

import graph

#dfs with vertices and adj edges
# def dfs(adj,bonds,start,end,current_path=[],visited=[]):
	# visited[start]=True
	# # print(current_path)
	# for i in range(len(adj[start])):
		# if len(current_path)==0 or (len(current_path)>0 and adj[start][i]!=current_path[-1]):
			# if (bonds[adj[start][i]].v1==start and not visited[bonds[adj[start][i]].v2]) or (bonds[adj[start][i]].v2==start and not visited[bonds[adj[start][i]].v1]):
				# # print("adding ",adj[start][i])
				# current_path.append(adj[start][i])
				# # print(current_path)
				# if bonds[adj[start][i]].v1==end or bonds[adj[start][i]].v2==end: #it's impossible for edges[i][0], edges[i][1] to be the same
					# return current_path
				# else:
					# res=dfs(adj,bonds,bonds[adj[start][i]].v1 if bonds[adj[start][i]].v2==start else bonds[adj[start][i]].v2,end,current_path,visited)
					# # print(current_path,res)
					# if res is not None:
						# # print("returning res ",res)
						# return res
					# # print("removing ",current_path[-1])
					# current_path.pop()
def dfs(adj,bonds,orig_start,start,current_path,visited,paths):
	# print(current_path)
	visited[start]=True
	for i in range(len(adj[start])):
		if len(current_path)==0 or (len(current_path)>0 and adj[start][i]!=current_path[-1]):
			if (bonds[adj[start][i]].v1==start and not visited[bonds[adj[start][i]].v2]) or (bonds[adj[start][i]].v2==start and not visited[bonds[adj[start][i]].v1]):
				# print("adding ",adj[start][i])
				current_path.append(adj[start][i])
				next_start=bonds[adj[start][i]].v1 if bonds[adj[start][i]].v2==start else bonds[adj[start][i]].v2
				# print(current_path)
				if next_start>orig_start:
					paths.append([[orig_start,next_start],copy.deepcopy(current_path)])
				# print(paths)
				res=dfs(adj,bonds,orig_start,next_start,current_path,visited,paths)
				# if res is not None:
					# print("returning res ",res)
					# return res
				# print("removing ",current_path[-1])
				current_path.pop()
	return paths


parser=argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
args=parser.parse_args()

file=args.input
data=np.load(file,allow_pickle=True)
stats=[]
times=[]
for file in data.files:
	start=time.time()
	beta=data[file][0]
	bonds=data[file][1]
	sites=data[file][2]
	n_sites=len(sites)

	#build edge adjacency
	adj=[[] for i in range(n_sites)]
	for i in range(len(bonds)):
		adj[bonds[i].v1].append(i)
		adj[bonds[i].v2].append(i)
	# print(sites)
	# print(bonds)
	# print(adj)

	#compute correlation matrix
	corr=np.eye(n_sites)
	# for i in range(n_sites):
		# for j in range(i+1,n_sites):
			# path=dfs(adj,bonds,i,j,current_path=[],visited=[False for i in range(n_sites)]) #path is list of bond idxs from site i to site j
			# corr[i,j]=np.prod([np.tanh(bonds[k].j) for k in path])
			# corr[j,i]=corr[i,j]
	all_paths=[]
	for i in range(n_sites):
		paths=dfs(adj,bonds,i,i,[],[False for i in range(n_sites)],[])
		all_paths+=paths
	# print(len(all_paths))
	for path in all_paths:
		corr[path[0][0],path[0][1]]=np.prod([np.tanh(bonds[k].j) for k in path[1]])
		corr[path[0][1],path[0][0]]=corr[path[0][0],path[0][1]]
	# paths=[[[] for j in range(n_sites)] for i in range(n_sites)]
	# corr=np.eye(n_sites)
	# for i in range(n_sites):
		# for j in range(i+1,n_sites):
			# paths[i][j]=dfs(adj,bonds,i,j,current_path=[],visited=[False for i in range(n_sites)]) #path is list of bond idxs from site i to site j
			# paths[j][i]=paths[i][j]
			# print(i,j,path)
			# for k in path:
				# print(edges[k])
			# corr[i,j]=np.prod([np.tanh(bonds[k].j) for k in paths[i][j]])
			# corr[j,i]=corr[i,j]
	# print(corr)
	# print(paths)
	# sys.exit(1)
	# for i in range(corr.shape[0]):
		# for j in range(corr.shape[1]):
			# # corr[i,j]=(-1)**(i+j)
			# # corr[i,j]=1
			# if i!=j:
				# corr[i,j]=0
	# print(corr)
	
	# q2_2=n_sites #diagonal of corr is all 1
	# for i in range(n_sites):
		# for j in range(i+1,n_sites):
			# q2_2+=2*(corr[i,j]) #2 corr(a,b)<->corr(b,a) pairs
	# print("<si sj>:",q2_2,q2_2/n_sites**2)
	
	# s=time.time()
	# q4_2=n_sites #contribution of i=j=k=l, <iiii>=1
	# q4_2+=6*(n_sites)*(n_sites-1)/2 #contribution of i=k,j=l, <kkll>=<kk><ll>=1, aabb abab abba and b<->a versions
	# for k in range(n_sites):
		# for l in range(k+1,n_sites):
			# # contribution of i=j=k, <kkkl>=<kk><kl>=<kl>
			# q4_2+=8*corr[k,l] #aaab aaba abaa baaa and b<->a versions
			# # contribution of i=j, <jjkl>=<jj><kl>=<kl>
			# q4_2+=12*(n_sites-2)*corr[k,l] #3! permutes of j,k,l and b<->a versions, n_sites-2 factor comes from summing over j, except j==k and j==l
			# pass
	# for i in range(n_sites):
		# for j in range(i+1,n_sites):
			# for k in range(j+1,n_sites):
				# for l in range(k+1,n_sites):
					# # the three possible pairings share the same set of edges, but some pairings have repeated edges, so take the max of the three pairings
					# paths=[corr[i,j]*corr[k,l],corr[i,k]*corr[j,l],corr[i,l]*corr[j,k]]
					# max_idx=np.argmax(np.abs(paths))
					# q4_2+=24*paths[max_idx] #4! permutes
					# pass
	# print("<si sj sk sl>:",q4_2,q4_2/n_sites**4)
	# e=time.time()
	# print("m2 and m4 sum time: ",e-s,"s")
	
	q2=n_sites #diagonal of corr is all 1
	for i in range(n_sites):
		for j in range(i+1,n_sites):
			q2+=2*(corr[i,j]**2) #2 corr(a,b)<->corr(b,a) pairs
	# print("<si sj>^2:",q2,q2/n_sites**2)
	q2/=n_sites**2
	
	q4=n_sites #contribution of i=j=k=l, <iiii>=1
	q4+=6*(n_sites)*(n_sites-1)/2 #contribution of i=k,j=l, <kkll>=<kk><ll>=1, aabb abab abba and b<->a versions
	for k in range(n_sites):
		for l in range(k+1,n_sites):
			#contribution of i=j=k, <kkkl>=<kk><kl>=<kl>
			q4+=8*(corr[k,l]**2) #aaab aaba abaa baaa and b<->a versions
			#contribution of i=j, <jjkl>=<jj><kl>=<kl>
			q4+=12*(n_sites-2)*(corr[k,l]**2) #3! permutes of j,k,l and b<->a versions, n_sites-2 factor comes from summing over j, except j==k and j==l
	s=time.time()
	for i in range(n_sites):
		for j in range(i+1,n_sites):
			for k in range(j+1,n_sites):
				for l in range(k+1,n_sites):
					#the three possible pairings share the same set of edges, but some pairings have repeated edges, so take the max of the three pairings
					paths=[corr[i,j]*corr[k,l],corr[i,k]*corr[j,l],corr[i,l]*corr[j,k]]
					max_idx=np.argmax(np.abs(paths))
					q4+=24*(paths[max_idx]**2) #4! permutes
	e=time.time()
	# print("4-fold sum time: ",e-s,"s")
	# print("<si sj sk sl>^2:",q4,q4/n_sites**4)
	q4/=n_sites**4
	
	q2sqrt=np.sqrt(q2)
	q2_var=q4-(q2**2)
	q2_std=np.sqrt(q2_var)
	sus_sg=n_sites*q2
	binder_q=0.5*(3-(q4/(q2**2)))
	print(beta,q2,q4)
	print(q2,q2sqrt,sus_sg,q2_std,binder_q)
	print("")
	# stats.append((beta,q2,q2sqrt,sus_sg))
	stats.append((beta,q2,q2sqrt,sus_sg,q2_std,binder_q))
	# print(beta,q2,q2sqrt,sus_sg)
	end=time.time()
	elapsed=end-start
	times.append(elapsed)
	print("Time elapsed for this beta: %f s"%times[-1])
	sys.exit(0)

print("Total time elapsed for this run: %f s"%np.sum(times))
print("Average time elapsed for this run per beta: %f+/-%f s"%(np.mean(times),np.std(times)))
path=os.path.split(args.output)[0]
if path!="" and not os.path.exists(path):
	os.mkdir(os.path.split(args.output)[0])
out=open(args.output,"w")
# out.write("beta q2 q sus_sg\n")
out.write("beta q2 q sus_sg q2_std binder_q\n")
for i in range(len(stats)):
	# out.write("%f %f %f %f\n"%stats[i])
	out.write("%f %f %f %f %f %f\n"%stats[i])
out.close()
