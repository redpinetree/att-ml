import sys,os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spopt

def power_law(x,a,b):
	return a*(x**b)

def gen_erdos_renyi(n,p):
	assert n>1
	assert p>0 and p<=1
	v_count=0
	e_count=0
	es=[]
	randoms=np.random.rand(n*(n-1)//2)
	r_idx=0
	for i in range(n):
		for j in range(i+1,n):
			if randoms[r_idx]<p:
				coupling=(int) (((np.random.rand(1)*2)-1)*1e5)
				es.append([i,j,coupling])
				# print(es[-1])
			r_idx+=1
	# print(n,len(es))
	header=[n,len(es)]
	return [header,es]

def gen_barabasi_albert(n,m): #start from star graph with m+1 vertices
	assert n>0
	assert m<n
	v_count=m+1
	e_count=m
	es=[]
	for i in range(m):
		coupling=(int) (((np.random.rand(1)*2)-1)*1e5)
		es.append([0,i+1,coupling])
	randoms=np.random.rand(n-v_count)
	r_idx=0
	for i in range(v_count,n):
		v_count+=1
		sampled_vs=[]
		while len(sampled_vs)!=m:
			sampled_e=es[np.random.choice(len(es),1)[0]]
			sampled_v_idx=np.random.choice([0,1],1)
			sampled_v=sampled_e[0] if sampled_v_idx==0 else sampled_e[1]
			if i==sampled_v:
				continue
			if sampled_v not in sampled_vs:
				sampled_vs.append(sampled_v)
				coupling=(int) (((np.random.rand(1)*2)-1)*1e5)
				es.append([i,sampled_v,coupling])
				# print(es[-1])
		r_idx+=1
	# print(n,len(es))
	header=[n,len(es)]
	return [header,es]

parser=argparse.ArgumentParser()
parser.add_argument("n_graphs",type=int)
parser.add_argument("output_dir")
parser.add_argument("file_prefix")
parser.add_argument("param1")
parser.add_argument("param2")
parser.add_argument('-g','--graph_type',required=True,choices=["er","ba"])
args=parser.parse_args()

#create output dir
try:
	os.makedirs(args.output_dir)
except OSError as err:
	pass

for n in range(args.n_graphs):
	#generate graph
	if args.graph_type=="er":
		g=gen_erdos_renyi(int(args.param1),float(args.param2))
		degs=[]
		# for v in range(g[0][0]):
			# degs.append(0)
		# for e in g[1]:
			# degs[e[0]]+=1
			# degs[e[1]]+=1
	elif args.graph_type=="ba":
		g=gen_barabasi_albert(int(args.param1),int(args.param2))
		degs=[]
		# for v in range(g[0][0]):
			# degs.append(0)
		# for e in g[1]:
			# degs[e[0]]+=1
			# degs[e[1]]+=1
		# uniques,freqs=np.unique(degs,return_counts=True)
		# freqs=freqs/np.sum(freqs)
		# fig=plt.figure()
		# ax=plt.subplot(1,1,1)
		# ax.scatter(uniques,freqs)
		# popt,pcov=spopt.curve_fit(power_law,uniques[int(len(uniques)*0.1):int(len(uniques)*0.9)],freqs[int(len(freqs)*0.1):int(len(freqs)*0.9)],p0=[1,-3])
		# ax.plot(uniques,power_law(uniques,*popt))
		# ax.set_xscale("log")
		# ax.set_yscale("log")
		# print(popt)
		# ax.set_title(r"$a=%.3f,b=%.3f$"%tuple(popt))
		# ax.set_xlabel(r"degree")
		# ax.set_ylabel(r"freq")
		# fig.savefig("test_ba.png")
		# plt.close()
	#output graph
	fn=os.path.join(args.output_dir,args.file_prefix+"_%d.txt"%n)
	with open(fn,"w") as f:
		f.write("%d %d\n"%tuple(g[0]))
		for e in g[1]:
			f.write("%d %d %d\n"%tuple(e))