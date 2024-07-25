import sys,os
import argparse
import numpy as np

def gen_randgauge2d(nx,ny,n_gauge_flips=0):
	assert nx>0 and ny>0
	n_gauge_flips=10*nx*ny if n_gauge_flips==0 else n_gauge_flips
	v_count=nx*ny
	es=[]
	rand_idxs=np.random.choice(v_count,n_gauge_flips)
	#determine which sites are flipped by counting freqs and checking parity
	flipped=np.bincount(rand_idxs)%2
	flipped=np.pad(flipped,(0,v_count-len(flipped)),'constant')
	# print(flipped)
	for j in range(ny):
		for i in range(nx):
			idx=(j*nx)+i
			next_idx_x=(j*nx)+i+1 if ((i+1)%nx)!=0 else (j*nx)
			next_idx_y=((j+1)*nx)+i if ((j+1)%ny)!=0 else i
			flipped_factor_x=-1 if flipped[idx]^flipped[next_idx_x] else 1
			flipped_factor_y=-1 if flipped[idx]^flipped[next_idx_y] else 1
			es.append([idx,next_idx_x,1e5*flipped_factor_x])
			es.append([idx,next_idx_y,1e5*flipped_factor_y])
	# print(v_count,len(es))
	# print(es)
	header=[v_count,len(es)]
	return [header,es]

parser=argparse.ArgumentParser()
parser.add_argument("nx",type=int)
parser.add_argument("ny",type=int)
parser.add_argument("output_fn")
args=parser.parse_args()

#generate graph
g=gen_randgauge2d(args.nx,args.ny)

#create output dir
path,fn=os.path.split(args.output_fn)
try:
	os.makedirs(path)
except OSError as err:
	pass
#output graph
fn=os.path.join(args.output_fn)
with open(fn,"w") as f:
	f.write("%d %d\n"%tuple(g[0]))
	for e in g[1]:
		f.write("%d %d %d\n"%tuple(e))