import sys,os
import argparse
import numpy as np

def gen_hier2d(nx,ny,p):
	assert nx>0 and ny>0
	assert p>0 and p<=1
	v_count=nx*ny
	es=[]
	for j in range(ny):
		for i in range(nx):
			#bithack: -x is two's complement, so -x is (~x)+1.
			#(~x)+1 induces carries up to x's LSB (since that will be 0 in ~x)
			#so the length of x&(-x), w/o 0s on left side, gives pos of LSB.
			if(i!=(nx-1)):
				i_lsb=(i&(-i)).bit_length()
				coupling_x=(int) ((p**i_lsb)*1e5)
				es.append([(j*nx)+i,(j*nx)+i+1,coupling_x])
			if(j!=(ny-1)):
				j_lsb=(j&(-j)).bit_length()
				coupling_y=(int) ((p**j_lsb)*1e5)
				es.append([(j*nx)+i,((j+1)*nx)+i,coupling_y])
	# print(v_count,len(es))
	# print(es)
	header=[v_count,len(es)]
	return [header,es]

parser=argparse.ArgumentParser()
parser.add_argument("nx",type=int)
parser.add_argument("ny",type=int)
parser.add_argument("p",type=float)
parser.add_argument("output_fn")
args=parser.parse_args()

#generate graph
g=gen_hier2d(args.nx,args.ny,args.p)

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