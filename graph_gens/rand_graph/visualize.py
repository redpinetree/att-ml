import sys,os
import argparse
import graphviz

parser=argparse.ArgumentParser()
parser.add_argument("fn")
args=parser.parse_args()

with open(args.fn,"r") as file:
	lines=file.readlines()
n_sites=int(lines[0].split(" ")[0])
es=[] #vertex1, vertex2, weight, todo
for n in range(len(lines)-1):
	es.append(list(int(i) for i in lines[n+1].split(" ")))

dot=graphviz.Graph()
for i in range(n_sites):
	dot.node("%d"%i)
for e in es:
	dot.edge("%d"%e[0],"%d"%e[1])
dot.render(os.path.basename(args.fn)+".gv")