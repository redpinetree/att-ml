import numpy as np
import graph

num_vs,es=graph.gen_hypercubic(2,(3,4),'gaussian',pbc=False)
num_vs,es=graph.gen_hypercubic(2,(3,4),'gaussian',pbc=True,mean=1.5)
num_vs,es=graph.gen_hypercubic(2,(3,4),'gaussian',pbc=True,mean=-2,std=0.01)
num_vs,es=graph.gen_hypercubic(2,(3,4),'bimodal',pbc=True,mean=1.5)
num_vs,es=graph.gen_hypercubic(2,(3,4),'bimodal',pbc=False,p=0.2)

num_vs,es=graph.gen_hypercubic(2,(4,4),'gaussian',pbc=True)
graph.save_graph("test_save.txt",num_vs,es)