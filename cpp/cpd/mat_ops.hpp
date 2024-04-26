#ifndef MAT_OPS
#define MAT_OPS

#include "../ndarray.hpp"

array3d<double> transpose(array3d<double>&);
array3d<double> matricize(array3d<double>&);
array3d<double> tensorize(array3d<double>&,size_t,size_t);
array3d<double> matmul_xTy(array3d<double>&,array3d<double>&);
array3d<double> hadamard(array3d<double>&,array3d<double>&);

#endif
