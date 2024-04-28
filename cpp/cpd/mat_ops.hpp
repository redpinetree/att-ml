#ifndef MAT_OPS
#define MAT_OPS

#include "../ndarray.hpp"

//LAPACK DGELSD function prototype
extern "C" {
    extern void dgelsd_(int* m,int* n,int* nrhs,double* a,int* lda,double* b,int* ldb,double* s,double* rcond,int* rank,double* work,int* lwork,int* iwork,int* info);
}

array3d<double> transpose(array3d<double>&);
array3d<double> matricize(array3d<double>&);
array3d<double> tensorize(array3d<double>&,size_t,size_t);
array3d<double> matmul_xTy(array3d<double>&,array3d<double>&);
array3d<double> hadamard(array3d<double>&,array3d<double>&);
array3d<double> lstsq(array3d<double>&,array3d<double>&,size_t&);

#endif
