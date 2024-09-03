#ifndef MAT_OPS_
#define MAT_OPS_

#include "ndarray.hpp"

array3d<double> transpose(array3d<double>&);
array3d<double> matricize(array3d<double>&,size_t);
array3d<double> tensorize(array3d<double>&,size_t,size_t,size_t);
array3d<double> matmul_xTy(array3d<double>&,array3d<double>&);

//LAPACK function prototypes
extern "C" {
    extern void dgelsd_(int* m,int* n,int* nrhs,double* a,int* lda,double* b,int* ldb,double* s,double* rcond,int* rank,double* work,int* lwork,int* iwork,int* info);
    extern void dgesvd_(char* jobu,char* jobvt,int* m,int* n,double* a,int* lda,double* s,double* u,int* ldu,double* vt,int* ldvt,double* work,int* lwork,int* info);
}
array3d<double> hadamard(array3d<double>&,array3d<double>&);
void lstsq(array3d<double>&,array3d<double>&,array3d<double>&,size_t&);
void svd(array3d<double>&,array3d<double>&,array1d<double>&,array3d<double>&,size_t&);

array3d<double> nn_hals(array3d<double>&,array3d<double>&,array3d<double>&,size_t);
array3d<double> mu_ls(array3d<double>&,array3d<double>&,array3d<double>&,size_t);
array3d<double> mu_ls2(array3d<double>&,array3d<double>&,array3d<double>&,size_t);
array3d<double> mu_kl(array3d<double>&,array3d<double>&,array3d<double>&,size_t);

#endif
