/*
Copyright 2025 Katsuya O. Akamatsu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef MAT_OPS_
#define MAT_OPS_

#include "ndarray.hpp"

array3d<double> transpose(array3d<double>&);
array3d<double> matricize(array3d<double>&,int);
array3d<double> tensorize(array3d<double>&,int,int,int);
array3d<double> matmul_xTy(array3d<double>&,array3d<double>&);

//LAPACK function prototypes
extern "C" {
    extern void dgelsd_(int* m,int* n,int* nrhs,double* a,int* lda,double* b,int* ldb,double* s,double* rcond,int* rank,double* work,int* lwork,int* iwork,int* info);
    extern void dgesvd_(char* jobu,char* jobvt,int* m,int* n,double* a,int* lda,double* s,double* u,int* ldu,double* vt,int* ldvt,double* work,int* lwork,int* info);
    extern void dgeqrf_(int* m,int* n,double* a,int* lda,double* tau,double* work,int* lwork,int* info);
    extern void dorgqr_(int* m,int* n,int* k,double* a,int* lda,double* tau,double* work,int* lwork,int* info);
}
array3d<double> hadamard(array3d<double>&,array3d<double>&);
void lstsq(array3d<double>&,array3d<double>&,array3d<double>&,int&);
void svd(array3d<double>&,array3d<double>&,array1d<double>&,array3d<double>&,int&);
void qr(array3d<double>&,array3d<double>&,array3d<double>&,int&);

array3d<double> nn_hals(array3d<double>&,array3d<double>&,array3d<double>&,int);
array3d<double> nn_hals2(array3d<double>&,array3d<double>&,array3d<double>&,int);
array3d<double> mu_ls(array3d<double>&,array3d<double>&,array3d<double>&,int);
array3d<double> mu_ls2(array3d<double>&,array3d<double>&,array3d<double>&,int);
array3d<double> mu_kl(array3d<double>&,array3d<double>&,array3d<double>&,int);
array3d<double> mu_kl2(array3d<double>&,array3d<double>&,array3d<double>&,int);
array3d<double> ccd_kl(array3d<double>&,array3d<double>&,array3d<double>&,int);
array3d<double> ccd_kl2(array3d<double>&,array3d<double>&,array3d<double>&,int);
array3d<double> sn_kl(array3d<double>&,array3d<double>&,array3d<double>&,int);
array3d<double> sn_kl2(array3d<double>&,array3d<double>&,array3d<double>&,int);

void nndsvda(array3d<double>&,array3d<double>&,array3d<double>&,int,int);
double nmf(array3d<double>&,array3d<double>&,array3d<double>&,int);
double truncated_svd(array3d<double>&,array3d<double>&,array3d<double>&,int);

#endif
