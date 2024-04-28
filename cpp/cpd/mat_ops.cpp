#include <algorithm>
#include <cmath>
#include <iostream>

#include "mat_ops.hpp"

array3d<double> transpose(array3d<double>& x){
    if(x.nz()!=1){
        std::cerr<<"Input is not a matrix.\n";
        exit(1);
    }
    array3d<double> res(x.ny(),x.nx(),1);
    for(size_t i=0;i<res.nx();i++){
        for(size_t j=0;j<res.ny();j++){
            res.at(i,j,0)=x.at(j,i,0);
        }
    }
    return res;
}

//matricize by fusing first two axes and leaving last axis untouched
array3d<double> matricize(array3d<double>& x){
    if(x.nz()==1){ //already a matrix
        return x;
    }
    array3d<double> res(x.nx()*x.ny(),x.nz(),1);
    for(size_t i=0;i<x.nx();i++){ //can this be accelerated?
        for(size_t j=0;j<x.ny();j++){
            for(size_t k=0;k<x.nz();k++){
                res.at((x.ny()*i)+j,k,0)=x.at(i,j,k);
            }
        }
    }
    return res;
}

//tensorize by breaking up first two axes, and leaving last axis untouched
array3d<double> tensorize(array3d<double>& x,size_t r_i,size_t r_j){
    if(x.nx()!=r_i*r_j){ //cannot reshape
        std::cerr<<"Sizes of first two axes to be broken up do not match the size of the matrix axis.\n";
        exit(1);
    }
    array3d<double> res(r_i,r_j,x.ny());
    for(size_t i=0;i<x.nx();i++){ //can this be accelerated?
        for(size_t j=0;j<x.ny();j++){
            res.at(i/r_j,i%r_j,j)=x.at(i,j,0);
        }
    }
    return res;
}

//matrix multiplication with first argument transposed
array3d<double> matmul_xTy(array3d<double>& x,array3d<double>& y){
    if((x.nz()!=1)||(y.nz()!=1)){
        std::cerr<<"One input is not a matrix.\n";
        exit(1);
    }
    if(x.nx()!=y.nx()){
        std::cerr<<"Matrix dimensions ("<<x.ny()<<","<<x.nx()<<") (transposed) and ("<<y.nx()<<","<<y.ny()<<") are not compatible in attempted matrix multiplication.\n";
        exit(1);
    }
    array3d<double> res(x.ny(),y.ny(),1);
    for(size_t i=0;i<res.nx();i++){ //can this be accelerated?
        for(size_t j=0;j<res.ny();j++){
            for(size_t k=0;k<x.nx();k++){
                res.at(i,j,0)+=x.at(k,i,0)*y.at(k,j,0);
            }
        }
    }
    return res;
}

//hadamard product or penetrating face product along last axis, where y must be a matrix
array3d<double> hadamard(array3d<double>& x,array3d<double>& y){
    if((x.nx()!=y.nx())||(x.ny()!=y.ny())||(y.nz()!=1)){
        std::cerr<<"Matrix dimensions ("<<x.nx()<<","<<x.ny()<<","<<x.nz()<<") (transposed) and ("<<y.nx()<<","<<y.ny()<<","<<y.nz()<<") are not compatible in attempted Hadamard/penetrating face product.\n";
        exit(1);
    }
    array3d<double> res(x.nx(),x.ny(),x.nz());
    for(size_t i=0;i<res.nx();i++){ //can this be accelerated?
        for(size_t j=0;j<res.ny();j++){
            for(size_t k=0;k<res.nz();k++){
                res.at(i,j,k)=x.at(i,j,k)*y.at(i,j,0);
            }
        }
    }
    return res;
}

//solve the least squares problem ax=b
array3d<double> lstsq(array3d<double>& a_mat,array3d<double>& b_mat,size_t& status){
    //input variables
    int m=a_mat.nx(); //rows of a
    int n=a_mat.ny(); //cols of a
    int nrhs=b_mat.ny(); //cols of b,x
    int lda=m; //leading dim of a=m
    int ldb=(m>n)?m:n; //leading dim of b=max(m,n)
    double rcond=-1.0; //negative means machine precision
    double a[lda*n]; //input a
    std::copy(a_mat.e().begin(),a_mat.e().end(),&a[0]);
    double b[ldb*nrhs]; //input b
    std::copy(b_mat.e().begin(),b_mat.e().end(),&b[0]);
    
    //intermediate quantities
    int smlsiz=25; //size of smallest problem, usually 25
    int nlvl=((int) log2(((m<n)?m:n)/(smlsiz+1)))+1; //number of levels in problem division
    nlvl=nlvl>0?nlvl:0;
    int iwork_size=(3*((m<n)?m:n)*nlvl)+(11*((m<n)?m:n)); //size of iwork aray
    int iwork[iwork_size];
    int lwork=-1; //workspace dim, if lwork=-1, query optimal workspace size
    double optimal_work; //double because actual argument takes double pointer
    double* work; //workspace
    
    //output variables
    int info; //output status
    int rank; //output rank
    double s[m]; //output singular values
    
    //obtain least squares solution
    dgelsd_(&m,&n,&nrhs,a,&lda,b,&ldb,s,&rcond,&rank,&optimal_work,&lwork,iwork,&info); //query optimal workspace size, found in optimal_work
    lwork=(int) optimal_work;
    work=(double*) malloc(lwork*sizeof(double)); //allocate optimal workspace memory
    dgelsd_(&m,&n,&nrhs,a,&lda,b,&ldb,s,&rcond,&rank,work,&lwork,iwork,&info); //solve ax=b
    if(info>0){
        std::cout<<"DGELSD's SVD failed to converge.\n";
        status=1; //status 1 means failure
        return array3d<double>(n,nrhs,1);
    }
    free((void*) work); //free workspace memory
    
    //construct array3d object to hold solution
    array3d<double> res(n,nrhs,1);
    std::copy(&b[0],&b[n*nrhs],res.e().begin());
    
    // std::cout<<(std::string) res;
    
    return res;
}