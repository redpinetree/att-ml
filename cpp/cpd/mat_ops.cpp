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