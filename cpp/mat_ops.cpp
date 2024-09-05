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

//matricize by fusing two axes and leaving one axis untouched
array3d<double> matricize(array3d<double>& x,size_t sep_ax){ //mttkrp order
    if(x.nz()==1){ //already a matrix
        return x;
    }
    array3d<double> res;
    if(sep_ax==0){
        res=array3d<double>(x.ny()*x.nz(),x.nx(),1);
        for(size_t i=0;i<x.nx();i++){ //can this be accelerated?
            for(size_t j=0;j<x.ny();j++){
                for(size_t k=0;k<x.nz();k++){
                    res.at((x.nz()*j)+k,i,0)=x.at(i,j,k);
                }
            }
        }
    }
    else if(sep_ax==1){
        res=array3d<double>(x.nx()*x.nz(),x.ny(),1);
        for(size_t i=0;i<x.nx();i++){ //can this be accelerated?
            for(size_t j=0;j<x.ny();j++){
                for(size_t k=0;k<x.nz();k++){
                    res.at((x.nz()*i)+k,j,0)=x.at(i,j,k);
                }
            }
        }
    }
    else if(sep_ax==2){
        res=array3d<double>(x.nx()*x.ny(),x.nz(),1);
        for(size_t i=0;i<x.nx();i++){ //can this be accelerated?
            for(size_t j=0;j<x.ny();j++){
                for(size_t k=0;k<x.nz();k++){
                    res.at((x.ny()*i)+j,k,0)=x.at(i,j,k);
                    // res.at((x.nx()*j)+i,k,0)=x.at(i,j,k);
                }
            }
        }
    }
    else{
        std::cout<<"Separated axis must be 0, 1, or 2.\n";
        exit(1);
    }
    return res;
}

//tensorize by breaking up first two axes, and leaving last axis untouched
array3d<double> tensorize(array3d<double>& x,size_t r_1,size_t r_2,size_t sep_ax){ //mttkrp order
    if(x.nx()!=r_1*r_2){ //cannot reshape
        std::cerr<<"Sizes of first two axes to be broken up do not match the size of the matrix axis.\n";
        exit(1);
    }
    array3d<double> res(r_1,r_2,x.ny());
    if(sep_ax==0){
        for(size_t i=0;i<x.nx();i++){
            for(size_t j=0;j<x.ny();j++){
                res.at(j,i/r_2,i%r_2)=x.at(i,j,0);
            }
        }
    }
    else if(sep_ax==1){
        for(size_t i=0;i<x.nx();i++){
            for(size_t j=0;j<x.ny();j++){
                res.at(i/r_2,j,i%r_2)=x.at(i,j,0);
            }
        }
    }
    else if(sep_ax==2){
        for(size_t i=0;i<x.nx();i++){ //can this be accelerated?
            for(size_t j=0;j<x.ny();j++){
                res.at(i/r_2,i%r_2,j)=x.at(i,j,0);
                // res.at(i%r_1,i/r_1,j)=x.at(i,j,0);
            }
        }
    }
    else{
        std::cout<<"Separated axis must be 0, 1, or 2.\n";
        exit(1);
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
void lstsq(array3d<double>& a_mat,array3d<double>& b_mat,array3d<double>& x_mat,size_t& status){
    //input variables
    a_mat=transpose(a_mat); //row major to column major order
    b_mat=transpose(b_mat); //row major to column major order
    int m=a_mat.nx(); //rows of a
    int n=a_mat.ny(); //cols of a
    int nrhs=b_mat.ny(); //cols of b,x
    int lda=m; //leading dim of a=m
    int ldb=(m>n)?m:n; //leading dim of b=max(m,n)
    double rcond=-1.0; //negative means machine precision
    double a[lda*n]; //input a
    std::copy(a_mat.e().begin(),a_mat.e().end(),&a[0]);
    a_mat=transpose(a_mat);
    double b[ldb*nrhs]; //input b
    std::copy(b_mat.e().begin(),b_mat.e().end(),&b[0]);
    b_mat=transpose(b_mat);
    
    //intermediate quantities
    int smlsiz=25; //size of smallest problem, usually 25
    int nlvl=((int) log2(((m<n)?m:n)/(smlsiz+1)))+1; //number of levels in problem division
    nlvl=nlvl>0?nlvl:0;
    int iwork_size=(3*((m<n)?m:n)*nlvl)+(11*((m<n)?m:n)); //size of iwork array
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
        return;
    }
    free((void*) work); //free workspace memory
    
    //construct array3d object to hold solution
    x_mat=array3d<double>(nrhs,n,1);
    std::copy(&b[0],&b[n*nrhs],x_mat.e().begin());
    b_mat=transpose(b_mat); //col major to row major order
    
    // std::cout<<(std::string) res;
}

//calculate the svd
void svd(array3d<double>& a_mat,array3d<double>& u_mat,array1d<double>& s_vec,array3d<double>& vt_mat,size_t& status){
    //input variables
    char jobu='S'; //thin svd
    char jobvt='S'; //thin svd
    a_mat=transpose(a_mat); //row major to column major order
    int m=a_mat.nx(); //rows of a
    int n=a_mat.ny(); //cols of a
    int r=(m<n)?m:n; //aux: # of singular values
    int lda=m; //leading dim of a=m
    int ldu=m; //leading dim of u=m
    int ldvt=r; //leading dim of vt=min(m,n)
    double a[lda*n]; //input a
    std::copy(a_mat.e().begin(),a_mat.e().end(),&a[0]);
    a_mat=transpose(a_mat);
    
    //intermediate quantities
    int lwork=-1; //workspace dim, if lwork=-1, query optimal workspace size
    double optimal_work; //double because actual argument takes double pointer
    double* work; //workspace
    
    //output variables
    int info; //output status
    int rank; //output rank
    double s[r]; //output singular values
    double u[ldu*r]; //output u
    double vt[ldvt*n]; //output vt
    
    //obtain svd
    dgesvd_(&jobu,&jobvt,&m,&n,a,&lda,s,u,&ldu,vt,&ldvt,&optimal_work,&lwork,&info); //query optimal workspace size, found in optimal_work
    lwork=(int) optimal_work;
    work=(double*) malloc(lwork*sizeof(double)); //allocate optimal workspace memory
    dgesvd_(&jobu,&jobvt,&m,&n,a,&lda,s,u,&ldu,vt,&ldvt,work,&lwork,&info); //compute svd
    if(info>0){
        std::cout<<"DGESVD's SVD failed to converge.\n";
        status=1; //status 1 means failure
        return;
    }
    free((void*) work); //free workspace memory
    
    //fill objects to hold solution
    //due to order differences, we are actually computing the svd of aT, so resulting u,vT of aT are vT,u of a
    s_vec=array1d<double>(r);
    std::copy(&s[0],&s[r],s_vec.e().begin());
    vt_mat=array3d<double>(m,r,1);
    std::copy(&u[0],&u[m*r],vt_mat.e().begin());
    vt_mat=transpose(vt_mat); //col major to row major order
    u_mat=array3d<double>(r,n,1);
    std::copy(&vt[0],&vt[r*n],u_mat.e().begin());
    u_mat=transpose(u_mat); //col major to row major order
    
    // std::cout<<"s:\n"<<(std::string) s_vec<<"\n";
    // std::cout<<"u:\n"<<(std::string) u_mat<<"\n";
    // std::cout<<"vt:\n"<<(std::string) vt_mat<<"\n";
}

array3d<double> nn_hals(array3d<double>& aTa,array3d<double>& aTb,array3d<double>& x,size_t max_it){
    if((aTa.nx()!=aTb.nx())||(aTa.ny()!=x.nx())||(x.ny()!=aTb.ny())){
        std::cout<<"Incompatible matrix equation in NN-HALS with dimensions ("<<aTa.nx()<<","<<aTa.ny()<<") ("<<x.nx()<<","<<x.ny()<<")=("<<aTb.nx()<<","<<aTb.ny()<<")\n";
        exit(1);
    }
    array3d<double> prev_x=x;
    double eps=1e-16;
    double delta_first=0;
    for(size_t it=0;it<max_it;it++){
        for(size_t k=0;k<aTb.nx();k++){
            if(aTa.at(k,k,0)!=0){
                double zero_check=0;
                std::vector<double> c;
                for(size_t col=0;col<x.ny();col++){
                    double c_sum=0;
                    for(size_t l=0;l<x.nx();l++){
                        c_sum+=aTa.at(k,l,0)*x.at(l,col,0);
                    }
                    c.push_back(c_sum);
                }
                for(size_t col=0;col<x.ny();col++){
                    c[col]-=x.at(k,col,0)*aTa.at(k,k,0); //this term in c removes current k column information
                }
                for(size_t col=0;col<x.ny();col++){
                    x.at(k,col,0)=(aTb.at(k,col,0)-c[col])/aTa.at(k,k,0);
                    if(x.at(k,col,0)>1e3){
                        std::cout<<aTb.at(k,col,0)<<" "<<c[col]<<" "<<aTa.at(k,k,0)<<"\n";
                        std::cout<<(std::string)aTb<<"\n";
                        std::cout<<(std::string)aTa<<"\n";
                        std::cout<<(std::string)x<<"\n";
                        exit(1);
                    }
                    x.at(k,col,0)=(x.at(k,col,0)>eps)?x.at(k,col,0):eps; //clip negative values
                    zero_check+=x.at(k,col,0);
                }
                
                //columns should not be zero
                // if(zero_check<eps){
                    // double x_max_val=*std::max_element(x.e().begin(),x.e().end());
                    // for(size_t col=0;col<x.ny();col++){
                        // x.at(k,col,0)=eps*x_max_val;
                    // }
                // }
            }
            
            
        }
        double delta=0;
        for(size_t i=0;i<x.nx();i++){
            for(size_t j=0;j<x.ny();j++){
                delta+=(x.at(i,j,0)-prev_x.at(i,j,0))*(x.at(i,j,0)-prev_x.at(i,j,0));
            }
        }
        delta=sqrt(delta);
        prev_x=x;
        if(it==0){
            delta_first=delta;
            continue;
        }
        if(delta<=(eps*delta_first)){
            // std::cout<<"HALS stopped early after "<<it<<" iterations.\n";
            break;
        }
    }
    // if(norm_flag){
        // double sum=0;
        // for(size_t i=0;i<x.nx();i++){
            // for(size_t j=0;j<x.ny();j++){
                // sum+=x.at(i,j,0);
            // }
        // }
        // for(size_t i=0;i<x.nx();i++){
            // for(size_t j=0;j<x.ny();j++){
                // x.at(i,j,0)=x.at(i,j,0)/sum;
            // }
        // }
    // }
    return x;
}


array3d<double> nn_hals2(array3d<double>& aaT,array3d<double>& baT,array3d<double>& x,size_t max_it){
    if((x.nx()!=baT.nx())||(x.ny()!=aaT.nx())||(aaT.ny()!=baT.ny())){
        std::cout<<"Incompatible matrix equation in NN-HALS with dimensions ("<<x.nx()<<","<<x.ny()<<") ("<<aaT.nx()<<","<<aaT.ny()<<")=("<<baT.nx()<<","<<baT.ny()<<")\n";
        exit(1);
    }
    array3d<double> prev_x=x;
    double eps=1e-16;
    double delta_first=0;
    for(size_t it=0;it<max_it;it++){
        for(size_t k=0;k<baT.ny();k++){
            if(aaT.at(k,k,0)!=0){
                double zero_check=0;
                std::vector<double> c;
                for(size_t col=0;col<x.nx();col++){
                    double c_sum=0;
                    for(size_t l=0;l<x.ny();l++){
                        c_sum+=x.at(col,l,0)*aaT.at(l,k,0);
                    }
                    c.push_back(c_sum);
                }
                for(size_t col=0;col<x.nx();col++){
                    c[col]-=x.at(col,k,0)*aaT.at(k,k,0); //this term in c removes current k column information
                }
                for(size_t col=0;col<x.nx();col++){
                    x.at(col,k,0)=(baT.at(col,k,0)-c[col])/aaT.at(k,k,0);
                    if(x.at(col,k,0)>1e3){
                        std::cout<<baT.at(col,k,0)<<" "<<c[col]<<" "<<aaT.at(k,k,0)<<"\n";
                        std::cout<<(std::string)baT<<"\n";
                        std::cout<<(std::string)aaT<<"\n";
                        std::cout<<(std::string)x<<"\n";
                        exit(1);
                    }
                    x.at(col,k,0)=(x.at(col,k,0)>eps)?x.at(col,k,0):eps; //clip negative values
                    zero_check+=x.at(col,k,0);
                }
                
                //columns should not be zero
                // if(zero_check<eps){
                    // double x_max_val=*std::max_element(x.e().begin(),x.e().end());
                    // for(size_t col=0;col<x.nx();col++){
                        // x.at(col,k,0)=eps*x_max_val;
                    // }
                // }
            }
        }
        double delta=0;
        for(size_t i=0;i<x.nx();i++){
            for(size_t j=0;j<x.ny();j++){
                delta+=(x.at(i,j,0)-prev_x.at(i,j,0))*(x.at(i,j,0)-prev_x.at(i,j,0));
            }
        }
        delta=sqrt(delta);
        prev_x=x;
        if(it==0){
            delta_first=delta;
            continue;
        }
        if(delta<=(eps*delta_first)){
            // std::cout<<"HALS stopped early after "<<it<<" iterations.\n";
            break;
        }
    }
    // if(norm_flag){
        // double sum=0;
        // for(size_t i=0;i<x.nx();i++){
            // for(size_t j=0;j<x.ny();j++){
                // sum+=x.at(i,j,0);
            // }
        // }
        // for(size_t i=0;i<x.nx();i++){
            // for(size_t j=0;j<x.ny();j++){
                // x.at(i,j,0)=x.at(i,j,0)/sum;
            // }
        // }
    // }
    return x;
}

array3d<double> mu_ls(array3d<double>& aTa,array3d<double>& aTb,array3d<double>& x,size_t max_it){
    if((aTa.nx()!=aTb.nx())||(aTa.ny()!=x.nx())||(x.ny()!=aTb.ny())){
        std::cout<<"Incompatible matrix equation in MU-LS with dimensions ("<<aTa.nx()<<","<<aTa.ny()<<") ("<<x.nx()<<","<<x.ny()<<")=("<<aTb.nx()<<","<<aTb.ny()<<")\n";
        exit(1);
    }
    array3d<double> prev_x=x;
    double eps=1e-16;
    double delta_first=0;
    for(size_t it=0;it<max_it;it++){
        array3d<double> aTax=matmul_xTy(aTa,x);
        for(size_t i=0;i<x.nx();i++){
            for(size_t j=0;j<x.ny();j++){
                double num=aTb.at(i,j,0);
                double denom=aTax.at(i,j,0);
                x.at(i,j,0)*=num/denom;
                x.at(i,j,0)=(x.at(i,j,0)>eps)?x.at(i,j,0):eps;
            }
        }
        double delta=0;
        for(size_t i=0;i<x.nx();i++){
            for(size_t j=0;j<x.ny();j++){
                delta+=(x.at(i,j,0)-prev_x.at(i,j,0))*(x.at(i,j,0)-prev_x.at(i,j,0));
            }
        }
        delta=sqrt(delta);
        prev_x=x;
        if(it==0){
            delta_first=delta;
            continue;
        }
        if(delta<=(eps*delta_first)){
            // std::cout<<"MU-LS stopped early after "<<it<<" iterations.\n";
            break;
        }
    }
    // if(norm_flag){
        // double sum=0;
        // for(size_t i=0;i<x.nx();i++){
            // for(size_t j=0;j<x.ny();j++){
                // sum+=x.at(i,j,0);
            // }
        // }
        // for(size_t i=0;i<x.nx();i++){
            // for(size_t j=0;j<x.ny();j++){
                // x.at(i,j,0)=x.at(i,j,0)/sum;
            // }
        // }
    // }
    return x;
}

array3d<double> mu_ls2(array3d<double>& aaT,array3d<double>& baT,array3d<double>& x,size_t max_it){
    if((x.nx()!=baT.nx())||(x.ny()!=aaT.nx())||(aaT.ny()!=baT.ny())){
        std::cout<<"Incompatible matrix equation in MU-LS with dimensions ("<<x.nx()<<","<<x.ny()<<") ("<<aaT.nx()<<","<<aaT.ny()<<")=("<<baT.nx()<<","<<baT.ny()<<")\n";
        exit(1);
    }
    array3d<double> prev_x=x;
    double eps=1e-16;
    double delta_first=0;
    for(size_t it=0;it<max_it;it++){
        x=transpose(x);
        array3d<double> xaaT=matmul_xTy(x,aaT);
        x=transpose(x);
        for(size_t i=0;i<x.nx();i++){
            for(size_t j=0;j<x.ny();j++){
                double num=baT.at(i,j,0);
                double denom=xaaT.at(i,j,0);
                x.at(i,j,0)*=num/denom;
                x.at(i,j,0)=(x.at(i,j,0)>eps)?x.at(i,j,0):eps;
            }
        }
        double delta=0;
        for(size_t i=0;i<x.nx();i++){
            for(size_t j=0;j<x.ny();j++){
                delta+=(x.at(i,j,0)-prev_x.at(i,j,0))*(x.at(i,j,0)-prev_x.at(i,j,0));
            }
        }
        delta=sqrt(delta);
        prev_x=x;
        if(it==0){
            delta_first=delta;
            continue;
        }
        if(delta<=(eps*delta_first)){
            // std::cout<<"MU-LS stopped early after "<<it<<" iterations.\n";
            break;
        }
    }
    // if(norm_flag){
        // double sum=0;
        // for(size_t i=0;i<x.nx();i++){
            // for(size_t j=0;j<x.ny();j++){
                // sum+=x.at(i,j,0);
            // }
        // }
        // for(size_t i=0;i<x.nx();i++){
            // for(size_t j=0;j<x.ny();j++){
                // x.at(i,j,0)=x.at(i,j,0)/sum;
            // }
        // }
    // }
    return x;
}

array3d<double> mu_kl_ls(array3d<double>& aTa,array3d<double>& aTb,array3d<double>& x,size_t max_it){
    if((aTa.nx()!=aTb.nx())||(aTa.ny()!=x.nx())||(x.ny()!=aTb.ny())){
        std::cout<<"Incompatible matrix equation in MU-KL-LS with dimensions ("<<aTa.nx()<<","<<aTa.ny()<<") ("<<x.nx()<<","<<x.ny()<<")=("<<aTb.nx()<<","<<aTb.ny()<<")\n";
        exit(1);
    }
    array3d<double> prev_x=x;
    double eps=1e-16;
    double delta_first=0;
    std::vector<double> denom=aTa.sum_over_axis(1,2);
    for(size_t it=0;it<max_it;it++){
        array3d<double> aTax=matmul_xTy(aTa,x);
        for(size_t i=0;i<x.nx();i++){
            for(size_t j=0;j<x.ny();j++){
                double num=0;
                for(size_t k=0;k<aTa.nx();k++){
                    num+=aTa.at(k,i,0)*aTb.at(k,j,0)/aTax.at(k,j,0);
                }
                x.at(i,j,0)*=num/denom[i];
                x.at(i,j,0)=(x.at(i,j,0)>eps)?x.at(i,j,0):eps;
            }
        }
        double delta=0;
        for(size_t i=0;i<x.nx();i++){
            for(size_t j=0;j<x.ny();j++){
                delta+=(x.at(i,j,0)-prev_x.at(i,j,0))*(x.at(i,j,0)-prev_x.at(i,j,0));
            }
        }
        delta=sqrt(delta);
        prev_x=x;
        if(it==0){
            delta_first=delta;
            continue;
        }
        if(delta<=(eps*delta_first)){
            // std::cout<<"MU-KL stopped early after "<<it<<" iterations.\n";
            break;
        }
    }
    // if(norm_flag){
        // double sum=0;
        // for(size_t i=0;i<x.nx();i++){
            // for(size_t j=0;j<x.ny();j++){
                // sum+=x.at(i,j,0);
            // }
        // }
        // for(size_t i=0;i<x.nx();i++){
            // for(size_t j=0;j<x.ny();j++){
                // x.at(i,j,0)=x.at(i,j,0)/sum;
            // }
        // }
    // }
    return x;
}

array3d<double> mu_kl(array3d<double>& m,array3d<double>& w,array3d<double>& h,size_t max_it){
    if((w.nx()!=m.nx())||(w.ny()!=h.nx())||(h.ny()!=m.ny())){
        std::cout<<"Incompatible matrix equation in MU-KL with dimensions ("<<w.nx()<<","<<w.ny()<<") ("<<h.nx()<<","<<h.ny()<<")=("<<m.nx()<<","<<m.ny()<<")\n";
        exit(1);
    }
    array3d<double> prev_h=h;
    double eps=1e-16;
    double delta_first=0;
    std::vector<double> denom=w.sum_over_axis(0,2);
    for(size_t it=0;it<max_it;it++){
        w=transpose(w);
        array3d<double> wh=matmul_xTy(w,h);
        w=transpose(w);
        for(size_t i=0;i<h.nx();i++){
            for(size_t j=0;j<h.ny();j++){
                double num=0;
                for(size_t k=0;k<m.nx();k++){
                    num+=w.at(k,i,0)*m.at(k,j,0)/wh.at(k,j,0);
                }
                h.at(i,j,0)*=num/denom[i];
                h.at(i,j,0)=(h.at(i,j,0)>eps)?h.at(i,j,0):eps;
            }
        }
        double delta=0;
        for(size_t i=0;i<h.nx();i++){
            for(size_t j=0;j<h.ny();j++){
                delta+=(h.at(i,j,0)-prev_h.at(i,j,0))*(h.at(i,j,0)-prev_h.at(i,j,0));
            }
        }
        delta=sqrt(delta);
        prev_h=h;
        if(it==0){
            delta_first=delta;
            continue;
        }
        if(delta<=(eps*delta_first)){
            // std::cout<<"MU-KL stopped early after "<<it<<" iterations.\n";
            break;
        }
    }
    return h;
}

array3d<double> mu_kl2(array3d<double>& m,array3d<double>& w,array3d<double>& h,size_t max_it){
    if((w.nx()!=m.nx())||(w.ny()!=h.nx())||(h.ny()!=m.ny())){
        std::cout<<"Incompatible matrix equation in MU-KL with dimensions ("<<w.nx()<<","<<w.ny()<<") ("<<h.nx()<<","<<h.ny()<<")=("<<m.nx()<<","<<m.ny()<<")\n";
        exit(1);
    }
    array3d<double> prev_w=w;
    double eps=1e-16;
    double delta_first=0;
    std::vector<double> denom=h.sum_over_axis(1,2);
    for(size_t it=0;it<max_it;it++){
        w=transpose(w);
        array3d<double> wh=matmul_xTy(w,h);
        w=transpose(w);
        for(size_t i=0;i<w.nx();i++){
            for(size_t j=0;j<w.ny();j++){
                double num=0;
                for(size_t k=0;k<m.ny();k++){
                    num+=h.at(j,k,0)*m.at(i,k,0)/wh.at(i,k,0);
                }
                w.at(i,j,0)*=num/denom[j];
                w.at(i,j,0)=(w.at(i,j,0)>eps)?w.at(i,j,0):eps;
            }
        }
        double delta=0;
        for(size_t i=0;i<w.nx();i++){
            for(size_t j=0;j<w.ny();j++){
                delta+=(w.at(i,j,0)-prev_w.at(i,j,0))*(w.at(i,j,0)-prev_w.at(i,j,0));
            }
        }
        delta=sqrt(delta);
        prev_w=w;
        if(it==0){
            delta_first=delta;
            continue;
        }
        if(delta<=(eps*delta_first)){
            // std::cout<<"MU-KL stopped early after "<<it<<" iterations.\n";
            break;
        }
    }
    return w;
}