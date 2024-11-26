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
    // if(((sep_ax==0)&&(x.nx()==1))&&((sep_ax==1)&&(x.ny()==1))&&((sep_ax==2)&&(x.nz()==1))){ //already a matrix
        // return x;
    // }
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
    array3d<double> res;
    if(sep_ax==0){
        res=array3d<double>(x.ny(),r_1,r_2);
        for(size_t i=0;i<x.nx();i++){
            for(size_t j=0;j<x.ny();j++){
                res.at(j,i/r_2,i%r_2)=x.at(i,j,0);
            }
        }
    }
    else if(sep_ax==1){
        res=array3d<double>(r_1,x.ny(),r_2);
        for(size_t i=0;i<x.nx();i++){
            for(size_t j=0;j<x.ny();j++){
                res.at(i/r_2,j,i%r_2)=x.at(i,j,0);
            }
        }
    }
    else if(sep_ax==2){
        res=array3d<double>(r_1,r_2,x.ny());
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
    x_mat=transpose(x_mat); //row major to column major order
    int m=a_mat.nx(); //rows of a
    int n=a_mat.ny(); //cols of a
    int nrhs=b_mat.nx(); //cols of b,x
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
    x_mat=transpose(x_mat); //col major to row major order
    
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

//calculate the qr factorization
void qr(array3d<double>& a_mat,array3d<double>& q_mat,array3d<double>& r_mat,size_t& status){
    //input variables
    // a_mat=transpose(a_mat); //row major to column major order
    int m=a_mat.nx(); //rows of a
    int n=a_mat.ny(); //cols of a
    int r=(m<n)?m:n; //aux: # of householder reflectors in tau
    int lda=m; //leading dim of a=m
    double a[lda*n]; //input a
    double tau[r]; //output tau
    std::copy(a_mat.e().begin(),a_mat.e().end(),&a[0]);
    // a_mat=transpose(a_mat); //row major to column major order
    
    //intermediate quantities
    int lwork=-1; //workspace dim, if lwork=-1, query optimal workspace size
    double optimal_work; //double because actual argument takes double pointer
    double* work; //workspace
    int lwork2=-1; //workspace dim, if lwork=-1, query optimal workspace size
    double optimal_work2; //double because actual argument takes double pointer
    double* work2; //workspace
    
    //output variables
    int info; //output status
    
    //obtain qr factorization
    dgeqrf_(&m,&n,a,&lda,tau,&optimal_work,&lwork,&info); //query optimal workspace size, found in optimal_work
    lwork=(int) optimal_work;
    work=(double*) malloc(lwork*sizeof(double)); //allocate optimal workspace memory
    dgeqrf_(&m,&n,a,&lda,tau,work,&lwork,&info); //compute qr
    if(info>0){
        std::cout<<"DGEQRF's QR decomposition failed to converge.\n";
        status=1; //status 1 means failure
        return;
    }
    free((void*) work); //free workspace memory
    
    //extract compressed qr representation h
    array1d<double> tau_vec=array1d<double>(r);
    std::copy(&tau[0],&tau[r],tau_vec.e().begin());
    array3d<double> h_mat=array3d<double>(m,n,1);
    std::copy(&a[0],&a[m*n],h_mat.e().begin());
    //prepare r matrix from h
    r_mat=array3d<double>(r,n,1);
    for(size_t i=0;i<r_mat.nx();i++){
        for(size_t j=i;j<r_mat.ny();j++){
            r_mat.at(i,j,0)=h_mat.at(i,j,0); //implicit transpose
        }
    }
    
    //obtain q
    dorgqr_(&m,&r,&r,a,&lda,tau,&optimal_work2,&lwork2,&info); //query optimal workspace size, found in optimal_work
    lwork2=(int) optimal_work2;
    work2=(double*) malloc(lwork2*sizeof(double)); //allocate optimal workspace memory
    q_mat=array3d<double>(m,r,1);
    dorgqr_(&m,&r,&r,a,&lda,tau,work2,&lwork2,&info); //query optimal workspace size, found in optimal_work
    std::copy(&a[0],&a[m*r],q_mat.e().begin());
    // q_mat=transpose(q_mat); //col major to row major order
    if(info>0){
        std::cout<<"DORGQR's Q calculation failed to converge.\n";
        status=1; //status 1 means failure
        return;
    }
    free((void*) work2); //free workspace memory
    
    // std::cout<<"tau:\n"<<(std::string) tau_vec<<"\n";
    // std::cout<<"h:\n"<<(std::string) h_mat<<"\n";
    // std::cout<<"q:\n"<<(std::string) q_mat<<"\n";
    // std::cout<<"r:\n"<<(std::string) r_mat<<"\n";
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
                    // if(x.at(k,col,0)>1e3){
                        // std::cout<<aTb.at(k,col,0)<<" "<<c[col]<<" "<<aTa.at(k,k,0)<<"\n";
                        // std::cout<<(std::string)aTb<<"\n";
                        // std::cout<<(std::string)aTa<<"\n";
                        // std::cout<<(std::string)x<<"\n";
                        // exit(1);
                    // }
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
                    // if(x.at(col,k,0)>1e3){
                        // std::cout<<baT.at(col,k,0)<<" "<<c[col]<<" "<<aaT.at(k,k,0)<<"\n";
                        // std::cout<<(std::string)baT<<"\n";
                        // std::cout<<(std::string)aaT<<"\n";
                        // std::cout<<(std::string)x<<"\n";
                        // exit(1);
                    // }
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

void nndsvda(array3d<double>& target,array3d<double>& w,array3d<double>& h,size_t r,size_t status){
    //compute svd
    array3d<double> u;
    array1d<double> s;
    array3d<double> vt;
    svd(target,u,s,vt,status);
    if(status!=0){
        std::cout<<"failed on:\n"<<(std::string) target<<"\n";
    }
    
    w=array3d<double>(u.nx(),r,u.nz());
    h=array3d<double>(r,vt.ny(),vt.nz());
    
    //nonnegative leading singular triplet can be used right away
    for(size_t i=0;i<w.nx();i++){
        w.at(i,0,0)=sqrt(s.at(0))*fabs(u.at(i,0,0));
    }
    for(size_t j=0;j<h.ny();j++){
        h.at(0,j,0)=sqrt(s.at(0))*fabs(vt.at(0,j,0));
    }
    
    for(size_t k=1;k<r;k++){ //only consider largest r singular triplets
        array1d<double> x_p(u.nx());
        array1d<double> x_n(u.nx());
        double x_p_norm=0;
        double x_n_norm=0;
        for(size_t i=0;i<u.nx();i++){
            if(u.at(i,k,0)>0){
                x_p.at(i)=u.at(i,k,0);
                x_p_norm+=u.at(i,k,0)*u.at(i,k,0);
           }
            else{
                x_n.at(i)=fabs(u.at(i,k,0));
                x_n_norm+=u.at(i,k,0)*u.at(i,k,0);
            }
        }
        array1d<double> y_p(vt.ny());
        array1d<double> y_n(vt.ny());
        double y_p_norm=0;
        double y_n_norm=0;
        for(size_t j=0;j<vt.ny();j++){
            if(vt.at(k,j,0)>0){
                y_p.at(j)=vt.at(k,j,0);
                y_p_norm+=vt.at(k,j,0)*vt.at(k,j,0);
            }
            else{
                y_n.at(j)=fabs(vt.at(k,j,0));
                y_n_norm+=vt.at(k,j,0)*vt.at(k,j,0);
            }
        }
        
        array1d<double> x(u.nx());
        array1d<double> y(vt.ny());
        double mult;
        if((x_p_norm*y_p_norm)>(x_n_norm*y_n_norm)){
            mult=x_p_norm*y_p_norm;
            for(size_t i=0;i<u.nx();i++){
                x.at(i)=x_p.at(i)/x_p_norm;
            }
            for(size_t j=0;j<vt.ny();j++){
                y.at(j)=y_p.at(j)/y_p_norm;
            }
        }
        else{
            mult=x_n_norm*y_n_norm;
            for(size_t i=0;i<u.nx();i++){
                x.at(i)=x_n.at(i)/x_n_norm;
            }
            for(size_t j=0;j<vt.ny();j++){
                y.at(j)=y_n.at(j)/y_n_norm;
            }
        }
        
        for(size_t i=0;i<w.nx();i++){
            w.at(i,k,0)=sqrt(s.at(k)*mult)*fabs(x.at(i));
        }
        for(size_t j=0;j<h.ny();j++){
            h.at(k,j,0)=sqrt(s.at(k)*mult)*fabs(y.at(j));
        }
    }
    
    double mean=target.sum_over_all()/(double) (target.nx()*target.ny()*target.nz());
    for(size_t i=0;i<w.nx();i++){
        for(size_t j=0;j<w.ny();j++){
            if(w.at(i,j,0)<1e-6){
                w.at(i,j,0)=mean;
            }
        }
    }
    for(size_t i=0;i<h.nx();i++){
        for(size_t j=0;j<h.ny();j++){
            if(h.at(i,j,0)<1e-6){
                h.at(i,j,0)=mean;
            }
        }
    }
    
    // std::cout<<"mat: "<<target.nx()<<" "<<target.ny()<<" "<<target.nz()<<"\n";
    // std::cout<<"mat:\n"<<(std::string) target;
    // std::cout<<"u: "<<u.nx()<<" "<<u.ny()<<" "<<u.nz()<<"\n";
    // std::cout<<"u:\n"<<(std::string) u;
    // std::cout<<"s:\n"<<(std::string) s;
    // std::cout<<"vt: "<<vt.nx()<<" "<<vt.ny()<<" "<<vt.nz()<<"\n";
    // std::cout<<"vt:\n"<<(std::string) vt;
    // std::cout<<"w: "<<w.nx()<<" "<<w.ny()<<" "<<w.nz()<<"\n";
    // std::cout<<"w:\n"<<(std::string) w;
    // std::cout<<"h: "<<h.nx()<<" "<<h.ny()<<" "<<h.nz()<<"\n";
    // std::cout<<"h:\n"<<(std::string) h;
    // exit(1);
}

double nmf(array3d<double>& target,array3d<double>& w,array3d<double>& h,size_t r){
    size_t status=0;
    nndsvda(target,w,h,r,status);
    // std::cout<<"w:\n"<<(std::string) w;
    // std::cout<<"h:\n"<<(std::string) h;
    // std::cout<<"w: "<<w.nx()<<" "<<w.ny()<<" "<<w.nz()<<"\n";
    // std::cout<<"h: "<<h.nx()<<" "<<h.ny()<<" "<<h.nz()<<"\n";
    
    //use NN-HALS to alternately optimize w and h against target
    // target=target.exp_form();
    // w=w.exp_form();
    // h=h.exp_form();
    // std::cout<<(std::string) target<<"\n";
    // std::cout<<(std::string) w<<"\n";
    // std::cout<<(std::string) h<<"\n";
    array3d<double> old_w=w;
    array3d<double> old_h=h;
    double old_recon_err=1e50;
    double recon_err;
    for(size_t opt_iter=0;opt_iter<10000;opt_iter++){
        // array3d<double> aTa=matmul_xTy(w,w);
        // array3d<double> aTb=matmul_xTy(w,target);
        // h=nn_hals(aTa,aTb,h,1);
        // h=mu_ls(aTa,aTb,h,1);
        h=mu_kl(target,w,h,1);
        
        // h=transpose(h);
        // target=transpose(target);
        // array3d<double> aaT=matmul_xTy(h,h);
        // array3d<double> baT=matmul_xTy(target,h);
        // h=transpose(h);
        // target=transpose(target);
        // w=transpose(w);
        // w=nn_hals2(aaT,baT,w,1);
        // w=transpose(w);
        // w=mu_ls2(aaT,baT,w,1);
        w=mu_kl2(target,w,h,1);
        
        array3d<double> recon_mat(target.nx(),target.ny(),target.nz());
        for(size_t i=0;i<recon_mat.nx();i++){
            for(size_t j=0;j<recon_mat.ny();j++){
                for(size_t k=0;k<w.ny();k++){
                    recon_mat.at(i,j,0)+=w.at(i,k,0)*h.at(k,j,0);
                }
            }
        }
        double tr_mTm=0;
        double tr_whTwh=0;
        double tr_mTwh=0;
        for(size_t i=0;i<recon_mat.nx();i++){
            for(size_t j=0;j<recon_mat.ny();j++){
                tr_mTm+=target.at(i,j,0)*target.at(i,j,0);
                tr_whTwh+=recon_mat.at(i,j,0)*recon_mat.at(i,j,0);
                tr_mTwh+=target.at(i,j,0)*recon_mat.at(i,j,0);
            }
        }
        recon_err=(tr_mTm+tr_whTwh-(2*tr_mTwh))/tr_mTm;
        if(recon_err<1e-12){
            // std::cout<<"Optimization converged after "<<(opt_iter+1)<<" iterations (recon err).\n";
            // if(opt_iter==0){
                // std::cout<<tr_mTm<<" "<<tr_whTwh<<" "<<tr_mTwh<<"\n";
                // std::cout<<(std::string) w<<"\n";
                // std::cout<<(std::string) h<<"\n";
                // std::cout<<(std::string) target<<"\n";
                // std::cout<<(std::string) recon_mat<<"\n";
                // exit(1);
            // }
            // std::cout<<"Final reconstruction err: "<<recon_err<<".\n";
            break;
        }
        // else if(fabs(recon_err-old_recon_err)<1e-6){
            // std::cout<<"Optimization converged after "<<(opt_iter+1)<<" iterations (recon err diff).\n";
            // std::cout<<"Final reconstruction err: "<<recon_err<<".\n";
            // break;
        // }
        old_recon_err=recon_err;
        // exit(1);
        
        double diff=0;
        for(size_t i=0;i<w.nx();i++){
            for(size_t j=0;j<w.ny();j++){
                diff+=(old_w.at(i,j,0)-w.at(i,j,0))*(old_w.at(i,j,0)-w.at(i,j,0));
            }
        }
        for(size_t i=0;i<h.nx();i++){
            for(size_t j=0;j<h.ny();j++){
                diff+=(old_h.at(i,j,0)-h.at(i,j,0))*(old_h.at(i,j,0)-h.at(i,j,0));
            }
        }
        if(diff<1e-12){
            // std::cout<<"Optimization converged after "<<(opt_iter+1)<<" iterations (diff).\n";
            // std::cout<<"Final diff: "<<diff<<".\n";
            // std::cout<<"Final reconstruction err: "<<recon_err<<".\n";
            // break;
        }
        old_w=w;
        old_h=h;
    }
    // std::cout<<(*std::max_element(w.e().begin(),w.e().end()))<<" "<<(*std::max_element(h.e().begin(),h.e().end()))<<"\n";
    double sum2=h.sum_over_all();
    for(size_t i=0;i<h.nx();i++){
        for(size_t j=0;j<h.ny();j++){
            h.at(i,j,0)/=sum2;
        }
    }
    double sum1=w.sum_over_all();
    for(size_t i=0;i<w.nx();i++){
        for(size_t j=0;j<w.ny();j++){
            w.at(i,j,0)/=sum1;
        }
    }
    return recon_err;
}

double truncated_svd(array3d<double>& target,array3d<double>& truncated_us,array3d<double>& truncated_vt,size_t r){
    size_t status=0;
    array3d<double> u;
    array1d<double> s;
    array3d<double> vt;
    svd(target,u,s,vt,status);
    
    truncated_us=array3d<double>(u.nx(),r,1);
    for(size_t i=0;i<truncated_us.nx();i++){
        for(size_t j=0;j<truncated_us.ny();j++){
            truncated_us.at(i,j,0)=u.at(i,j,0)*s.at(j);
        }
    }
    truncated_vt=array3d<double>(r,vt.ny(),1);
    for(size_t i=0;i<truncated_vt.nx();i++){
        for(size_t j=0;j<truncated_vt.ny();j++){
            truncated_vt.at(i,j,0)=vt.at(i,j,0);
        }
    }
    
    array3d<double> recon_mat(truncated_us.nx(),truncated_vt.ny(),1);
    for(size_t i=0;i<recon_mat.nx();i++){
        for(size_t j=0;j<recon_mat.ny();j++){
            for(size_t k=0;k<r;k++){
                recon_mat.at(i,j,0)+=truncated_us.at(i,k,0)*truncated_vt.at(k,j,0);
            }
        }
    }
    double tr_mTm=0;
    double tr_whTwh=0;
    double tr_mTwh=0;
    for(size_t i=0;i<recon_mat.nx();i++){
        for(size_t j=0;j<recon_mat.ny();j++){
            tr_mTm+=target.at(i,j,0)*target.at(i,j,0);
            tr_whTwh+=recon_mat.at(i,j,0)*recon_mat.at(i,j,0);
            tr_mTwh+=target.at(i,j,0)*recon_mat.at(i,j,0);
        }
    }
    double recon_err=(tr_mTm+tr_whTwh-(2*tr_mTwh))/tr_mTm;
    return recon_err;
}