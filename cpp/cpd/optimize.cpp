#include <fstream>

#include "mat_ops.hpp"
#include "../mpi_utils.hpp"
#include "optimize.hpp"
#include "../observables.hpp"

array3d<double> optimize::calc_aTa(std::vector<array3d<double> >& aTa_legs,std::vector<double>& weights,size_t m){
    size_t r_k=aTa_legs[0].nx();
    array3d<double> aTa(r_k,r_k,1);
    for(size_t x=0;x<aTa.nx();x++){
        for(size_t y=0;y<aTa.ny();y++){
            double factors=1;
            for(size_t m2=0;m2<aTa_legs.size();m2++){
                if(m==m2){continue;}
                factors*=aTa_legs[m2].at(x,y,0);
            }
            aTa.at(x,y,0)=factors*weights[x]*weights[y];
        }
    }
    return aTa;
}

array3d<double> optimize::calc_aTb(std::vector<array3d<double> >& aTb_legs,bond& old_current,std::vector<bond>& old_cluster,std::vector<double>& weights,size_t m){
    size_t r_i=old_current.w().nx();
    size_t r_j=old_current.w().ny();
    size_t r_k=aTb_legs[0].nz();
    size_t dim;
    if(m==0){
        dim=r_i*r_j;
    }
    else{
        //dim is the rank at the site that is not part of the active bond
        bool dim_flag=((old_cluster[m-1].v1()==old_current.v1())||(old_cluster[m-1].v1()==old_current.v2()));
        dim=dim_flag?old_cluster[m-1].w().ny():old_cluster[m-1].w().nx();
    }
    array3d<double> aTb(r_k,dim,1);
    if(m!=0){
        //this flag checks if the bond is connected to site i or site j of the active bond
        bool m_ij_flag=((old_cluster[m-1].v1()==old_current.v1())||(old_cluster[m-1].v2()==old_current.v1()));
        for(size_t x=0;x<aTb.nx();x++){
            for(size_t y=0;y<aTb.ny();y++){
                double sum=0;
                for(size_t i=0;i<r_i;i++){
                    for(size_t j=0;j<r_j;j++){
                        double factors=1;
                        for(size_t m2=0;m2<aTb_legs.size();m2++){
                            if(m==m2){continue;}
                            if(m2==0){ //w_ijk
                                factors*=aTb_legs[m2].at(i,j,x);
                                continue;
                            }
                            else{
                                //this flag checks if the bond is connected to site i or site j of the active bond
                                bool m2_ij_flag=((old_cluster[m2-1].v1()==old_current.v1())||(old_cluster[m2-1].v2()==old_current.v1()));
                                factors*=aTb_legs[m2].at(x,m2_ij_flag?i:j,0);
                            }
                        }
                        if(m_ij_flag){ //connected to i
                            if((old_cluster[m-1].v1()==old_current.v1())){ //imu<i
                                sum+=factors*old_cluster[m-1].w().at(i,y,0);
                            }
                            else{ //imu>i
                                sum+=factors*old_cluster[m-1].w().at(y,i,0);
                            }
                        }
                        else{ //connected to j
                            if((old_cluster[m-1].v1()==old_current.v2())){ //imu<j
                                sum+=factors*old_cluster[m-1].w().at(j,y,0);
                            }
                            else{ //imu>j
                                sum+=factors*old_cluster[m-1].w().at(y,j,0);
                            }
                        }
                    }
                }
                aTb.at(x,y,0)=sum*weights[x];
            }
        }
    }
    else{ //w_ijk
        for(size_t x=0;x<aTb.nx();x++){
            for(size_t y=0;y<aTb.ny();y++){
                double factors=1;
                for(size_t m2=0;m2<aTb_legs.size();m2++){
                    if(m==m2){continue;}
                    else{
                        //this flag checks if the bond is connected to site i or site j of the active bond
                        bool m2_ij_flag=((old_cluster[m2-1].v1()==old_current.v1())||(old_cluster[m2-1].v2()==old_current.v1()));
                        factors*=aTb_legs[m2].at(x,m2_ij_flag?y/r_j:y%r_j,0);
                    }
                }
                aTb.at(x,y,0)=factors*old_current.w().at(y/r_j,y%r_j,0)*weights[x];
            }
        }
    }
    return aTb;
}

double optimize::calc_tr_axTax(std::vector<array3d<double> >& aTa_legs,std::vector<double>& weights){
    double tr_axTax=0;
    for(size_t x=0;x<aTa_legs[0].nx();x++){
        for(size_t y=0;y<aTa_legs[0].ny();y++){
            double factors=1;
            for(size_t m=0;m<aTa_legs.size();m++){
                factors*=aTa_legs[m].at(x,y,0);
            }
            tr_axTax+=factors*weights[x]*weights[y];
        }
    }
    return tr_axTax;
}

double optimize::calc_tr_axTb(std::vector<array3d<double> >& aTb_legs,bond& old_current,std::vector<bond>& old_cluster,std::vector<double>& weights){
    double tr_axTb=0;
    for(size_t x=0;x<aTb_legs[0].nx();x++){
        for(size_t y=0;y<aTb_legs[0].ny();y++){
            for(size_t z=0;z<aTb_legs[0].nz();z++){
                double factors=1;
                for(size_t m=0;m<aTb_legs.size();m++){
                    if(m==0){ //w_ijk
                        factors*=aTb_legs[m].at(x,y,z);
                        continue;
                    }
                    else{
                        bool m_ij_flag=((old_cluster[m-1].v1()==old_current.v1())||(old_cluster[m-1].v2()==old_current.v1()));
                        factors*=aTb_legs[m].at(z,m_ij_flag?x:y,0);
                    }
                }
                tr_axTb+=factors*weights[z];
            }
        }
    }
    return tr_axTb;
}

double optimize::calc_tr_bTb(std::vector<array3d<double> >& bTb_legs,bond& old_current,std::vector<bond>& old_cluster){
    double tr_bTb=0;
    for(size_t x=0;x<old_current.w().nx();x++){
        for(size_t y=0;y<old_current.w().ny();y++){
            double factors=1;
            for(size_t m=0;m<bTb_legs.size();m++){
                    bool m_ij_flag=((old_cluster[m].v1()==old_current.v1())||(old_cluster[m].v2()==old_current.v1()));
                    factors*=m_ij_flag?bTb_legs[m].at(x,x,0):bTb_legs[m].at(y,y,0);
            }
            tr_bTb+=factors*old_current.w().at(x,y,0)*old_current.w().at(x,y,0);
        }
    }
    return tr_bTb;
}

array3d<double> optimize::nn_hals(array3d<double>& aTa,array3d<double>& aTb,array3d<double>& x){
    if((aTa.nx()!=aTb.nx())||(aTa.ny()!=x.nx())||(x.ny()!=aTb.ny())){
        std::cout<<"Incompatible matrix equation in NN-HALS with dimensions ("<<aTa.nx()<<","<<aTa.ny()<<") ("<<x.nx()<<","<<x.ny()<<")=("<<aTb.nx()<<","<<aTb.ny()<<")\n";
        exit(1);
    }
    array3d<double> prev_x=x;
    double err=0;
    double err_first=0;
    for(size_t it=0;it<1000;it++){
        for(size_t k=0;k<aTb.nx();k++){
            if(aTa.at(k,k,0)!=0){
                double zero_check=0;
                std::vector<double> c;
                for(size_t col=0;col<x.ny();col++){
                    double c_sum=0;
                    for(size_t l=0;l<x.nx();l++){
                        c_sum+=aTa.at(l,k,0)*x.at(l,col,0);
                    }
                    c.push_back(c_sum);
                }
                for(size_t col=0;col<x.ny();col++){
                    c[col]-=x.at(k,col,0)*aTa.at(k,k,0); //this term in c removes current k column information
                }
                for(size_t col=0;col<x.ny();col++){
                    x.at(k,col,0)=(aTb.at(k,col,0)-c[col])/aTa.at(k,k,0);
                    x.at(k,col,0)=(x.at(k,col,0)>0)?x.at(k,col,0):0; //clip negative values
                    zero_check+=x.at(k,col,0);
                }
                
                //columns should not be zero
                if(zero_check<1e-32){
                    double x_max_val=*std::max_element(x.e().begin(),x.e().end());
                    for(size_t col=0;col<x.ny();col++){
                        x.at(k,col,0)=1e-16*x_max_val;
                    }
                }
            }
        }
        double err=0;
        for(size_t i=0;i<x.nx();i++){
            for(size_t j=0;j<x.ny();j++){
                err+=(x.at(i,j,0)-prev_x.at(i,j,0))*(x.at(i,j,0)-prev_x.at(i,j,0));
            }
        }
        err=sqrt(err);
        prev_x=x;
        if(it==0){
            err_first=err;
        }
        if(err<(1e-8*err_first)){ //eps=1e-8
            // std::cout<<"HALS stopped early after "<<it<<" iterations.\n";
            break;
        }
    }
    return transpose(x);
}

void optimize::normalize(bond& current,std::vector<bond>& cluster,std::vector<double>& weights){
    size_t r_k=current.w().nz();
    //undo normalization
    optimize::unnormalize(current,weights);
    //calculate norms along outward axis
    for(size_t m=0;m<cluster.size()+1;m++){
        std::vector<double> norms(r_k,0);
        if(m==0){
            for(size_t i=0;i<current.w().nx();i++){
                for(size_t j=0;j<current.w().ny();j++){
                    for(size_t k=0;k<current.w().nz();k++){
                        norms[k]+=current.w().at(i,j,k)*current.w().at(i,j,k);
                    }
                }
            }
        }
        else{
            for(size_t i=0;i<cluster[m-1].w().nx();i++){
                for(size_t j=0;j<cluster[m-1].w().ny();j++){
                    norms[j]+=cluster[m-1].w().at(i,j,0)*cluster[m-1].w().at(i,j,0);
                }
            }
        }
        //update weights
        // std::cout<<"norms:\n";
        // for(size_t k=0;k<norms.size();k++){
            // std::cout<<norms[k]<<" ";
        // }
        // std::cout<<"\n";
        // std::cout<<"weights:\n";
        // for(size_t k=0;k<weights.size();k++){
            // std::cout<<weights[k]<<" ";
        // }
        // std::cout<<"\n";
        for(size_t k=0;k<weights.size();k++){
            weights[k]*=sqrt(norms[k]);
            norms[k]=(norms[k]!=0)?sqrt(norms[k]):1; //avoid div by 0 later
        }
        //update factor matrices
        if(m==0){
            for(size_t i=0;i<current.w().nx();i++){
                for(size_t j=0;j<current.w().ny();j++){
                    for(size_t k=0;k<current.w().nz();k++){
                        current.w().at(i,j,k)/=norms[k];
                    }
                }
            }
        }
        else{
            for(size_t i=0;i<cluster[m-1].w().nx();i++){
                for(size_t j=0;j<cluster[m-1].w().ny();j++){
                    cluster[m-1].w().at(i,j,0)/=norms[j];
                }
            }
        }
    }
}

void optimize::unnormalize(bond& current,std::vector<double>& weights){
    size_t r_k=current.w().nz();
    //apply the weights to the current active bond and reset weights
    for(size_t i=0;i<current.w().nx();i++){
        for(size_t j=0;j<current.w().ny();j++){
            for(size_t k=0;k<current.w().nz();k++){
                current.w().at(i,j,k)*=weights[k];
            }
        }
    }
    weights=std::vector<double>(r_k,1);
}


double optimize::tree_cpd(bond& old_current,std::vector<bond>& old_cluster,bond& current,std::vector<bond>& cluster,size_t max_it,std::string init_method,bool unnormalize_flag){
    std::uniform_real_distribution<> unif_dist(1e-10,1.0);
    size_t r_i=current.w().nx();
    size_t r_j=current.w().ny();
    size_t r_k=current.w().nz();
    std::vector<double> weights(r_k,1);
    
    //precompute intermediate quantities
    std::vector<array3d<double> > aTa_legs;
    std::vector<array3d<double> > aTb_legs;
    std::vector<array3d<double> > bTb_legs;
    // std::cout<<"next set:\n";
    // std::cout<<"target:\n";
    // std::cout<<old_current<<"\n";
    // std::cout<<(std::string) old_current.w()<<"\n";
    // for(size_t n=0;n<old_cluster.size();n++){
        // std::cout<<old_cluster[n]<<"\n";
        // std::cout<<(std::string) old_cluster[n].w()<<"\n";
    // }
    // std::cout<<"trial:\n";
    // std::cout<<current<<"\n";
    // std::cout<<(std::string) current.w()<<"\n";
    // for(size_t n=0;n<cluster.size();n++){
        // std::cout<<cluster[n]<<"\n";
        // std::cout<<(std::string) cluster[n].w()<<"\n";
    // }
    // std::cout<<"\n";
    
    array3d<double> mat_current=matricize(current.w());
    aTa_legs.push_back(matmul_xTy(mat_current,mat_current));
    aTb_legs.push_back(hadamard(current.w(),old_current.w()));
    for(size_t n=0;n<cluster.size();n++){
        //axis 1 of old_cluster[n] corresponds to site i/j, axis 1 of cluster[n] corresponds to site k
        array3d<double> arg=((old_cluster[n].v2()==old_current.v1())||(old_cluster[n].v2()==old_current.v2()))?old_cluster[n].w():transpose(old_cluster[n].w());
        aTa_legs.push_back(matmul_xTy(cluster[n].w(),cluster[n].w())); //axis 1 of cluster[n] corresponds to site k
        aTb_legs.push_back(matmul_xTy(cluster[n].w(),arg));
        bTb_legs.push_back(matmul_xTy(arg,arg));
    }
    
    double prev_err=0;
    double err=1;
    double tr_bTb=optimize::calc_tr_bTb(bTb_legs,old_current,old_cluster);
    for(size_t it=0;it<max_it;it++){
        // std::cout<<"iteration "<<it<<"\n";
        for(size_t m=0;m<cluster.size()+1;m++){
            //calculate aTa and aTb
            array3d<double> aTa=optimize::calc_aTa(aTa_legs,weights,m);
            array3d<double> aTb=optimize::calc_aTb(aTb_legs,old_current,old_cluster,weights,m);
            
            //solve aTax=aTb via NN-HALS
            array3d<double> init(aTb.nx(),aTb.ny(),1);
            if(init_method=="prev"){ //initialize using previous iterate
                init=(m==0)?matricize(current.w()):cluster[m-1].w();
                init=transpose(init);
            }
            else if(init_method=="rand"){
                double sum=0;
                for(size_t i=0;i<init.nx();i++){
                    for(size_t j=0;j<init.ny();j++){
                        init.at(i,j,0)=unif_dist(mpi_utils::prng);
                        sum+=init.at(i,j,0);
                    }
                }
                for(size_t i=0;i<init.nx();i++){
                    for(size_t j=0;j<init.ny();j++){
                        init.at(i,j,0)/=sum;
                    }
                }
            }
            else if(init_method=="lstsq"){
                size_t status=0;
                init=lstsq(aTa,aTb,status);
                if(status==1){ //failed to converge, so fall back to "prev" method
                    std::cout<<"Falling back to \"prev\" method...\n";
                    init=(m==0)?matricize(current.w()):cluster[m-1].w();
                    init=transpose(init);
                }
            }
            else{
                std::cout<<"Invalid initializer for HALS subroutine ("<<init_method<<" was passed), aborting...\n";
                exit(1);
            }
            array3d<double> x=optimize::nn_hals(aTa,aTb,init);
            // std::cout<<(std::string)x<<"\n";
            if(m==0){
                current.w()=tensorize(x,r_i,r_j);
            }
            else{
                cluster[m-1].w()=x;
            }
            
            //normalize factors based on weights for numerical stability
            optimize::normalize(current,cluster,weights);
            
            //update cached quantities
            if(m==0){ //w_ijk
                array3d<double> mat_current=matricize(current.w());
                aTa_legs[0]=matmul_xTy(mat_current,mat_current);
                aTb_legs[0]=hadamard(current.w(),old_current.w());
                continue;
            }
            array3d<double> arg=((old_cluster[m-1].v2()==old_current.v1())||(old_cluster[m-1].v2()==old_current.v2()))?old_cluster[m-1].w():transpose(old_cluster[m-1].w());
            aTa_legs[m]=matmul_xTy(cluster[m-1].w(),cluster[m-1].w());
            aTb_legs[m]=matmul_xTy(cluster[m-1].w(),arg);
        }
        //calculate reconstruction error |ax-b|^2_F, normalized by tr_bTb
        double tr_axTax=optimize::calc_tr_axTax(aTa_legs,weights);
        double tr_axTb=optimize::calc_tr_axTb(aTb_legs,old_current,old_cluster,weights);
        err=(tr_axTax+tr_bTb-(2*tr_axTb))/tr_bTb;
        if((it>0)&&(fabs(prev_err-err)<1e-8)){
            // std::cout<<"CPD converged after "<<it<<" iterations.\n";
            break;
        }
        prev_err=err;
    }
    
    //convert superdiagonal core tensor to just plain delta tensor
    if(unnormalize_flag){
        optimize::unnormalize(current,weights);
    }
    
    return err;
}

double optimize::opt(size_t master,size_t slave,size_t r_k,std::vector<site> sites,bond& old_current,std::vector<bond>& old_cluster,bond& current,std::vector<bond>& cluster,size_t max_it,std::string init_method,size_t max_restarts){
    std::uniform_real_distribution<> unif_dist(1e-10,1.0);
    double best_err=1;
    if((init_method!="prev")&&(init_method!="lstsq")&&(init_method!="rand")&&(init_method!="hybrid")){
        std::cout<<"Invalid initializer for HALS subroutine ("<<init_method<<" was passed), aborting...\n";
        exit(1);
    }
    
    //TEMP: convert old_current, old_cluster to exp_form since not in log space
    old_current.w()=old_current.w().exp_form();
    for(size_t n=0;n<old_cluster.size();n++){
        old_cluster[n].w()=old_cluster[n].w().exp_form();
    }
    
    for(size_t restarts=1;restarts<max_restarts+1;restarts++){
        //initialize new current and cluster sizes
        bond trial_current=current;
        std::vector<bond> trial_cluster;
        for(size_t n=0;n<cluster.size();n++){
            trial_cluster.push_back(cluster[n]);
        }
        array3d<double> new_w(trial_current.w().nx(),trial_current.w().ny(),r_k);
        double sum=0;
        for(size_t i=0;i<new_w.nx();i++){
            for(size_t j=0;j<new_w.ny();j++){
                for(size_t k=0;k<new_w.nz();k++){
                    // new_w.at(i,j,k)=unif_dist(mpi_utils::prng);
                    new_w.at(i,j,k)=((i<old_current.w().nx())&&(j<old_current.w().ny()))?(old_current.w().at(i,j,0)*old_current.w().nx()*old_current.w().ny())/(double)(current.w().nx()*current.w().ny()*current.w().nz()):1/(double)(current.w().nx()*current.w().ny()*current.w().nz());
                    sum+=new_w.at(i,j,k);
                }
            }
        }
        for(size_t i=0;i<new_w.nx();i++){
            for(size_t j=0;j<new_w.ny();j++){
                for(size_t k=0;k<new_w.nz();k++){
                    new_w.at(i,j,k)/=sum;
                    // new_w.at(i,j,k)=new_w.at(i,j,k)==0?-100:log(new_w.at(i,j,k)); //TEMP: not in log space
                }
            }
        }
        trial_current.w()=new_w;
        if(restarts==1){ //initial resize
            current.w()=new_w;
        }
        for(size_t n=0;n<trial_cluster.size();n++){
            array3d<double> new_w(sites[trial_cluster[n].v1()].rank(),r_k,1);
            double sum=0;
            for(size_t i=0;i<new_w.nx();i++){
                for(size_t j=0;j<new_w.ny();j++){
                    // new_w.at(i,j,0)=unif_dist(mpi_utils::prng);
                    new_w.at(i,j,0)=((i<old_cluster[n].w().nx())&&(j<old_cluster[n].w().ny()))?(old_cluster[n].w().at(i,j,0)*old_cluster[n].w().nx()*old_cluster[n].w().ny())/(double)(cluster[n].w().nx()*cluster[n].w().ny()):1/(double)(cluster[n].w().nx()*cluster[n].w().ny());
                    sum+=new_w.at(i,j,0);
                }
            }
            for(size_t i=0;i<new_w.nx();i++){
                for(size_t j=0;j<new_w.ny();j++){
                    new_w.at(i,j,0)/=sum;
                    // new_w.at(i,j,0)=new_w.at(i,j,0)==0?-100:log(new_w.at(i,j,0)); //TEMP: not in log space
                }
            }
            trial_cluster[n].w()=new_w;
            if(restarts==1){ //initial resize
                cluster[n].w()=new_w;
            }
        }
        
        //perform tree-cpd
        double err;
        if(init_method=="hybrid"){
            std::string hybrid_init_method;
            if(restarts==1){ //first attempt uses padded target data
                hybrid_init_method="prev";
            }
            else if(restarts==2){ //second attempt uses least-squares approximation
                hybrid_init_method="lstsq";
            }
            else{ //remaining attempts use random initialization
                hybrid_init_method="rand";
            }
            err=optimize::tree_cpd(old_current,old_cluster,trial_current,trial_cluster,max_it,hybrid_init_method,1);
        }
        else{
            err=optimize::tree_cpd(old_current,old_cluster,trial_current,trial_cluster,max_it,init_method,1);
        }
        // std::cout<<"error: "<<err<<"\n";
        
        //check if better and update best result
        if(fabs(err)<best_err){
            best_err=err;
            current=trial_current;
            for(size_t n=0;n<cluster.size();n++){
                cluster[n]=trial_cluster[n];
            }
        }
    }
    //normalize each factor matrix by its sum so that it can be treated as probabilities
    double sum=0;
    for(size_t i=0;i<current.w().nx();i++){
        for(size_t j=0;j<current.w().ny();j++){
            for(size_t k=0;k<current.w().nz();k++){
                sum+=current.w().at(i,j,k);
            }
        }
    }
    for(size_t i=0;i<current.w().nx();i++){
        for(size_t j=0;j<current.w().ny();j++){
            for(size_t k=0;k<current.w().nz();k++){
                current.w().at(i,j,k)/=sum;
            }
        }
    }
    for(size_t n=0;n<cluster.size();n++){
        double sum=0;
        for(size_t i=0;i<cluster[n].w().nx();i++){
            for(size_t j=0;j<cluster[n].w().ny();j++){
                sum+=cluster[n].w().at(i,j,0);
            }
        }
        for(size_t i=0;i<cluster[n].w().nx();i++){
            for(size_t j=0;j<cluster[n].w().ny();j++){
                cluster[n].w().at(i,j,0)/=sum;
            }
        }
    }
    
    // std::cout<<"final weight matrices:\n";
    // std::cout<<(std::string) current.w()<<"\n";
    // for(size_t n=0;n<cluster.size();n++){
        // std::cout<<(std::string) cluster[n].w()<<"\n";
    // }
    
    //TEMP: convert old_current, old_cluster to exp_form since not in log space
    for(size_t i=0;i<current.w().nx();i++){
        for(size_t j=0;j<current.w().ny();j++){
            for(size_t k=0;k<current.w().nz();k++){
                current.w().at(i,j,k)=(current.w().at(i,j,k)>0)?log(current.w().at(i,j,k)):log(1e-100);
                current.w().at(i,j,k)=(current.w().at(i,j,k)>log(1e-100))?current.w().at(i,j,k):log(1e-100);
            }
        }
    }
    for(size_t n=0;n<cluster.size();n++){
        for(size_t i=0;i<cluster[n].w().nx();i++){
            for(size_t j=0;j<cluster[n].w().ny();j++){
                cluster[n].w().at(i,j,0)=(cluster[n].w().at(i,j,0)>0)?log(cluster[n].w().at(i,j,0)):log(1e-100);
                cluster[n].w().at(i,j,0)=(cluster[n].w().at(i,j,0)>log(1e-100))?cluster[n].w().at(i,j,0):log(1e-100);
            }
        }
    }
    return best_err;
}
