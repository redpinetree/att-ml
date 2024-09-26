#include <fstream>

#include "../mat_ops.hpp"
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

array3d<double> optimize::mu_renyi(bond& old_current,std::vector<bond>& old_cluster,bond& current,std::vector<bond>& cluster,size_t m,array3d<double>& x,std::vector<double>& weights,double z,double rho,double& final_cost){
    if(m!=0){
        for(size_t i=0;i<current.w().nx();i++){
            for(size_t j=0;j<current.w().ny();j++){
                for(size_t k=0;k<current.w().nz();k++){
                    current.w().at(i,j,k)*=weights[k];
                }
            }
        }
    }
    for(size_t k=0;k<current.w().nz();k++){
        weights[k]=1;
    }
    
    
    size_t r_i=current.w().nx();
    size_t r_j=current.w().ny();
    size_t r_k=current.w().nz();
    size_t dim;
    if(m==0){
        dim=r_i*r_j;
    }
    else{
        //dim is the rank at the site that is not part of the active bond
        bool dim_flag=((old_cluster[m-1].v1()==old_current.v1())||(old_cluster[m-1].v1()==old_current.v2()));
        dim=dim_flag?old_cluster[m-1].w().ny():old_cluster[m-1].w().nx();
    }
    // std::cout<<r_i<<" "<<r_j<<" "<<r_k<<" "<<dim<<"\n";
    std::vector<size_t> a_idx_max;
    //first index is whichever of i or j isn't connected to mode
    if(m!=0){
        a_idx_max.push_back(((old_cluster[m-1].v1()==old_current.v1())||(old_cluster[m-1].v2()==old_current.v1()))?r_j:r_i);
    }
    //other indices have rank r_imu, but skip mode m
    for(size_t n=0;n<old_cluster.size();n++){
        if((m!=0)&&(n==(m-1))){
            continue;
        }
        a_idx_max.push_back(((old_cluster[n].v1()==old_current.v1())||(old_cluster[n].v1()==old_current.v2()))?old_cluster[n].w().ny():old_cluster[n].w().nx());
    }
    array3d<double> prev_x=x;
        
    double eps=1e-16;
    double delta_first=0;
    double prev_cost=1e50;
    for(size_t it=0;it<1000;it++){
        //convergence check variables
        double cost=0;
        
        //calculate G'_i(S_k) and G'_j(S_k)
        std::vector<double> gi_prime(r_k,1);
        std::vector<double> gj_prime(r_k,1);
        for(size_t k=0;k<r_k;k++){
            std::vector<double> gi_prime_factors;
            std::vector<double> gj_prime_factors;
            for(size_t n=0;n<cluster.size();n++){
                if((old_cluster[n].v1()==old_current.v1())||(old_cluster[n].v2()==old_current.v1())){ //connected to site i
                    gi_prime_factors.push_back((cluster[n].w().sum_over_axis(0,2))[k]);
                }
                else{ //connected to site j
                    gj_prime_factors.push_back((cluster[n].w().sum_over_axis(0,2))[k]);
                }
            }
            gi_prime[k]=vec_mult_float(gi_prime_factors);
            gj_prime[k]=vec_mult_float(gj_prime_factors);
        }
        //calculate F'_i(S_i,S_k,S_k',a) and F'_j(S_j,S_k,S_k',a) and keep F factors
        std::vector<array3d<double> > f_factors;
        array3d<double> fi_prime(current.w().nx(),r_k,r_k);
        array3d<double> fj_prime(current.w().ny(),r_k,r_k);
        for(size_t n=0;n<cluster.size();n++){
            if((old_cluster[n].v1()==old_current.v1())||(old_cluster[n].v2()==old_current.v1())){ //connected to site i
                f_factors.push_back(array3d<double>(current.w().nx(),r_k,r_k));
            }
            else{ //connected to site j
                f_factors.push_back(array3d<double>(current.w().ny(),r_k,r_k));
            }
        }
        for(size_t k=0;k<r_k;k++){
            for(size_t k2=0;k2<r_k;k2++){
                std::vector<std::vector<double> > fi_prime_factors;
                std::vector<std::vector<double> > fj_prime_factors;
                for(size_t i=0;i<current.w().nx();i++){
                    fi_prime_factors.push_back(std::vector<double>());
                }
                for(size_t j=0;j<current.w().ny();j++){
                    fj_prime_factors.push_back(std::vector<double>());
                }
                for(size_t n=0;n<cluster.size();n++){
                    if((old_cluster[n].v1()==old_current.v1())||(old_cluster[n].v2()==old_current.v1())){ //connected to site i
                        for(size_t i=0;i<old_current.w().nx();i++){
                            std::vector<double> sum_addends;
                            for(size_t imu=0;imu<cluster[n].w().nx();imu++){
                                double num=cluster[n].w().at(imu,k2,0);
                                double denom;
                                if(old_cluster[n].v1()==current.v1()){ //site imu > site i
                                    denom=old_cluster[n].w().at(i,imu,0);
                                }
                                else{ //site imu < site i
                                    denom=old_cluster[n].w().at(imu,i,0);
                                }
                                sum_addends.push_back(pow(cluster[n].w().at(imu,k,0),1/(rho-1))*num/denom);
                            }
                            double sum=vec_add_float(sum_addends);
                            fi_prime_factors[i].push_back(sum);
                            f_factors[n].at(i,k,k2)=sum;
                        }
                    }
                    else{ //connected to site j
                        for(size_t j=0;j<old_current.w().ny();j++){
                            std::vector<double> sum_addends;
                            for(size_t imu=0;imu<cluster[n].w().nx();imu++){
                                double num=cluster[n].w().at(imu,k2,0);
                                double denom;
                                if(old_cluster[n].v1()==current.v2()){ //site imu > site j
                                    denom=old_cluster[n].w().at(j,imu,0);
                                }
                                else{ //site imu < site j
                                    denom=old_cluster[n].w().at(imu,j,0);
                                }
                                sum_addends.push_back(pow(cluster[n].w().at(imu,k,0),1/(rho-1))*num/denom);
                            }
                            double sum=vec_add_float(sum_addends);
                            fj_prime_factors[j].push_back(sum);
                            f_factors[n].at(j,k,k2)=sum;
                        }
                    }
                }
                for(size_t i=0;i<current.w().nx();i++){
                    fi_prime.at(i,k,k2)=vec_mult_float(fi_prime_factors[i]);
                }
                for(size_t j=0;j<current.w().ny();j++){
                    fj_prime.at(j,k,k2)=vec_mult_float(fj_prime_factors[j]);
                }
            }
        }
        //calculate P'_{ki_\mu} and Z' (as sum_p_prime_env)
        array2d<double> p_prime(x.nx(),x.ny());
        std::vector<std::vector<double> > addends_env(p_prime.nx()*p_prime.ny());
        std::vector<double> sum_p_prime_addends;
        if(m==0){
            for(size_t i=0;i<current.w().nx();i++){
                for(size_t j=0;j<current.w().ny();j++){
                    for(size_t k=0;k<current.w().nz();k++){
                        std::vector<double> factors_env;
                        factors_env.push_back(gi_prime[k]);
                        factors_env.push_back(gj_prime[k]);
                        p_prime.at(k,(current.w().ny()*i)+j)=vec_mult_float(factors_env);
                        sum_p_prime_addends.push_back(x.at(k,(current.w().ny()*i)+j,0)*p_prime.at(k,(current.w().ny()*i)+j));
                    }
                }
            }
        }
        else{
            for(size_t i=0;i<current.w().nx();i++){
                for(size_t j=0;j<current.w().ny();j++){
                    for(size_t k=0;k<x.nx();k++){
                        for(size_t imu=0;imu<x.ny();imu++){
                            std::vector<double> factors_env;
                            factors_env.push_back(current.w().at(i,j,k));
                            factors_env.push_back(gi_prime[k]);
                            factors_env.push_back(gj_prime[k]);
                            factors_env.push_back(1/(cluster[m-1].w().sum_over_axis(0,2))[k]); //divide appropriate factor
                            addends_env[(imu*p_prime.nx())+k].push_back(vec_mult_float(factors_env));
                            sum_p_prime_addends.push_back(x.at(k,imu,0)*vec_mult_float(factors_env)); //restore divided factor to get z'
                        }
                    }
                }
            }
            for(size_t k=0;k<x.nx();k++){
                for(size_t imu=0;imu<x.ny();imu++){
                    p_prime.at(k,imu)=vec_add_float(addends_env[(imu*p_prime.nx())+k]);
                }
            }
        }
        double z_prime=vec_add_float(sum_p_prime_addends);
        
        //normalize P'_{ki_\mu} by Z'
        // for(size_t k=0;k<x.nx();k++){
            // for(size_t imu=0;imu<x.ny();imu++){
                // p_prime.at(k,imu)/=z_prime;
            // }
        // }
        // std::cout<<"p_prime: "<<(std::string) p_prime<<"\n";
        // std::cout<<"z': "<<z_prime<<"\n";
        
        //calculate intermediate factors (and cost function as sum_ki_factors in the process)
        array2d<double> factors(x.nx(),x.ny());
        std::vector<double> sum_factors_addends;
        if(m==0){
            for(size_t i=0;i<current.w().nx();i++){
                for(size_t j=0;j<current.w().ny();j++){
                    for(size_t k=0;k<x.nx();k++){
                        std::vector<double> k2_sum_addends;
                        for(size_t k2=0;k2<x.nx();k2++){
                            k2_sum_addends.push_back(current.w().at(i,j,k2)/old_current.w().at(i,j,0)*fi_prime.at(i,k,k2)*fj_prime.at(j,k,k2));
                        }
                        factors.at(k,(current.w().ny()*i)+j)=pow(vec_add_float(k2_sum_addends),rho-1);
                        // factors.at(k,(current.w().ny()*i)+j)*=pow(z,rho-1)/pow(z_prime,rho);
                        sum_factors_addends.push_back(current.w().at(i,j,k)*factors.at(k,(current.w().ny()*i)+j));
                    }
                }
            }
        }
        else{
            //determine whether this bond was connected to site i or to site j
            size_t source=(old_cluster[m-1].v1()==old_current.v1()||old_cluster[m-1].v2()==old_current.v1())?old_current.v1():old_current.v2();
            for(size_t k=0;k<x.nx();k++){
                for(size_t imu=0;imu<x.ny();imu++){
                    std::vector<double> i2j2k2_sum_addends;
                    for(size_t i2=0;i2<current.w().nx();i2++){
                        for(size_t j2=0;j2<current.w().ny();j2++){
                            for(size_t k2=0;k2<x.nx();k2++){
                                double res=pow(current.w().at(i2,j2,k),1/(rho-1))*cluster[m-1].w().at(imu,k2,0);
                                double restored_f_factor;
                                if(source==current.v1()){ //connected to site i
                                    restored_f_factor=f_factors[m-1].at(i2,k,k2);
                                    if(old_cluster[m-1].v1()==current.v1()){ //site imu > site i
                                        res/=old_cluster[m-1].w().at(i2,imu,0);
                                    }
                                    else{ //site imu < site i
                                        res/=old_cluster[m-1].w().at(imu,i2,0);
                                    }
                                }
                                else{ //connected to site j
                                    restored_f_factor=f_factors[m-1].at(j2,k,k2);
                                    if(old_cluster[m-1].v1()==current.v2()){ //site imu > site j
                                        res/=old_cluster[m-1].w().at(j2,imu,0);
                                    }
                                    else{ //site imu < site j
                                        res/=old_cluster[m-1].w().at(imu,j2,0);
                                    }
                                }
                                res*=current.w().at(i2,j2,k2)/old_current.w().at(i2,j2,0)*fi_prime.at(i2,k,k2)*fj_prime.at(j2,k,k2)/restored_f_factor;
                                i2j2k2_sum_addends.push_back(res);
                            }
                        }
                    }
                    factors.at(k,imu)=pow(vec_add_float(i2j2k2_sum_addends),rho-1);
                    // factors.at(k,imu)*=pow(z,rho-1)/pow(z_prime,rho);
                    sum_factors_addends.push_back(cluster[m-1].w().at(imu,k,0)*factors.at(k,imu));
                }
            }
        }
        double sum_factors=vec_add_float(sum_factors_addends);
        cost=log(sum_factors*pow(z,rho-1)/pow(z_prime,rho));
        // std::cout<<"factors:"<<(std::string)factors<<"\n";
        // std::cout<<"sum_factors:"<<sum_factors<<"\n";
        // std::cout<<"cost:"<<cost<<"\n";
        
        std::vector<double> sum_addends;
        for(size_t k=0;k<x.nx();k++){
            for(size_t imu=0;imu<x.ny();imu++){
                // x.at(k,imu,0)*=pow((p_prime.at(k,imu)*sum_factors)/factors.at(k,imu),(rho>1)?1/rho:1);
                x.at(k,imu,0)*=pow((p_prime.at(k,imu)*sum_factors)/(factors.at(k,imu)*z_prime),(rho>1)?1/rho:1);
                sum_addends.push_back(x.at(k,imu,0));
            }
        }
        double sum=vec_add_float(sum_addends);
        for(size_t k=0;k<x.nx();k++){
            for(size_t imu=0;imu<x.ny();imu++){
                x.at(k,imu,0)/=sum;
                if(x.at(k,imu,0)<1e-100){ //in case the weight is too small
                    x.at(k,imu,0)=1e-100;
                }
            }
        }
        // std::cout<<"updated x:"<<(std::string)x<<"\n";
        
        array3d<double> xT=transpose(x);
        if(m==0){
            current.w()=tensorize(xT,r_i,r_j,2);
        }
        else{
            cluster[m-1].w()=xT;
        }
        
        if((cost>prev_cost)&&(fabs(cost-prev_cost)>1e-16)&&(it>1)){ //at it=1, prev_cost is init's cost
            std::cout<<prev_cost<<" "<<cost<<" objective function increased at iteration "<<it<<"\n";
            break;
            // exit(1);
        }
        else{
            // std::cout<<prev_cost<<" "<<cost<<"\n";
        }
        prev_cost=cost;
    
        double delta=0;
        for(size_t i=0;i<x.nx();i++){
            for(size_t j=0;j<x.ny();j++){
                delta+=(x.at(i,j,0)-prev_x.at(i,j,0))*(x.at(i,j,0)-prev_x.at(i,j,0));
            }
        }
        delta=sqrt(delta);
        prev_x=x;
        if(fabs(cost)<eps){
            // std::cout<<"MU-Renyi stopped early after "<<it<<" iterations (cost).\n";
            break;
        }
        else if(it==0){
            delta_first=delta;
        }
        else if(delta<=(eps*delta_first)){
            // std::cout<<"MU-Renyi stopped early after "<<it<<" iterations (delta).\n";
            break;
        }
    }
    final_cost=prev_cost;
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

double optimize::tree_cpd(size_t master,size_t slave,std::vector<site>& sites,bond& old_current,std::vector<bond>& old_cluster,bond& current,std::vector<bond>& cluster,size_t max_it,std::string init_method,std::string solver,bool unnormalize_flag){
    std::uniform_real_distribution<> unif_dist(1e-16,1.0);
    size_t r_i=current.w().nx();
    size_t r_j=current.w().ny();
    size_t r_k=current.w().nz();
    std::vector<double> weights(r_k,1);
    double eps=1e-16;
    
    //precompute intermediate quantities for weight optimization for murenyi or cmd (if needed)
    //calculate G_i(S_i) and G_j(S_j)
    std::vector<double> gi(old_current.w().nx(),1);
    std::vector<double> gj(old_current.w().ny(),1);
    double z;
    if((init_method=="cmd")||(solver=="murenyi")){
        for(size_t i=0;i<old_current.w().nx();i++){
            std::vector<double> gi_factors;
            for(size_t n=0;n<old_cluster.size();n++){
                if((old_cluster[n].v1()==old_current.v1())||(old_cluster[n].v2()==old_current.v1())){
                    gi_factors.push_back((old_cluster[n].w().sum_over_axis((old_cluster[n].v1()==old_current.v1())?1:0,2))[i]);
                }
            }
            gi[i]=vec_mult_float(gi_factors);
        }
        for(size_t j=0;j<old_current.w().ny();j++){
            std::vector<double> gj_factors;
            for(size_t n=0;n<old_cluster.size();n++){
                if((old_cluster[n].v1()==old_current.v2())||(old_cluster[n].v2()==old_current.v2())){
                    gj_factors.push_back((old_cluster[n].w().sum_over_axis((old_cluster[n].v1()==old_current.v2())?1:0,2))[j]);
                }
            }
            gj[j]=vec_mult_float(gj_factors);
        }
        //calculate Z stored as ln(Z)
        std::vector<double> z_addends;
        for(size_t i=0;i<old_current.w().nx();i++){
            for(size_t j=0;j<old_current.w().ny();j++){
                z_addends.push_back(old_current.w().at(i,j,0)*gi[i]*gj[j]);
            }
        }
        z=vec_add_float(z_addends);
    }
    
    //update spin cluster via cmd if needed
    if(init_method=="cmd"){
        //TEMP: convert old_current, old_cluster, current, cluster to log form for this calculation
        for(size_t i=0;i<old_current.w().nx();i++){
            for(size_t j=0;j<old_current.w().ny();j++){
                old_current.w().at(i,j,0)=log(old_current.w().at(i,j,0));
            }
        }
        for(size_t n=0;n<old_cluster.size();n++){
            for(size_t i=0;i<old_cluster[n].w().nx();i++){
                for(size_t j=0;j<old_cluster[n].w().ny();j++){
                    old_cluster[n].w().at(i,j,0)=log(old_cluster[n].w().at(i,j,0));
                }
            }
        }
        for(size_t i=0;i<current.w().nx();i++){
            for(size_t j=0;j<current.w().ny();j++){
                for(size_t k=0;k<current.w().nz();k++){
                    current.w().at(i,j,k)=log(current.w().at(i,j,k));
                }
            }
        }
        for(size_t n=0;n<cluster.size();n++){
            for(size_t i=0;i<cluster[n].w().nx();i++){
                for(size_t j=0;j<cluster[n].w().ny();j++){
                    cluster[n].w().at(i,j,0)=log(cluster[n].w().at(i,j,0));
                }
            }
        }
        calc_cmd(master,slave,sites,old_current,old_cluster,current,cluster,gi,gj,z);
        old_current.w()=old_current.w().exp_form();
        current.w()=current.w().exp_form();
        // std::cout<<(std::string)current.w()<<"\n";
        for(size_t n=0;n<cluster.size();n++){
            old_cluster[n].w()=old_cluster[n].w().exp_form();
            cluster[n].w()=cluster[n].w().exp_form();
            // std::cout<<(std::string)cluster[n].w()<<"\n";
        }
    }
    
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
    
    array3d<double> mat_current=matricize(current.w(),2);
    aTa_legs.push_back(matmul_xTy(mat_current,mat_current));
    aTb_legs.push_back(hadamard(current.w(),old_current.w()));
    for(size_t n=0;n<cluster.size();n++){
        //axis 1 of old_cluster[n] corresponds to site i/j, axis 1 of cluster[n] corresponds to site k
        array3d<double> arg=((old_cluster[n].v2()==old_current.v1())||(old_cluster[n].v2()==old_current.v2()))?old_cluster[n].w():transpose(old_cluster[n].w());
        aTa_legs.push_back(matmul_xTy(cluster[n].w(),cluster[n].w())); //axis 1 of cluster[n] corresponds to site k
        aTb_legs.push_back(matmul_xTy(cluster[n].w(),arg));
        bTb_legs.push_back(matmul_xTy(arg,arg));
    }
    
    double tr_axTax=optimize::calc_tr_axTax(aTa_legs,weights);
    double tr_axTb=optimize::calc_tr_axTb(aTb_legs,old_current,old_cluster,weights);
    double tr_bTb=optimize::calc_tr_bTb(bTb_legs,old_current,old_cluster);
    double prev_err=0;
    double final_cost;
    double err;
    if(solver!="murenyi"){
        err=(tr_axTax+tr_bTb-(2*tr_axTb))/tr_bTb;
    }
    else{
        err=renyi_div(old_current,old_cluster,current,cluster,weights,z,2);
    }
    
    std::cout<<"initial cost: "<<err<<"\n";
    for(size_t it=0;it<max_it;it++){
        // std::cout<<"iteration "<<it<<"\n";
        for(size_t m=0;m<cluster.size()+1;m++){
            //calculate aTa and aTb
            array3d<double> aTa=optimize::calc_aTa(aTa_legs,weights,m);
            array3d<double> aTb=optimize::calc_aTb(aTb_legs,old_current,old_cluster,weights,m);
            
            array3d<double> init(aTb.nx(),aTb.ny(),1);
            if(init_method=="prev"){ //initialize using previous iterate
                init=(m==0)?matricize(current.w(),2):cluster[m-1].w();
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
                // std::cout<<(std::string)init<<"\n";
                // array3d<double> a=tensorize(init,r_i,r_j,2);
                // std::cout<<(std::string)a<<"\n";
                // array3d<double> b=matricize(a,2);
                // std::cout<<(std::string)b<<"\n";
            }
            else if(init_method=="lstsq"){
                size_t status=0;
                lstsq(aTa,aTb,init,status);
                if(status==1){ //failed to converge, so fall back to "prev" method
                    std::cout<<"Falling back to \"prev\" method...\n";
                    init=(m==0)?matricize(current.w(),2):cluster[m-1].w();
                    init=transpose(init);
                }
            }
            else if(init_method=="svd"){
                array3d<double> dummy_mat1;
                size_t status=0;
                nndsvda(aTb,dummy_mat1,init,init.nx(),status);
                if(status==1){ //failed to converge, so fall back to "prev" method
                    std::cout<<"Falling back to \"prev\" method...\n";
                    init=(m==0)?matricize(current.w(),2):cluster[m-1].w();
                    init=transpose(init);
                }
            }
            else if(init_method=="cmd"){
                init=(m==0)?matricize(current.w(),2):cluster[m-1].w();
                init=transpose(init);
            }
            else{
                std::cout<<"Invalid initializer ("<<init_method<<" was passed), aborting...\n";
                exit(1);
            }
            
            array3d<double> x;
            // std::cout<<(std::string)init<<"\n";
            if(solver=="nnhals"){ //solve aTax=aTb via NN-HALS
                x=nn_hals(aTa,aTb,init,1000);
                x=transpose(x);
            }
            else if(solver=="muls"){ //solve aTax=aTb via MU
                x=mu_ls(aTa,aTb,init,1000);
                x=transpose(x);
            }
            else if(solver=="mukl"){ //solve aTax=aTb via MU
                x=mu_kl_ls(aTa,aTb,init,1000);
                x=transpose(x);
            }
            else if(solver=="murenyi"){ //solve aTax=aTb via MU
                x=optimize::mu_renyi(old_current,old_cluster,current,cluster,m,init,weights,z,2,final_cost);
            }
            // std::cout<<(std::string)x<<"\n";
            
            if(m==0){
                current.w()=tensorize(x,r_i,r_j,2);
            }
            else{
                cluster[m-1].w()=x;
            }
            
            //normalize factors based on weights for numerical stability
            // std::cout<<(std::string)current.w()<<"\n";
            // std::cout<<current.w().sum_over_all()<<"\n";
            if(solver!="murenyi"){ //do not normalize in renyi approach so that sites are delta tensors
                optimize::normalize(current,cluster,weights);
            }
            // std::cout<<(std::string)current.w()<<"\n";
            // std::cout<<current.w().sum_over_all()<<"\n";
            // optimize::unnormalize(current,weights);
            // std::cout<<(std::string)current.w()<<"\n";
            // std::cout<<current.w().sum_over_all()<<"\n";
            // array3d<double> b=matricize(current.w(),2);
            // std::cout<<(std::string)b<<"\n";
            // exit(1);
            
            //update cached quantities
            if(m==0){ //w_ijk
                array3d<double> mat_current=matricize(current.w(),2);
                aTa_legs[0]=matmul_xTy(mat_current,mat_current);
                aTb_legs[0]=hadamard(current.w(),old_current.w());
                continue;
            }
            array3d<double> arg=((old_cluster[m-1].v2()==old_current.v1())||(old_cluster[m-1].v2()==old_current.v2()))?old_cluster[m-1].w():transpose(old_cluster[m-1].w());
            aTa_legs[m]=matmul_xTy(cluster[m-1].w(),cluster[m-1].w());
            aTb_legs[m]=matmul_xTy(cluster[m-1].w(),arg);
        }
        //calculate reconstruction error |ax-b|^2_F, normalized by tr_bTb
        tr_axTax=optimize::calc_tr_axTax(aTa_legs,weights);
        tr_axTb=optimize::calc_tr_axTb(aTb_legs,old_current,old_cluster,weights);
        if(solver!="murenyi"){
            err=(tr_axTax+tr_bTb-(2*tr_axTb))/tr_bTb;
        }
        else{
            err=final_cost;
        }
        if((it>0)&&(fabs(prev_err-err)<eps)){
            std::cout<<"CPD converged after "<<it<<" iterations.\n";
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

double optimize::opt(size_t master,size_t slave,size_t r_k,std::vector<site>& sites,bond& old_current,std::vector<bond>& old_cluster,bond& current,std::vector<bond>& cluster,size_t max_it,std::string init_method,std::string solver,size_t max_restarts){
    std::cout<<"cluster size: "<<cluster.size()<<"\n";
    std::uniform_real_distribution<> unif_dist(1e-16,1.0);
    double best_err=1;
    if((init_method!="prev")&&(init_method!="lstsq")&&(init_method!="svd")&&(init_method!="rand")&&(init_method!="hybrid")&&(init_method!="cmd")){
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
        size_t r_i=old_current.w().nx();
        size_t r_j=old_current.w().ny();
        double err;
        if(init_method=="hybrid"){
            std::string hybrid_init_method;
            if(restarts==1){ //first attempt uses padded target data
                hybrid_init_method="prev";
            }
            else if((restarts==2)&&(r_i==r_j)&&(r_j==r_k)){ //second attempt uses cmd initialization, if possible
                hybrid_init_method="cmd";
            }
            else if(restarts==3){ //third attempt uses least-squares approximation
                hybrid_init_method="lstsq";
            }
            else{ //remaining attempts use random initialization
                hybrid_init_method="rand";
            }
            err=optimize::tree_cpd(master,slave,sites,old_current,old_cluster,trial_current,trial_cluster,max_it,hybrid_init_method,solver,1);
        }
        else{
            err=optimize::tree_cpd(master,slave,sites,old_current,old_cluster,trial_current,trial_cluster,max_it,init_method,solver,1);
        }
        // double kl=kl_div(sites,old_current,old_cluster,current,cluster);
        std::cout<<"cost: "<<err<<"\n";
        
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

double optimize::calc_cmd(size_t master,size_t slave,std::vector<site>& sites,bond& old_current,std::vector<bond>& old_cluster,bond& current,std::vector<bond>& cluster,std::vector<double>& gi,std::vector<double>& gj,double z){
    size_t max_it=100;
    double r_k=current.w().ny();
    //compute cmd function representing alg1
    site v_i=sites[old_current.v1()];
    site v_j=sites[old_current.v2()];
    //determine old volumes before reconnection
    size_t vol_i=(old_current.v1()==master)?v_i.vol()-v_j.vol():v_i.vol();
    size_t vol_j=(old_current.v2()==master)?v_j.vol()-v_i.vol():v_j.vol();
    array2d<size_t> f(v_i.rank(),v_j.rank());
    for(size_t s_i=0;s_i<v_i.rank();s_i++){
        for(size_t s_j=0;s_j<v_j.rank();s_j++){
            f.at(s_i,s_j)=(vol_i>vol_j)?s_i:s_j;
        }
    }
    
    bond trial_current=current;
    std::vector<bond> trial_cluster;
    for(size_t n=0;n<cluster.size();n++){
        trial_cluster.push_back(cluster[n]);
    }
    //reinitialize weight matrices because cmd is only valid when r_i=r_j=r_k
    array3d<double> new_w(trial_current.w().nx(),trial_current.w().ny(),r_k);
    double sum=0;
    for(size_t i=0;i<new_w.nx();i++){
        for(size_t j=0;j<new_w.ny();j++){
            new_w.at(i,j,f.at(i,j))=exp(old_current.w().at(i,j,0));
            sum+=new_w.at(i,j,f.at(i,j));
        }
    }
    for(size_t i=0;i<new_w.nx();i++){
        for(size_t j=0;j<new_w.ny();j++){
            for(size_t k=0;k<new_w.nz();k++){
                new_w.at(i,j,k)/=sum;
                new_w.at(i,j,k)=log(new_w.at(i,j,k));
            }
        }
    }
    trial_current.w()=new_w;
    array3d<double> prev_current=trial_current.w();
    std::vector<array3d<double> > prev_cluster;
    for(size_t n=0;n<trial_cluster.size();n++){
        prev_cluster.push_back(trial_cluster[n].w());
    }
    double cost=0;
    double prev_cost=1e50;
    array2d<double> p_ij(trial_current.w().nx(),trial_current.w().ny());
    array2d<double> p_prime_ij_env(trial_current.w().nx(),trial_current.w().ny());
    for(size_t t=0;t<max_it;t++){
        //calculate P_{ij},P'^{env}_{ij} and Z' (as sum_p_prime_ij_env)
        std::vector<double> sum_p_ij_addends;
        std::vector<double> sum_p_prime_ij_env_addends;
        std::vector<double> gi_prime(r_k,1);
        std::vector<double> gj_prime(r_k,1);
        for(size_t i=0;i<trial_current.w().nx();i++){
            for(size_t j=0;j<trial_current.w().ny();j++){
                //calculate G'(f_k(S_i,S_j))
                std::vector<double> gi_prime_factors;
                std::vector<double> gj_prime_factors;
                size_t k=f.at(i,j); //select where output is added to
                for(size_t m=0;m<trial_cluster.size();m++){
                    if((old_cluster[m].v1()==old_current.v1())||(old_cluster[m].v2()==old_current.v1())){ //connected to site i
                        gi_prime_factors.push_back((trial_cluster[m].w().lse_over_axis(0,2))[k]);
                    }
                    else{ //connected to site j
                        gj_prime_factors.push_back((trial_cluster[m].w().lse_over_axis(0,2))[k]);
                    }
                }
                gi_prime[k]=vec_add_float(gi_prime_factors);
                gj_prime[k]=vec_add_float(gj_prime_factors);
            }
        }
        for(size_t i=0;i<p_ij.nx();i++){
            for(size_t j=0;j<p_ij.ny();j++){
                std::vector<double> factors;
                std::vector<double> factors_env;
                //calculate P_{ij}
                factors.push_back(old_current.w().at(i,j,0));
                factors.push_back(log(gi[i]));
                factors.push_back(log(gj[j]));
                p_ij.at(i,j)=vec_add_float(factors);
                sum_p_ij_addends.push_back(p_ij.at(i,j));
                //calculate P'^{env}_{ij}
                size_t k=f.at(i,j); //select where output is added to
                factors_env.push_back(gi_prime[k]);
                factors_env.push_back(gj_prime[k]);
                p_prime_ij_env.at(i,j)=vec_add_float(factors_env);
                sum_p_prime_ij_env_addends.push_back(p_prime_ij_env.at(i,j)+trial_current.w().at(i,j,(trial_current.w().nz()==1)?0:f.at(i,j))); //restore divided factor to get z'
            }
        }
        //normalize P_{ij},P'^{env}_{ij}
        double sum_p_ij=lse(sum_p_ij_addends);
        double sum_p_prime_ij_env=lse(sum_p_prime_ij_env_addends);
        for(size_t i=0;i<p_ij.nx();i++){
            for(size_t j=0;j<p_ij.ny();j++){
                p_ij.at(i,j)-=sum_p_ij;
                p_prime_ij_env.at(i,j)-=sum_p_prime_ij_env;
            }
        }
        double z_prime=sum_p_prime_ij_env;
        cost=kl_div(z,exp(z_prime),gi,gj,old_current,old_cluster,trial_current,trial_cluster,f);
        // std::cout<<sum_p_prime_ij_env<<"\n";
        // std::cout<<(std::string)trial_current.w()<<"\n";
        // std::cout<<(std::string)p_ij<<"\n";
        // std::cout<<(std::string)p_prime_ij_env<<"\n";
        
        std::vector<double> sum_addends;
        for(size_t i=0;i<p_ij.nx();i++){
            for(size_t j=0;j<p_ij.ny();j++){
                trial_current.w().at(i,j,f.at(i,j))=p_ij.at(i,j)-((fabs(p_prime_ij_env.at(i,j))<1e-16)?0:p_prime_ij_env.at(i,j));
                sum_addends.push_back(trial_current.w().at(i,j,f.at(i,j)));
            }
        }
        double sum=lse(sum_addends);
        for(size_t i=0;i<trial_current.w().nx();i++){
            for(size_t j=0;j<trial_current.w().ny();j++){
                trial_current.w().at(i,j,f.at(i,j))-=sum;
                if(trial_current.w().at(i,j,f.at(i,j))<log(1e-100)){ //in case the weight is negative, force it to be nonnegative!
                    trial_current.w().at(i,j,f.at(i,j))=log(1e-100);
                }
            }
        }
        // std::cout<<"trial_current.w():\n"<<(std::string)trial_current.w().exp_form()<<"\n";
        
        //p_prime_ki,imu if source==current.v1(), p_prime_kj,jnu if source==current.v2(). wlog, use p_prime_ki and imu.
        //k, the new index, is always second because virtual indices > physical indices.
        for(size_t n=0;n<trial_cluster.size();n++){
            //determine whether this bond was connected to site i or to site j
            size_t source;
            if(old_cluster[n].v1()==master||old_cluster[n].v1()==slave){
                source=(old_cluster[n].v1()==master)?master:slave;
            }
            else{
                source=(old_cluster[n].v2()==master)?master:slave;
            }
            
            array2d<double> p_ki(trial_cluster[n].w().nx(),trial_cluster[n].w().ny());
            array2d<double> p_prime_ki_env(trial_cluster[n].w().nx(),trial_cluster[n].w().ny());
            
            //prep to calculate P^_{ki_\mu}, P'^{env}_{ki_\mu} and Z' (as sum_p_prime_ki_env)
            std::vector<std::vector<double> > addends(p_ki.nx()*p_ki.ny());
            std::vector<std::vector<double> > addends_env(p_prime_ki_env.nx()*p_prime_ki_env.ny());
            std::vector<double> sum_p_ki_addends;
            std::vector<double> sum_p_prime_ki_env_addends;
            for(size_t i=0;i<trial_current.w().nx();i++){
                for(size_t j=0;j<trial_current.w().ny();j++){
                    size_t k=f.at(i,j); //select where output is added to
                    //get addends
                    for(size_t imu=0;imu<p_ki.nx();imu++){
                        std::vector<double> factors;
                        std::vector<double> factors_env;
                        //calculate P_{ki_\mu} addends
                        factors.push_back(old_current.w().at(i,j,0));
                        if(source==trial_current.v1()){ //connected to site i
                            if(old_cluster[n].v1()==trial_current.v1()){ //site imu > site i
                                factors.push_back(old_cluster[n].w().at(i,imu,0));
                                factors.push_back(-(old_cluster[n].w().lse_over_axis(1,2))[i]);
                            }
                            else{ //site imu < site i
                                factors.push_back(old_cluster[n].w().at(imu,i,0));
                                factors.push_back(-(old_cluster[n].w().lse_over_axis(0,2))[i]);
                            }
                        }
                        else{ //connected to site j
                            if(old_cluster[n].v1()==trial_current.v2()){ //site imu > site j
                                factors.push_back(old_cluster[n].w().at(j,imu,0));
                                factors.push_back(-(old_cluster[n].w().lse_over_axis(1,2))[j]);
                            }
                            else{ //site imu < site j
                                factors.push_back(old_cluster[n].w().at(imu,j,0));
                                factors.push_back(-(old_cluster[n].w().lse_over_axis(0,2))[j]);
                            }
                        }
                        factors.push_back(log(gi[i]));
                        factors.push_back(log(gj[j]));
                        addends[(k*p_ki.nx())+imu].push_back(vec_add_float(factors));
                        sum_p_ki_addends.push_back(vec_add_float(factors));
                        //calculate P'^{env}_{ki_\mu} addends
                        factors_env.push_back(trial_current.w().at(i,j,k));
                        factors_env.push_back(gi_prime[k]);
                        factors_env.push_back(gj_prime[k]);
                        factors_env.push_back(-(trial_cluster[n].w().lse_over_axis(0,2))[k]); //divide appropriate factor
                        addends_env[(k*p_prime_ki_env.nx())+imu].push_back(vec_add_float(factors_env));
                        sum_p_prime_ki_env_addends.push_back(vec_add_float(factors_env)+trial_cluster[n].w().at(imu,k,0)); //restore divided factor to get z'
                    }
                }
            }
            //calculate P^_{ki_\mu}, P'^{env}_{ki_\mu}
            for(size_t imu=0;imu<p_ki.nx();imu++){
                for(size_t k=0;k<p_ki.ny();k++){
                    p_ki.at(imu,k)=lse(addends[(k*p_ki.nx())+imu]);
                    p_prime_ki_env.at(imu,k)=lse(addends_env[(k*p_ki.nx())+imu]);
                }
            }
            double sum_p_ki=lse(sum_p_ki_addends);
            double sum_p_prime_ki_env=lse(sum_p_prime_ki_env_addends);
            // std::cout<<"p_ki: "<<(std::string) p_ki<<"\n";
            // std::cout<<"p_ki_env: "<<(std::string) p_prime_ki_env<<"\n";
            
            //normalize P_{ki_\mu},P'^{env}_{ki_\mu}
            for(size_t i=0;i<p_ki.nx();i++){
                for(size_t j=0;j<p_ki.ny();j++){
                    p_ki.at(i,j)-=sum_p_ki;
                    p_prime_ki_env.at(i,j)-=sum_p_prime_ki_env;
                }
            }
            z_prime=sum_p_prime_ki_env;
            // std::cout<<sum_p_prime_ki_env<<"\n";
            
            std::vector<double> sum_addends;
            for(size_t imu=0;imu<p_ki.nx();imu++){
                for(size_t k=0;k<p_ki.ny();k++){
                    trial_cluster[n].w().at(imu,k,0)=p_ki.at(imu,k)-((fabs(p_prime_ki_env.at(imu,k))<1e-16)?0:p_prime_ki_env.at(imu,k));
                    sum_addends.push_back(trial_cluster[n].w().at(imu,k,0));
                }
            }
            double sum=lse(sum_addends);
            for(size_t imu=0;imu<trial_cluster[n].w().nx();imu++){
                for(size_t k=0;k<trial_cluster[n].w().ny();k++){
                    trial_cluster[n].w().at(imu,k,0)-=sum;
                    if(trial_cluster[n].w().at(imu,k,0)<log(1e-100)){ //in case the weight is negative, force it to be nonnegative!
                        trial_cluster[n].w().at(imu,k,0)=log(1e-100);
                    }
                }
            }
            // std::cout<<"trial_cluster[n].w():\n"<<(std::string)trial_cluster[n].w().exp_form()<<"\n";
        }
        
        // std::cout<<cost<<"\n";
        if(t>0){
            if(fabs(prev_cost-cost)<1e-16){
                std::cout<<"converged after "<<(t+1)<<" iterations (cost)\n";
                break;
            }
            if(t==max_it-1){
                std::cout<<"no convergence after "<<(max_it)<<" iterations\n";
                // std::cout<<"fail:\n";
                // std::cout<<(std::string) current.w().exp_form()<<"\n";
                // for(size_t a=0;a<cluster.size();a++){
                    // std::cout<<(std::string) cluster[a].w().exp_form()<<"\n";
                // }
                // std::cout<<(std::string) trial_current.w().exp_form()<<"\n";
                // for(size_t a=0;a<trial_cluster.size();a++){
                    // std::cout<<(std::string) trial_cluster[a].w().exp_form()<<"\n";
                // }
                // exit(1);
            }
            prev_cost=cost;
        }
        
        //update convergence check variables
        prev_current=trial_current.w();
        for(size_t n=0;n<trial_cluster.size();n++){
            prev_cluster[n]=trial_cluster[n].w();
        }
    }
    //update variables
    current.w()=trial_current.w();
    for(size_t n=0;n<trial_cluster.size();n++){
        cluster[n].w()=trial_cluster[n].w();
    }
    std::cout<<"final KL cost: "<<cost<<"\n";
    return cost;
}

double optimize::kl_div(double z,double z_prime,std::vector<double>& gi,std::vector<double>& gj,bond& old_current,std::vector<bond>& old_cluster,bond& current,std::vector<bond>& cluster,array2d<size_t>& f){
    array2d<double> gi_tilde(old_current.w().nx(),old_current.w().ny());
    array2d<double> gj_tilde(old_current.w().nx(),old_current.w().ny());
    for(size_t i=0;i<old_current.w().nx();i++){
        for(size_t j=0;j<old_current.w().ny();j++){
            double gi_tilde_res=0;
            double gj_tilde_res=0;
            for(size_t n=0;n<old_cluster.size();n++){
                double spin_sum_i_tilde=0;
                double spin_sum_j_tilde=0;
                double div_factor=0;
                if(old_cluster[n].v1()==old_current.v1()||old_cluster[n].v2()==old_current.v1()){ //connected to site i
                    if(old_cluster[n].v1()==old_current.v1()){ //site imu > site i
                        for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                            div_factor+=exp(old_cluster[n].w().at(i,imu,0));
                            spin_sum_i_tilde+=exp(old_cluster[n].w().at(i,imu,0))*old_cluster[n].w().at(i,imu,0);
                        }
                    }
                    else{ //site imu < site i
                        for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                            div_factor+=exp(old_cluster[n].w().at(imu,i,0));
                            spin_sum_i_tilde+=exp(old_cluster[n].w().at(imu,i,0))*old_cluster[n].w().at(imu,i,0);
                        }
                    }
                    gi_tilde_res+=spin_sum_i_tilde/div_factor;
                }
                else{ //connected to site j
                    if(old_cluster[n].v1()==old_current.v2()){ //site imu > site j
                        for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                            div_factor+=exp(old_cluster[n].w().at(j,imu,0));
                            spin_sum_j_tilde+=exp(old_cluster[n].w().at(j,imu,0))*old_cluster[n].w().at(j,imu,0);
                        }
                    }
                    else{ //site imu < site j
                        for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                            div_factor+=exp(old_cluster[n].w().at(imu,j,0));
                            spin_sum_j_tilde+=exp(old_cluster[n].w().at(imu,j,0))*old_cluster[n].w().at(imu,j,0);
                        }
                    }
                    gj_tilde_res+=spin_sum_j_tilde/div_factor;
                }
            }
            gi_tilde.at(i,j)=gi_tilde_res*gi[i];
            gj_tilde.at(i,j)=gj_tilde_res*gj[j];
        }
    }
    
    array2d<double> gi_prime_tilde(old_current.w().nx(),old_current.w().ny());
    array2d<double> gj_prime_tilde(old_current.w().nx(),old_current.w().ny());
    for(size_t i=0;i<current.w().nx();i++){
        for(size_t j=0;j<current.w().ny();j++){
            double gi_prime_tilde_res=0;
            double gj_prime_tilde_res=0;
            for(size_t n=0;n<old_cluster.size();n++){
                double spin_sum_i_tilde=0;
                double spin_sum_j_tilde=0;
                double div_factor=0;
                size_t k=f.at(i,j);
                if(old_cluster[n].v1()==old_current.v1()||old_cluster[n].v2()==old_current.v1()){ //connected to site i
                    if(old_cluster[n].v1()==old_current.v1()){ //site imu > site i
                        for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                            div_factor+=exp(old_cluster[n].w().at(i,imu,0));
                            spin_sum_i_tilde+=exp(old_cluster[n].w().at(i,imu,0))*cluster[n].w().at(imu,k,0);
                        }
                    }
                    else{ //site imu < site i
                        for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                            div_factor+=exp(old_cluster[n].w().at(imu,i,0));
                            spin_sum_i_tilde+=exp(old_cluster[n].w().at(imu,i,0))*cluster[n].w().at(imu,k,0);
                        }
                    }
                    gi_prime_tilde_res+=spin_sum_i_tilde/div_factor;
                }
                else{ //connected to site j
                    if(old_cluster[n].v1()==old_current.v2()){ //site imu > site j
                        for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                            div_factor+=exp(old_cluster[n].w().at(j,imu,0));
                            spin_sum_j_tilde+=exp(old_cluster[n].w().at(j,imu,0))*cluster[n].w().at(imu,k,0);
                        }
                    }
                    else{ //site imu < site j
                        for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                            div_factor+=exp(old_cluster[n].w().at(imu,j,0));
                            spin_sum_j_tilde+=exp(old_cluster[n].w().at(imu,j,0))*cluster[n].w().at(imu,k,0);
                        }
                    }
                    gj_prime_tilde_res+=spin_sum_j_tilde/div_factor;
                }
            }
            gi_prime_tilde.at(i,j)=gi_prime_tilde_res*gi[i];
            gj_prime_tilde.at(i,j)=gj_prime_tilde_res*gj[j];
        }
    }
    
    double a=0;
    double b=0;
    for(size_t i=0;i<old_current.w().nx();i++){
        for(size_t j=0;j<old_current.w().ny();j++){
            double contrib_a=exp(old_current.w().at(i,j,0))*gi[i]*gj[j]*old_current.w().at(i,j,0);
            double contrib_b=exp(old_current.w().at(i,j,0))*((gi_tilde.at(i,j)*gj[j])+(gi[i]*gj_tilde.at(i,j)));
            a+=contrib_a;
            b+=contrib_b;
        }
    }
    double a_prime=0;
    double b_prime=0;
    for(size_t i=0;i<current.w().nx();i++){
        for(size_t j=0;j<current.w().ny();j++){
            double contrib_a_prime=exp(old_current.w().at(i,j,0))*gi[i]*gj[j]*current.w().at(i,j,f.at(i,j));
            double contrib_b_prime=exp(old_current.w().at(i,j,0))*((gi_prime_tilde.at(i,j)*gj[j])+(gi[i]*gj_prime_tilde.at(i,j)));
            a_prime+=contrib_a_prime;
            b_prime+=contrib_b_prime;
        }
    }
    
    
    double res=((a/z)+(b/z)-log(z))-((a_prime/z)+(b_prime/z)-log(z_prime));
    return res;
}

double optimize::renyi_div(bond& old_current,std::vector<bond>& old_cluster,bond& current,std::vector<bond>& cluster,std::vector<double>& weights,double z,double rho){
    bond current_bk=current;
    for(size_t i=0;i<current.w().nx();i++){
        for(size_t j=0;j<current.w().ny();j++){
            for(size_t k=0;k<current.w().nz();k++){
                current.w().at(i,j,k)*=weights[k];
            }
        }
    }
    size_t r_i=current.w().nx();
    size_t r_j=current.w().ny();
    size_t r_k=current.w().nz();
    
    double cost=0;
        
    //calculate G'_i(S_k) and G'_j(S_k)
    std::vector<double> gi_prime(r_k,1);
    std::vector<double> gj_prime(r_k,1);
    for(size_t k=0;k<r_k;k++){
        std::vector<double> gi_prime_factors;
        std::vector<double> gj_prime_factors;
        for(size_t n=0;n<cluster.size();n++){
            if((old_cluster[n].v1()==old_current.v1())||(old_cluster[n].v2()==old_current.v1())){ //connected to site i
                gi_prime_factors.push_back((cluster[n].w().sum_over_axis(0,2))[k]);
            }
            else{ //connected to site j
                gj_prime_factors.push_back((cluster[n].w().sum_over_axis(0,2))[k]);
            }
        }
        gi_prime[k]=vec_mult_float(gi_prime_factors);
        gj_prime[k]=vec_mult_float(gj_prime_factors);
    }
    //calculate F'_i(S_i,S_k,S_k',a) and F'_j(S_j,S_k,S_k',a) and keep F factors
    std::vector<array3d<double> > f_factors;
    array3d<double> fi_prime(current.w().nx(),r_k,r_k);
    array3d<double> fj_prime(current.w().ny(),r_k,r_k);
    for(size_t n=0;n<cluster.size();n++){
        if((old_cluster[n].v1()==old_current.v1())||(old_cluster[n].v2()==old_current.v1())){ //connected to site i
            f_factors.push_back(array3d<double>(current.w().nx(),r_k,r_k));
        }
        else{ //connected to site j
            f_factors.push_back(array3d<double>(current.w().ny(),r_k,r_k));
        }
    }
    for(size_t k=0;k<r_k;k++){
        for(size_t k2=0;k2<r_k;k2++){
            std::vector<std::vector<double> > fi_prime_factors;
            std::vector<std::vector<double> > fj_prime_factors;
            for(size_t i=0;i<current.w().nx();i++){
                fi_prime_factors.push_back(std::vector<double>());
            }
            for(size_t j=0;j<current.w().ny();j++){
                fj_prime_factors.push_back(std::vector<double>());
            }
            for(size_t n=0;n<cluster.size();n++){
                if((old_cluster[n].v1()==old_current.v1())||(old_cluster[n].v2()==old_current.v1())){ //connected to site i
                    for(size_t i=0;i<old_current.w().nx();i++){
                        std::vector<double> sum_addends;
                        for(size_t imu=0;imu<cluster[n].w().nx();imu++){
                            double num=cluster[n].w().at(imu,k2,0);
                            double denom;
                            if(old_cluster[n].v1()==current.v1()){ //site imu > site i
                                denom=old_cluster[n].w().at(i,imu,0);
                            }
                            else{ //site imu < site i
                                denom=old_cluster[n].w().at(imu,i,0);
                            }
                            sum_addends.push_back(pow(cluster[n].w().at(imu,k,0),1/(rho-1))*num/denom);
                        }
                        double sum=vec_add_float(sum_addends);
                        fi_prime_factors[i].push_back(sum);
                        f_factors[n].at(i,k,k2)=sum;
                    }
                }
                else{ //connected to site j
                    for(size_t j=0;j<old_current.w().ny();j++){
                        std::vector<double> sum_addends;
                        for(size_t imu=0;imu<cluster[n].w().nx();imu++){
                            double num=cluster[n].w().at(imu,k2,0);
                            double denom;
                            if(old_cluster[n].v1()==current.v2()){ //site imu > site j
                                denom=old_cluster[n].w().at(j,imu,0);
                            }
                            else{ //site imu < site j
                                denom=old_cluster[n].w().at(imu,j,0);
                            }
                            sum_addends.push_back(pow(cluster[n].w().at(imu,k,0),1/(rho-1))*num/denom);
                        }
                        double sum=vec_add_float(sum_addends);
                        fj_prime_factors[j].push_back(sum);
                        f_factors[n].at(j,k,k2)=sum;
                    }
                }
            }
            for(size_t i=0;i<current.w().nx();i++){
                fi_prime.at(i,k,k2)=vec_mult_float(fi_prime_factors[i]);
            }
            for(size_t j=0;j<current.w().ny();j++){
                fj_prime.at(j,k,k2)=vec_mult_float(fj_prime_factors[j]);
            }
        }
    }
    //calculate P'_{ki_\mu} and Z' (as sum_p_prime_env)
    std::vector<double> sum_p_prime_addends;
    for(size_t i=0;i<current.w().nx();i++){
        for(size_t j=0;j<current.w().ny();j++){
            for(size_t k=0;k<current.w().nz();k++){
                std::vector<double> factors_env;
                factors_env.push_back(gi_prime[k]);
                factors_env.push_back(gj_prime[k]);
                factors_env.push_back(current.w().at(i,j,k));
                sum_p_prime_addends.push_back(vec_mult_float(factors_env));
            }
        }
    }
    double z_prime=vec_add_float(sum_p_prime_addends);
    
    //calculate intermediate factors (and cost function in the process)
    std::vector<double> sum_factors_addends;
    for(size_t i=0;i<current.w().nx();i++){
        for(size_t j=0;j<current.w().ny();j++){
            for(size_t k=0;k<current.w().nz();k++){
                std::vector<double> k2_sum_addends;
                for(size_t k2=0;k2<current.w().nz();k2++){
                    k2_sum_addends.push_back(current.w().at(i,j,k2)/old_current.w().at(i,j,0)*fi_prime.at(i,k,k2)*fj_prime.at(j,k,k2));
                }
                sum_factors_addends.push_back(current.w().at(i,j,k)*pow(vec_add_float(k2_sum_addends),rho-1));
            }
        }
    }
    double sum_factors=vec_add_float(sum_factors_addends);
    cost=log(sum_factors*pow(z,rho-1)/pow(z_prime,rho));
    current=current_bk;
    return cost;
}