#include <fstream>

#include "optimize.hpp"
#include "observables.hpp"

//DEBUG GLOBALS
/* size_t p_prime_ij_opt_count_idx=0;
size_t p_prime_ki_opt_count_idx=0;
std::ofstream ij_ofs("pair_ij_bmi_dump.txt");
std::ofstream ki_ofs("pair_ki_bmi_dump.txt");
std::ofstream ij_cost_ofs("pair_ij_cost_dump.txt");
std::ofstream ki_cost_ofs("pair_ki_cost_dump.txt");
 */
double vec_add_float(std::vector<double> v){
    std::sort(v.begin(),v.end());
    double sum=0;
    for(size_t i=0;i<v.size();i++){
        sum+=v[i];
    }
    return sum;
}

double vec_mult_float(std::vector<double> v){
    std::sort(v.begin(),v.end());
    double prod=1;
    for(size_t i=0;i<v.size();i++){
        prod*=v[i];
    }
    return prod;
}

// void optimize::potts_renorm(size_t slave,std::vector<bond>& old_cluster,bond& current,std::vector<bond>& cluster){
    // size_t q=current.w().nx(); //assumption is that rank(=q) is constant!
    // double current_j=-log(((1/current.w().at(0,0))-current.w().nx())/(double) (current.w().nx()*(current.w().nx()-1)));
    // for(size_t n=0;n<cluster.size();n++){
        // if((old_cluster[n].v1()==slave)||(old_cluster[n].v2()==slave)){
            // double cluster_n_j=-log(((1/cluster[n].w().at(0,0))-cluster[n].w().nx())/(double) (cluster[n].w().nx()*(cluster[n].w().nx()-1)));
            // cluster_n_j=renorm_coupling(q,cluster_n_j,current_j); //update bond weight
            // for(size_t i=0;i<q;i++){
                // for(size_t j=0;j<q;j++){
                    // cluster[n].w().at(i,j)=(i==j)?(1/(q+(q*(q-1)*exp(-cluster_n_j)))):(1/((q*exp(cluster_n_j))+(q*(q-1))));
                // }
            // }
            // cluster[n].bmi(cluster[n].w());
        // }
    // }
// }

//debug ver
//TODO: add convergence criteria for both iterative eq and grad desc methods
//TODO: move weight matrix reinit to dedicated function in ndarray (like expand_dims)
void optimize::renyi_opt(size_t master,size_t slave,size_t r_k,std::vector<site> sites,bond& old_current,std::vector<bond>& old_cluster,bond& current,std::vector<bond>& cluster,size_t max_it,double lr=0){
    /* std::vector<std::string> ij_bmi_dump_lines;
    std::vector<std::string> ij_cost_dump_lines;
    std::vector<std::string> ki_bmi_dump_lines;
    std::vector<std::string> ki_cost_dump_lines;
    
    //DEBUG OUTPUT
    {
        for(size_t n=0;n<cluster.size();n++){
            std::stringstream ki_bmi_dump_line;
            std::stringstream ki_cost_dump_line;
            ki_bmi_dump_line<<(p_prime_ki_opt_count_idx+n)<<" "<<cluster[n].v1()<<" "<<cluster[n].v2()<<" "<<0<<" "<<cluster[n].bmi()<<"\n";
            // std::cout<<ki_eff_k_dump_lines.size()<<"\n";
            ki_bmi_dump_lines.push_back(ki_bmi_dump_line.str());
            //DEBUG: calculate cost function
            double cost=optimize::kl_div(sites,old_current,old_cluster,current,cluster);
            ki_cost_dump_line<<(p_prime_ki_opt_count_idx+n)<<" "<<cluster[n].v1()<<" "<<cluster[n].v2()<<" "<<0<<" "<<cost<<"\n";
            ki_cost_dump_lines.push_back(ki_cost_dump_line.str());
        }
        std::stringstream ij_bmi_dump_line;
        std::stringstream ij_cost_dump_line;
        ij_bmi_dump_line<<p_prime_ij_opt_count_idx<<" "<<current.v1()<<" "<<current.v2()<<" "<<0<<" "<<current.bmi()<<"\n";
        ij_bmi_dump_lines.push_back(ij_bmi_dump_line.str());
        //DEBUG: calculate cost function
        double cost=optimize::kl_div(sites,old_current,old_cluster,current,cluster);
        ij_cost_dump_line<<p_prime_ij_opt_count_idx<<" "<<current.v1()<<" "<<current.v2()<<" "<<0<<" "<<cost<<"\n";
        ij_cost_dump_lines.push_back(ij_cost_dump_line.str());
    } */
    //reinitialize weight matrix to correct size if needed on first iteration
    if(current.w().nz()!=r_k){
        array3d<double> new_w(current.w().nx(),current.w().ny(),r_k);
        for(size_t i=0;i<new_w.nx();i++){
            for(size_t j=0;j<new_w.ny();j++){
                for(size_t k=0;k<new_w.nz();k++){
                    if(k<current.w().nz()){
                        // new_w.at(i,j,k)=current.w().at(i,j,k)*(current.w().nx()*current.w().ny())/(double) (new_w.nx()*new_w.ny()*new_w.nz());
                        new_w.at(i,j,k)=1/(double) (new_w.nx()*new_w.ny()*new_w.nz());
                        // new_w.at(i,j,k)=current.w().at(i,j,0);
                        // new_w.at(i,j,k)=current.w().at(i,j,k);
                    }
                    else{
                        new_w.at(i,j,k)=1/(double) (new_w.nx()*new_w.ny()*new_w.nz());
                        // new_w.at(i,j,k)=1/(double) (new_w.nx()*new_w.ny());
                        // new_w.at(i,j,k)=current.w().at(i,j,0);
                        // new_w.at(i,j,k)=current.w().at(i,j,0)/r_k;
                        // new_w.at(i,j,k)=0;
                    }
                }
            }
        }
        current.w()=new_w;
        current.bmi(current.w());
        // std::cout<<(std::string) new_w<<"\n";
    }
    for(size_t n=0;n<cluster.size();n++){
        // std::cout<<"old_cluster: "<<(std::string)old_cluster[n]<<"\n";
        // std::cout<<"cluster: "<<(std::string)cluster[n]<<"\n";
        if((cluster[n].w().nx()!=sites[cluster[n].v1()].rank())||(cluster[n].w().ny()!=r_k)){
            array3d<double> new_w(sites[cluster[n].v1()].rank(),r_k,1);
            for(size_t i=0;i<new_w.nx();i++){
                for(size_t j=0;j<new_w.ny();j++){
                    if((i<cluster[n].w().nx())&&(j<cluster[n].w().ny())){
                        new_w.at(i,j,0)=cluster[n].w().at(i,j,0)*(cluster[n].w().nx()*cluster[n].w().ny())/(double) (new_w.nx()*new_w.ny());
                        // new_w.at(i,j,0)=cluster[n].w().at(i,j,0);
                    }
                    else{
                        new_w.at(i,j,0)=1/(double) (new_w.nx()*new_w.ny());
                        // new_w.at(i,j,0)=0;
                    }
                }
            }
            cluster[n].w()=new_w;
            cluster[n].bmi(cluster[n].w());
            // std::cout<<(std::string) new_w<<"\n";
        }
        // std::cout<<"new cluster: "<<(std::string)cluster[n]<<"\n";
    }
    
    //convergence check variables
    array3d<double> prev_current=current.w();
    std::vector<array3d<double> > prev_cluster;
    for(size_t n=0;n<cluster.size();n++){
        prev_cluster.push_back(cluster[n].w());
    }
    
    //precompute intermediate quantities for weight optimization
    //calculate G_i(S_i) and G_j(S_j)
    std::vector<double> gi(old_current.w().nx(),1);
    std::vector<double> gj(old_current.w().ny(),1);
    array3d<double> p_prime_ijk_env(current.w().nx(),current.w().ny(),r_k);
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
    //calculate Z
    double z=0;
    for(size_t i=0;i<old_current.w().nx();i++){
        for(size_t j=0;j<old_current.w().ny();j++){
            z+=old_current.w().at(i,j,0)*gi[i]*gj[j];
        }
    }
    // std::cout<<"gi:";
    // for(size_t i=0;i<gi.size();i++){
        // std::cout<<gi[i]<<" ";
    // }
    // std::cout<<"\n";
    // std::cout<<"gj:";
    // for(size_t j=0;j<gj.size();j++){
        // std::cout<<gj[j]<<" ";
    // }
    // std::cout<<"\n";
    // std::cout<<"z:"<<z<<"\n";
    
    for(size_t n1=0;n1<max_it;n1++){
        // std::stringstream ij_bmi_dump_line;
        // std::stringstream ij_cost_dump_line;
        // std::cout<<"old_current.w():\n"<<(std::string)old_current.w()<<"\n";
        //calculate G'_i(S_k) and G'_j(S_k)
        std::vector<double> gi_prime(r_k,1);
        std::vector<double> gj_prime(r_k,1);
        for(size_t k=0;k<r_k;k++){
            std::vector<double> gi_prime_factors;
            std::vector<double> gj_prime_factors;
            for(size_t m=0;m<cluster.size();m++){
                // std::cout<<(std::string)cluster[m].w()<<"\n";
                if((old_cluster[m].v1()==old_current.v1())||(old_cluster[m].v2()==old_current.v1())){ //connected to site i
                    if((cluster[m].w().sum_over_axis(0,2))[k]>1e-10){ //avoid 0 factors
                        gi_prime_factors.push_back((cluster[m].w().sum_over_axis(0,2))[k]);
                    }
                }
                else{ //connected to site j
                    if((cluster[m].w().sum_over_axis(0,2))[k]>1e-10){ //avoid 0 factors
                        gj_prime_factors.push_back((cluster[m].w().sum_over_axis(0,2))[k]);
                    }
                }
            }
            gi_prime[k]=vec_mult_float(gi_prime_factors);
            gj_prime[k]=vec_mult_float(gj_prime_factors);
        }
        // std::cout<<"gi_prime:";
        // for(size_t i=0;i<gi_prime.size();i++){
            // std::cout<<gi_prime[i]<<" ";
        // }
        // std::cout<<"\n";
        // std::cout<<"gj_prime:";
        // for(size_t j=0;j<gj_prime.size();j++){
            // std::cout<<gj_prime[j]<<" ";
        // }
        // std::cout<<"\n";
        //calculate F'_i(S_i,S_k,S_k',a) and F'_j(S_j,S_k,S_k',a)
        array3d<double> fi_prime(current.w().nx(),r_k,r_k);
        array3d<double> fj_prime(current.w().ny(),r_k,r_k);
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
                for(size_t m=0;m<cluster.size();m++){
                    if((old_cluster[m].v1()==old_current.v1())||(old_cluster[m].v2()==old_current.v1())){ //connected to site i
                        for(size_t i=0;i<old_current.w().nx();i++){
                            double sum=0;
                            for(size_t imu=0;imu<cluster[m].w().nx();imu++){
                                double num=cluster[m].w().at(imu,k,0)*cluster[m].w().at(imu,k2,0);
                                double denom;
                                if(old_cluster[m].v1()==current.v1()){ //site imu > site i
                                    denom=old_cluster[m].w().at(i,imu,0);
                                }
                                else{ //site imu < site i
                                    denom=old_cluster[m].w().at(imu,i,0);
                                }
                                sum+=num/denom;
                            }
                            fi_prime_factors[i].push_back(sum);
                        }
                    }
                    else{ //connected to site j
                        for(size_t j=0;j<old_current.w().ny();j++){
                            double sum=0;
                            for(size_t imu=0;imu<cluster[m].w().nx();imu++){
                                double num=cluster[m].w().at(imu,k,0)*cluster[m].w().at(imu,k2,0);
                                double denom;
                                if(old_cluster[m].v1()==current.v2()){ //site imu > site j
                                    denom=old_cluster[m].w().at(j,imu,0);
                                }
                                else{ //site imu < site j
                                    denom=old_cluster[m].w().at(imu,j,0);
                                }
                                sum+=num/denom;
                            }
                            fj_prime_factors[j].push_back(sum);
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
        // std::cout<<"fi_prime:"<<(std::string)fi_prime<<"\n";
        // std::cout<<"fj_prime:"<<(std::string)fj_prime<<"\n";
        //calculate P'^{env}_{ijk} and Z' (as sum_p_prime_ijk_env)
        std::vector<double> sum_p_prime_ijk_env_addends;
        for(size_t i=0;i<p_prime_ijk_env.nx();i++){
            for(size_t j=0;j<p_prime_ijk_env.ny();j++){
                for(size_t k=0;k<p_prime_ijk_env.nz();k++){
                    std::vector<double> factors_env;
                    factors_env.push_back(gi_prime[k]);
                    factors_env.push_back(gj_prime[k]);
                    p_prime_ijk_env.at(i,j,k)=vec_mult_float(factors_env);
                    sum_p_prime_ijk_env_addends.push_back(p_prime_ijk_env.at(i,j,k)*current.w().at(i,j,k)); //restore divided factor to get z'
                }
            }
        }
        //normalize P'^{env}_{ij} by Z' (as sum_p_prime_ijk_env)
        double sum_p_prime_ijk_env=vec_add_float(sum_p_prime_ijk_env_addends);
        for(size_t i=0;i<p_prime_ijk_env.nx();i++){
            for(size_t j=0;j<p_prime_ijk_env.ny();j++){
                for(size_t k=0;k<p_prime_ijk_env.nz();k++){
                    // if(fabs(sum_p_prime_ij_env)>1e-10){
                        p_prime_ijk_env.at(i,j,k)/=sum_p_prime_ijk_env;
                    // }
                }
            }
        }
        // std::cout<<"current:"<<(std::string)current.w()<<"\n";
        // std::cout<<"p_prime_ijk_env:"<<(std::string)p_prime_ijk_env<<"\n";
        // std::cout<<"z':"<<sum_p_prime_ijk_env<<"\n";
        //calculate intermediate factors (and cost function as sum_factors in the process)
        array3d<double> ij_factors(p_prime_ijk_env.nx(),p_prime_ijk_env.ny(),p_prime_ijk_env.nz());
        double sum_ij_factors=0;
        for(size_t i=0;i<p_prime_ijk_env.nx();i++){
            for(size_t j=0;j<p_prime_ijk_env.ny();j++){
                for(size_t k=0;k<p_prime_ijk_env.nz();k++){
                    std::vector<double> k2_sum_addends;
                    for(size_t k2=0;k2<p_prime_ijk_env.nz();k2++){
                        k2_sum_addends.push_back(current.w().at(i,j,k2)*fi_prime.at(i,k,k2)*fj_prime.at(j,k,k2));
                    }
                    ij_factors.at(i,j,k)=(current.w().at(i,j,k)/old_current.w().at(i,j,0))*vec_add_float(k2_sum_addends);
                    ij_factors.at(i,j,k)*=(z/sum_p_prime_ijk_env)/sum_p_prime_ijk_env; //cancelled out in later division but needed in gradient descent
                    sum_ij_factors+=ij_factors.at(i,j,k);
                }
            }
        }
        // std::cout<<"ij_factors:"<<(std::string)ij_factors<<"\n";
        // std::cout<<"sum_ij_factors:"<<sum_ij_factors<<"\n";
        
        double sum=0;
        for(size_t i=0;i<p_prime_ijk_env.nx();i++){
            for(size_t j=0;j<p_prime_ijk_env.ny();j++){
                for(size_t k=0;k<p_prime_ijk_env.nz();k++){
                    if(lr==0){ //lr==0 means use iterative method based on stationarity condition
                        if(fabs(p_prime_ijk_env.at(i,j,k))>1e-10){
                            current.w().at(i,j,k)=ij_factors.at(i,j,k)/(sum_ij_factors*p_prime_ijk_env.at(i,j,k));
                            if(current.w().at(i,j,k)<1e-10){
                                current.w().at(i,j,k)=1e-10;
                            }
                        }
                    }
                    else{ //lr!=0 means use gradient descent with lr
                        if(fabs(current.w().at(i,j,k))>1e-10){
                            current.w().at(i,j,k)-=lr*2*((ij_factors.at(i,j,k)/current.w().at(i,j,k))-(p_prime_ijk_env.at(i,j,k)*sum_ij_factors));
                            if(current.w().at(i,j,k)<1e-10){ //in case the weight is negative, force it to be nonnegative!
                                current.w().at(i,j,k)=1e-10;
                            }
                            sum+=current.w().at(i,j,k);
                        }
                    }
                }
            }
        }
        if(lr!=0){ //lr==0 means use iterative method based on stationarity condition
            for(size_t i=0;i<p_prime_ijk_env.nx();i++){
                for(size_t j=0;j<p_prime_ijk_env.ny();j++){
                    for(size_t k=0;k<p_prime_ijk_env.nz();k++){
                        current.w().at(i,j,k)/=sum;
                    }
                }
            }
        }
        
        current.bmi(current.w());
        // std::cout<<"updated current:"<<(std::string)current.w()<<"\n";
        
        //NOTE: right now, this uses a workaround for floating-point non-associativity, which involves storing addends in vectors and then sorting them before adding.
        //p_prime_ki,imu if source==current.v1(), p_prime_kj,jnu if source==current.v2(). wlog, use p_prime_ki and imu.
        //k, the new index, is always second because virtual indices > physical indices.
        for(size_t n=0;n<cluster.size();n++){
            // std::cout<<"old cluster[n]: "<<(std::string) old_cluster[n].w()<<"\n";
            //determine whether this bond was connected to site i or to site j
            size_t source;
            if(old_cluster[n].v1()==master||old_cluster[n].v1()==slave){
                source=(old_cluster[n].v1()==master)?master:slave;
            }
            else{
                source=(old_cluster[n].v2()==master)?master:slave;
            }
        
            // std::stringstream ki_bmi_dump_line;
            // std::stringstream ki_cost_dump_line;
            array3d<double> p_prime_ki_env(cluster[n].w().nx(),cluster[n].w().ny(),1);
            cluster[n].bmi(cluster[n].w());
            
            //calculate G'_i(S_k) and G'_j(S_k)
            std::vector<double> gi_prime(r_k,1);
            std::vector<double> gj_prime(r_k,1);
            for(size_t k=0;k<r_k;k++){
                std::vector<double> gi_prime_factors;
                std::vector<double> gj_prime_factors;
                for(size_t m=0;m<cluster.size();m++){
                    if((old_cluster[m].v1()==old_current.v1())||(old_cluster[m].v2()==old_current.v1())){ //connected to site i
                        if((cluster[m].w().sum_over_axis(0,2))[k]>1e-10){ //avoid 0 factors
                            gi_prime_factors.push_back((cluster[m].w().sum_over_axis(0,2))[k]);
                        }
                    }
                    else{ //connected to site j
                        if((cluster[m].w().sum_over_axis(0,2))[k]>1e-10){ //avoid 0 factors
                            gj_prime_factors.push_back((cluster[m].w().sum_over_axis(0,2))[k]);
                        }
                    }
                }
                gi_prime[k]=vec_mult_float(gi_prime_factors);
                gj_prime[k]=vec_mult_float(gj_prime_factors);
            }
            // std::cout<<"gi_prime:";
            // for(size_t i=0;i<gi_prime.size();i++){
                // std::cout<<gi_prime[i]<<" ";
            // }
            // std::cout<<"\n";
            // std::cout<<"gj_prime:";
            // for(size_t j=0;j<gj_prime.size();j++){
                // std::cout<<gj_prime[j]<<" ";
            // }
            // std::cout<<"\n";
            //calculate F'_i(S_i,S_k,S_k',a) and F'_j(S_j,S_k,S_k',a) and keep F factors
            std::vector<array3d<double> > f_factors;
            array3d<double> fi_prime(current.w().nx(),r_k,r_k);
            array3d<double> fj_prime(current.w().ny(),r_k,r_k);
            for(size_t m=0;m<cluster.size();m++){
                if((old_cluster[m].v1()==old_current.v1())||(old_cluster[m].v2()==old_current.v1())){ //connected to site i
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
                    for(size_t m=0;m<cluster.size();m++){
                        if((old_cluster[m].v1()==old_current.v1())||(old_cluster[m].v2()==old_current.v1())){ //connected to site i
                            for(size_t i=0;i<old_current.w().nx();i++){
                                double sum=0;
                                for(size_t imu=0;imu<cluster[m].w().nx();imu++){
                                    double num=cluster[m].w().at(imu,k,0)*cluster[m].w().at(imu,k2,0);
                                    double denom;
                                    if(old_cluster[m].v1()==current.v1()){ //site imu > site i
                                        denom=old_cluster[m].w().at(i,imu,0);
                                    }
                                    else{ //site imu < site i
                                        denom=old_cluster[m].w().at(imu,i,0);
                                    }
                                    sum+=num/denom;
                                }
                                fi_prime_factors[i].push_back(sum);
                                f_factors[m].at(i,k,k2)=sum;
                            }
                        }
                        else{ //connected to site j
                            for(size_t j=0;j<old_current.w().ny();j++){
                                double sum=0;
                                for(size_t imu=0;imu<cluster[m].w().nx();imu++){
                                    double num=cluster[m].w().at(imu,k,0)*cluster[m].w().at(imu,k2,0);
                                    double denom;
                                    if(old_cluster[m].v1()==current.v2()){ //site imu > site j
                                        denom=old_cluster[m].w().at(j,imu,0);
                                    }
                                    else{ //site imu < site j
                                        denom=old_cluster[m].w().at(imu,j,0);
                                    }
                                    sum+=num/denom;
                                }
                                fj_prime_factors[j].push_back(sum);
                                f_factors[m].at(j,k,k2)=sum;
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
            // std::cout<<"fi_prime:"<<(std::string)fi_prime<<"\n";
            // std::cout<<"fj_prime:"<<(std::string)fj_prime<<"\n";
            //calculate P'^{env}_{ki_\mu} and Z' (as sum_p_prime_ki_env)
            std::vector<std::vector<double> > addends_env(p_prime_ki_env.nx()*p_prime_ki_env.ny());
            std::vector<double> sum_p_prime_ki_env_addends;
            for(size_t i=0;i<current.w().nx();i++){
                for(size_t j=0;j<current.w().ny();j++){
                    for(size_t k=0;k<p_prime_ki_env.ny();k++){
                        for(size_t imu=0;imu<p_prime_ki_env.nx();imu++){
                            std::vector<double> factors_env;
                            factors_env.push_back(current.w().at(i,j,k));
                            factors_env.push_back(gi_prime[k]);
                            factors_env.push_back(gj_prime[k]);
                            factors_env.push_back(1/(cluster[n].w().sum_over_axis(0,2))[k]); //divide appropriate factor
                            addends_env[(k*p_prime_ki_env.nx())+imu].push_back(vec_mult_float(factors_env));
                            sum_p_prime_ki_env_addends.push_back(vec_mult_float(factors_env)*cluster[n].w().at(imu,k,0)); //restore divided factor to get z'
                        }
                    }
                }
            }
            for(size_t imu=0;imu<p_prime_ki_env.nx();imu++){
                for(size_t k=0;k<p_prime_ki_env.ny();k++){
                    p_prime_ki_env.at(imu,k,0)=vec_add_float(addends_env[(k*p_prime_ki_env.nx())+imu]);
                }
            }
            double sum_p_prime_ki_env=vec_add_float(sum_p_prime_ki_env_addends);
            // std::cout<<"cluster[n]: "<<(std::string) cluster[n].w()<<"\n";
            // std::cout<<"p_prime_ki_env: "<<(std::string) p_prime_ki_env<<"\n";
            //normalize P'^{env}_{ki_\mu} by Z'
            for(size_t imu=0;imu<p_prime_ki_env.nx();imu++){
                for(size_t k=0;k<p_prime_ki_env.ny();k++){
                    // if(fabs(sum_p_prime_ki_env)>1e-10){
                        p_prime_ki_env.at(imu,k,0)/=sum_p_prime_ki_env;
                    // }
                }
            }
            //calculate intermediate factors (and cost function as sum_ki_factors in the process)
            array2d<double> ki_factors(p_prime_ki_env.nx(),p_prime_ki_env.ny());
            double sum_ki_factors=0;
            for(size_t k=0;k<p_prime_ki_env.ny();k++){
                for(size_t imu=0;imu<p_prime_ki_env.nx();imu++){
                    std::vector<double> i2j2k2_sum_addends;
                    for(size_t i2=0;i2<current.w().nx();i2++){
                        for(size_t j2=0;j2<current.w().ny();j2++){
                            for(size_t k2=0;k2<p_prime_ki_env.ny();k2++){
                                double res=(current.w().at(i2,j2,k2)*current.w().at(i2,j2,k)/old_current.w().at(i2,j2,0))*cluster[n].w().at(imu,k2,0);
                                double restored_f_factor;
                                if(source==current.v1()){ //connected to site i
                                    restored_f_factor=f_factors[n].at(i2,k,k2);
                                    if(old_cluster[n].v1()==current.v1()){ //site imu > site i
                                        res/=old_cluster[n].w().at(i2,imu,0);
                                    }
                                    else{ //site imu < site i
                                        res/=old_cluster[n].w().at(imu,i2,0);
                                    }
                                }
                                else{ //connected to site j
                                    restored_f_factor=f_factors[n].at(j2,k,k2);
                                    if(old_cluster[n].v1()==current.v2()){ //site imu > site j
                                        res/=old_cluster[n].w().at(j2,imu,0);
                                    }
                                    else{ //site imu < site j
                                        res/=old_cluster[n].w().at(imu,j2,0);
                                    }
                                }
                                // std::cout<<pow(old_current.w().at(i,j,0),1-(double) a)<<" "<<pow(current.w().at(i,j,k),(double) a)<<" "<<res<<" "<<fi_prime.at(i,k)<<" "<<fj_prime.at(j,k)<<"\n";
                                res*=fi_prime.at(i2,k,k2)*fj_prime.at(j2,k,k2)/restored_f_factor;
                                i2j2k2_sum_addends.push_back(res);
                            }
                        }
                    }
                    ki_factors.at(imu,k)=cluster[n].w().at(imu,k,0)*vec_add_float(i2j2k2_sum_addends);
                    ki_factors.at(imu,k)*=(z/sum_p_prime_ki_env)/sum_p_prime_ki_env; //cancelled out in later division
                    sum_ki_factors+=ki_factors.at(imu,k);
                }
            }
            
            double sum=0;
            for(size_t imu=0;imu<p_prime_ki_env.nx();imu++){
                for(size_t k=0;k<p_prime_ki_env.ny();k++){
                    if(lr==0){ //lr==0 means use iterative method based on stationarity condition
                        if(fabs(p_prime_ki_env.at(imu,k,0))>1e-10){
                            cluster[n].w().at(imu,k,0)=ki_factors.at(imu,k)/(sum_ki_factors*p_prime_ki_env.at(imu,k,0));
                            if(cluster[n].w().at(imu,k,0)<1e-10){
                                cluster[n].w().at(imu,k,0)=1e-10;
                            }
                        }
                    }
                    else{ //lr!=0 means use gradient descent with lr
                        if(fabs(cluster[n].w().at(imu,k,0))>1e-10){
                            cluster[n].w().at(imu,k,0)-=lr*2*((ki_factors.at(imu,k)/cluster[n].w().at(imu,k,0))-(p_prime_ki_env.at(imu,k,0)*sum_ki_factors));
                            sum+=cluster[n].w().at(imu,k,0);
                            if(cluster[n].w().at(imu,k,0)<1e-10){ //in case the weight is negative, force it to be nonnegative!
                                cluster[n].w().at(imu,k,0)=1e-10;
                            }
                        }
                    }
                }
            }
            if(lr!=0){ //lr==0 means use iterative method based on stationarity condition
                for(size_t i=0;i<p_prime_ki_env.nx();i++){
                    for(size_t j=0;j<p_prime_ki_env.ny();j++){
                        cluster[n].w().at(i,j,0)/=sum;
                    }
                }
            }
            // std::cout<<"updated cluster[n]: "<<(std::string) cluster[n].w()<<"\n";
            cluster[n].bmi(cluster[n].w());
            /* //DEBUG OUTPUT
            ki_bmi_dump_line<<(p_prime_ki_opt_count_idx+n)<<" "<<cluster[n].v1()<<" "<<cluster[n].v2()<<" "<<(n1+1)<<" "<<cluster[n].bmi()<<"\n";
            // std::cout<<ki_eff_k_dump_lines.size()<<"\n";
            ki_bmi_dump_lines.push_back(ki_bmi_dump_line.str());
            //DEBUG: calculate cost function
            double cost=optimize::kl_div(sites,old_current,old_cluster,current,cluster);
            ki_cost_dump_line<<(p_prime_ki_opt_count_idx+n)<<" "<<cluster[n].v1()<<" "<<cluster[n].v2()<<" "<<(n1+1)<<" "<<cost<<"\n";
            ki_cost_dump_lines.push_back(ki_cost_dump_line.str()); */
        }
        /* //DEBUG OUTPUT
        ij_bmi_dump_line<<p_prime_ij_opt_count_idx<<" "<<current.v1()<<" "<<current.v2()<<" "<<(n1+1)<<" "<<current.bmi()<<"\n";
        ij_bmi_dump_lines.push_back(ij_bmi_dump_line.str());
        //DEBUG: calculate cost function
        double cost=optimize::kl_div(sites,old_current,old_cluster,current,cluster);
        ij_cost_dump_line<<p_prime_ij_opt_count_idx<<" "<<current.v1()<<" "<<current.v2()<<" "<<(n1+1)<<" "<<cost<<"\n";
        ij_cost_dump_lines.push_back(ij_cost_dump_line.str()); */
        
        //check for convergence if after first iteration (since current must be resized)
        if(n1>0){
            double diff=0;
            for(size_t i=0;i<current.w().nx();i++){
                for(size_t j=0;j<current.w().ny();j++){
                    for(size_t k=0;k<current.w().nz();k++){
                        diff+=fabs(current.w().at(i,j,k)-prev_current.at(i,j,k));
                    }
                }
            }
            for(size_t n=0;n<cluster.size();n++){
                for(size_t i=0;i<cluster[n].w().nx();i++){
                    for(size_t j=0;j<cluster[n].w().ny();j++){
                        diff+=fabs(cluster[n].w().at(i,j,0)-prev_cluster[n].at(i,j,0));
                    }
                }
            }
            if(diff<1e-10){
                std::cout<<"converged after "<<(n1+1)<<" iterations\n";
                break;
            }
        }
        
        //update convergence check variables
        prev_current=current.w();
        for(size_t n=0;n<cluster.size();n++){
            prev_cluster[n]=cluster[n].w();
        }
    }
    /* p_prime_ij_opt_count_idx++;
    for(size_t i=0;i<ij_bmi_dump_lines.size();i++){
        ij_ofs<<ij_bmi_dump_lines[i];
        ij_cost_ofs<<ij_cost_dump_lines[i];
    }
    p_prime_ki_opt_count_idx+=cluster.size();
    for(size_t i=0;i<ki_bmi_dump_lines.size();i++){
        ki_ofs<<ki_bmi_dump_lines[i];
        ki_cost_ofs<<ki_cost_dump_lines[i];
    } */
    // std::cout<<(std::string) current.w()<<"\n";
}

double optimize::renorm_coupling(size_t q,double k1,double k2){
    double num=exp(k1+k2)+(q-1);
    double denom=exp(k1)+exp(k2)+(q-2);
    return log(num/denom);
}

/*
double optimize::kl_div(std::vector<site> sites,bond old_current,std::vector<bond> old_cluster,bond current,std::vector<bond> cluster){
    // std::cout<<(std::string)current<<"\n";
    // std::cout<<(std::string)current.w()<<"\n";
    
    std::vector<double> g_prime(current.w().nz());
    for(size_t k=0;k<g_prime.size();k++){
        double g_prime_res=1;
        for(size_t n=0;n<cluster.size();n++){
            double spin_sum_prime=0;
            for(size_t imu=0;imu<cluster[n].w().nx();imu++){
                spin_sum_prime+=cluster[n].w().at(imu,k,0); //bond still as weight matrix
            }
            g_prime_res*=spin_sum_prime;
        }
        g_prime[k]=g_prime_res;
    }
    
    std::vector<double> gi(sites[old_current.v1()].rank());
    std::vector<double> gj(sites[old_current.v2()].rank());
    for(size_t i=0;i<sites[current.v1()].rank();i++){
        double gi_res=1;
        for(size_t n=0;n<old_cluster.size();n++){
            double spin_sum_i=0;
            size_t source;
            if(old_cluster[n].v1()==old_current.v1()||old_cluster[n].v1()==old_current.v2()){
                source=(old_cluster[n].v1()==old_current.v1())?old_current.v1():old_current.v2();
            }
            else{
                source=(old_cluster[n].v2()==old_current.v1())?old_current.v1():old_current.v2();
            }
            if(source==old_current.v1()){ //connected to site i
                if(old_cluster[n].v1()==old_current.v2()){ //site imu > site i
                    for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                        spin_sum_i+=old_cluster[n].w().at(i,imu,0); //bond still as weight matrix
                    }
                }
                else{ //site imu < site i
                    for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                        spin_sum_i+=old_cluster[n].w().at(imu,i,0); //bond still as weight matrix
                    }
                }
                gi_res*=spin_sum_i;
            }
        }
        gi[i]=gi_res;
    }
    for(size_t j=0;j<sites[current.v2()].rank();j++){
        double gj_res=1;
        for(size_t n=0;n<old_cluster.size();n++){
            double spin_sum_j=0;
            size_t source;
            if(old_cluster[n].v1()==old_current.v1()||old_cluster[n].v1()==old_current.v2()){
                source=(old_cluster[n].v1()==old_current.v1())?old_current.v1():old_current.v2();
            }
            else{
                source=(old_cluster[n].v2()==old_current.v1())?old_current.v1():old_current.v2();
            }
            if(source==old_current.v2()){ //connected to site j
                if(old_cluster[n].v1()==old_current.v2()){ //site imu > site j
                    for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                        spin_sum_j+=old_cluster[n].w().at(j,imu,0); //bond still as weight matrix
                    }
                }
                else{ //site imu < site j
                    for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                        spin_sum_j+=old_cluster[n].w().at(imu,j,0); //bond still as weight matrix
                    }
                }
                gj_res*=spin_sum_j;
            }
        }
        gj[j]=gj_res;
    }
    
    array2d<double> gi_tilde(sites[old_current.v1()].rank(),sites[old_current.v2()].rank());
    array2d<double> gj_tilde(sites[old_current.v1()].rank(),sites[old_current.v2()].rank());
    for(size_t i=0;i<sites[current.v1()].rank();i++){
        for(size_t j=0;j<sites[current.v2()].rank();j++){
            for(size_t k=0;k<current.w().nz();k++){
                double gi_tilde_res=0;
                double gj_tilde_res=0;
                for(size_t n=0;n<old_cluster.size();n++){
                    double spin_sum_i_tilde=0;
                    double spin_sum_j_tilde=0;
                    double div_factor=0;
                    size_t source;
                    if(old_cluster[n].v1()==old_current.v1()||old_cluster[n].v1()==old_current.v2()){
                        source=(old_cluster[n].v1()==old_current.v1())?old_current.v1():old_current.v2();
                    }
                    else{
                        source=(old_cluster[n].v2()==old_current.v1())?old_current.v1():old_current.v2();
                    }
                    if(source==old_current.v1()){ //connected to site i
                        if(old_cluster[n].v1()==old_current.v1()){ //site imu > site i
                            for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                                div_factor+=old_cluster[n].w().at(i,imu,0); //bond still as weight matrix
                            }
                            for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                                if(cluster[n].w().at(imu,k,0)>0){ //bond still as weight matrix
                                    spin_sum_i_tilde+=old_cluster[n].w().at(i,imu,0)*log(cluster[n].w().at(imu,k,0)); //bond still as weight matrix
                                }
                            }
                        }
                        else{ //site imu < site i
                            for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                                div_factor+=old_cluster[n].w().at(imu,i,0); //bond still as weight matrix
                            }
                            for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                                if(cluster[n].w().at(imu,k,0)>0){ //bond still as weight matrix
                                    spin_sum_i_tilde+=old_cluster[n].w().at(imu,i,0)*log(cluster[n].w().at(imu,k,0)); //bond still as weight matrix
                                }
                            }
                        }
                        gi_tilde_res+=(div_factor>0)?(spin_sum_i_tilde*gi[i]/div_factor):0;
                    }
                    else{ //connected to site j
                        if(old_cluster[n].v1()==old_current.v2()){ //site imu > site j
                            for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                                div_factor+=old_cluster[n].w().at(j,imu,0); //bond still as weight matrix
                            }
                            for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                                if(cluster[n].w().at(imu,k,0)>0){ //bond still as weight matrix
                                    spin_sum_j_tilde+=old_cluster[n].w().at(j,imu,0)*log(cluster[n].w().at(imu,k,0)); //bond still as weight matrix
                                }
                            }
                        }
                        else{ //site imu < site j
                            for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                                div_factor+=old_cluster[n].w().at(imu,j,0); //bond still as weight matrix
                            }
                            for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                                if(cluster[n].w().at(imu,k,0)>0){ //bond still as weight matrix
                                    spin_sum_j_tilde+=old_cluster[n].w().at(imu,j,0)*log(cluster[n].w().at(imu,k,0)); //bond still as weight matrix
                                }
                            }
                        }
                        gj_tilde_res+=(div_factor>0)?(spin_sum_j_tilde*gj[j]/div_factor):0;
                    }
                }
                gi_tilde.at(i,j)=gi_tilde_res;
                gj_tilde.at(i,j)=gj_tilde_res;
            }
        }
    }
    
    double z_prime=0;
    double z=0;
    double a=0;
    double b=0;
    for(size_t i=0;i<sites[current.v1()].rank();i++){
        for(size_t j=0;j<sites[current.v2()].rank();j++){
            for(size_t k=0;k<current.w().nz();k++){
                double contrib_z_prime=current.w().at(i,j,k)*g_prime_i[i]*g_prime_j[j];
                double contrib_z=old_current.w().at(i,j,0)*gi[i]*gj[j];
                double contrib_a=(current.w().at(i,j,k)>0)?old_current.w().at(i,j)*gi[i]*gj[j]*log(current.w().at(i,j,k)):0; //bond still as weight matrix
                double contrib_b=old_current.w().at(i,j,0)*((gi_tilde.at(i,j)*gj[j])+(gi[i]*gj_tilde.at(i,j)));
                z_prime+=contrib_z_prime;
                z+=contrib_z;
                a+=contrib_a;
                b+=contrib_b;
            }
        }
    }
    // std::cout<<"z: "<<z<<"\nz': "<<z_prime<<"\nz': "<<z_prime2<<"\n";
    // std::cout<<"a: "<<a<<"\nb: "<<b<<"\n";
    double res=log(z_prime)-(a/z)-(b/z);
    return res;
}
*/