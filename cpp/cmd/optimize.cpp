#include <fstream>

#include "../mpi_utils.hpp"
#include "optimize.hpp"
#include "../observables.hpp"

void optimize::opt(size_t master,size_t slave,size_t r_k,std::vector<site> sites,bond& old_current,std::vector<bond>& old_cluster,bond& current,std::vector<bond>& cluster,size_t max_it,double lr=0,size_t max_restarts=10){
    std::uniform_real_distribution<> unif_dist(1e-10,1.0);
    double best_cost=1e300;
    bond best_current=current;
    std::vector<bond> best_cluster;
    for(size_t n=0;n<cluster.size();n++){
        best_cluster.push_back(cluster[n]);
    }
    
    //precompute intermediate quantities for weight optimization
    std::vector<double> gi(old_current.w().nx(),1);
    std::vector<double> gj(old_current.w().ny(),1);
    for(size_t i=0;i<old_current.w().nx();i++){
        std::vector<double> gi_factors;
        for(size_t n=0;n<old_cluster.size();n++){
            if((old_cluster[n].v1()==old_current.v1())||(old_cluster[n].v2()==old_current.v1())){
                gi_factors.push_back((old_cluster[n].w().sum_over_axis((old_cluster[n].v1()==old_current.v1())?1:0))[i]);
            }
        }
        gi[i]=vec_mult_float(gi_factors);
    }
    for(size_t j=0;j<old_current.w().ny();j++){
        std::vector<double> gj_factors;
        for(size_t n=0;n<old_cluster.size();n++){
            if((old_cluster[n].v1()==old_current.v2())||(old_cluster[n].v2()==old_current.v2())){
                gj_factors.push_back((old_cluster[n].w().sum_over_axis((old_cluster[n].v1()==old_current.v2())?1:0))[j]);
            }
        }
        gj[j]=vec_mult_float(gj_factors);
    }
    //calculate Z
    double z=0;
    for(size_t i=0;i<old_current.w().nx();i++){
        for(size_t j=0;j<old_current.w().ny();j++){
            z+=old_current.w().at(i,j)*gi[i]*gj[j];
        }
    }
    
    for(size_t restarts=1;restarts<max_restarts+1;restarts++){
        bond trial_current=current;
        std::vector<bond> trial_cluster;
        for(size_t n=0;n<cluster.size();n++){
            trial_cluster.push_back(cluster[n]);
        }
        //reinitialize weight matrix to correct size if needed on first iteration
        if(lr==0){
            for(size_t n=0;n<trial_cluster.size();n++){
                if((trial_cluster[n].w().nx()!=sites[trial_cluster[n].v1()].rank())||(trial_cluster[n].w().ny()!=r_k)){
                    array2d<double> new_w(sites[trial_cluster[n].v1()].rank(),r_k);
                    double sum=0;
                    for(size_t i=0;i<new_w.nx();i++){
                        for(size_t j=i;j<new_w.ny();j++){
                            new_w.at(i,j)=unif_dist(mpi_utils::prng);
                            sum+=new_w.at(i,j);
                            if(j!=i){
                                new_w.at(j,i)=new_w.at(i,j);
                                sum+=new_w.at(j,i);
                            }
                        }
                    }
                    for(size_t i=0;i<new_w.nx();i++){
                        for(size_t j=0;j<new_w.ny();j++){
                            new_w.at(i,j)/=sum;
                        }
                    }
                    trial_cluster[n].w()=new_w;
                    // std::cout<<(std::string) new_w<<"\n";
                }
            }
        }
        else{
            array2d<double> new_w(trial_current.w().nx(),trial_current.w().ny());
            double sum=0;
            for(size_t i=0;i<new_w.nx();i++){
                for(size_t j=i;j<new_w.ny();j++){
                    new_w.at(i,j)=unif_dist(mpi_utils::prng);
                    sum+=new_w.at(i,j);
                    if(j!=i){
                        new_w.at(j,i)=new_w.at(i,j);
                        sum+=new_w.at(j,i);
                    }
                }
            }
            for(size_t i=0;i<new_w.nx();i++){
                for(size_t j=0;j<new_w.ny();j++){
                    new_w.at(i,j)/=sum;
                }
            }
            trial_current.w()=new_w;
            // std::cout<<(std::string) new_w<<"\n";
            for(size_t n=0;n<trial_cluster.size();n++){
                array2d<double> new_w(sites[trial_cluster[n].v1()].rank(),r_k);
                double sum=0;
                for(size_t i=0;i<new_w.nx();i++){
                    for(size_t j=i;j<new_w.ny();j++){
                        new_w.at(i,j)=unif_dist(mpi_utils::prng);
                        sum+=new_w.at(i,j);
                        if(j!=i){
                            new_w.at(j,i)=new_w.at(i,j);
                            sum+=new_w.at(j,i);
                        }
                    }
                }
                for(size_t i=0;i<new_w.nx();i++){
                    for(size_t j=0;j<new_w.ny();j++){
                        new_w.at(i,j)/=sum;
                    }
                }
                trial_cluster[n].w()=new_w;
                // std::cout<<(std::string) new_w<<"\n";
            }
        }
        
        //nadam variables
        double alpha=0.0001;
        double beta1=0.9;
        double beta2=0.999;
        double epsilon=1e-10;
        array2d<double> m_current(trial_current.w().nx(),trial_current.w().ny());
        std::vector<array2d<double> > m_cluster;
        for(size_t n=0;n<trial_cluster.size();n++){
            m_cluster.push_back(array2d<double>(trial_cluster[n].w().nx(),trial_cluster[n].w().ny()));
        }
        array2d<double> v_current(trial_current.w().nx(),trial_current.w().ny());
        std::vector<array2d<double> > v_cluster;
        for(size_t n=0;n<trial_cluster.size();n++){
            v_cluster.push_back(array2d<double>(trial_cluster[n].w().nx(),trial_cluster[n].w().ny()));
        }
        array2d<double> g_current(trial_current.w().nx(),trial_current.w().ny());
        std::vector<array2d<double> > g_cluster;
        for(size_t n=0;n<trial_cluster.size();n++){
            g_cluster.push_back(array2d<double>(trial_cluster[n].w().nx(),trial_cluster[n].w().ny()));
        }
        
        //convergence check variables
        array2d<double> prev_current=trial_current.w();
        std::vector<array2d<double> > prev_cluster;
        for(size_t n=0;n<trial_cluster.size();n++){
            prev_cluster.push_back(trial_cluster[n].w());
        }
        double diff=0;
        double cost=0;
        array2d<double> p_ij(trial_current.w().nx(),trial_current.w().ny());
        array2d<double> p_prime_ij_env(trial_current.w().nx(),trial_current.w().ny());
        for(size_t t=0;t<max_it;t++){
            //calculate P_{ij},P'^{env}_{ij} and Z' (as sum_p_prime_ij_env)
            std::vector<double> sum_p_ij_addends;
            std::vector<double> sum_p_prime_ij_env_addends;
            for(size_t i=0;i<p_ij.nx();i++){
                for(size_t j=0;j<p_ij.ny();j++){
                    //calculate G'(f_k(S_i,S_j))
                    std::vector<double> gi_prime(r_k,1);
                    std::vector<double> gj_prime(r_k,1);
                    std::vector<double> gi_prime_factors;
                    std::vector<double> gj_prime_factors;
                    size_t k=trial_current.f().at(i,j); //select where output is added to
                    for(size_t m=0;m<cluster.size();m++){
                        if((old_cluster[m].v1()==old_current.v1())||(old_cluster[m].v2()==old_current.v1())){ //connected to site i
                            if((cluster[m].w().sum_over_axis(0))[k]>1e-10){ //avoid 0 factors
                                gi_prime_factors.push_back((cluster[m].w().sum_over_axis(0))[k]);
                            }
                        }
                        else{ //connected to site j
                            if((cluster[m].w().sum_over_axis(0))[k]>1e-10){ //avoid 0 factors
                                gj_prime_factors.push_back((cluster[m].w().sum_over_axis(0))[k]);
                            }
                        }
                    }
                    gi_prime[k]=vec_mult_float(gi_prime_factors);
                    gj_prime[k]=vec_mult_float(gj_prime_factors);
                    
                    std::vector<double> factors;
                    std::vector<double> factors_env;
                    //calculate P_{ij}
                    factors.push_back(old_current.w().at(i,j));
                    factors.push_back(gi[i]);
                    factors.push_back(gj[j]);
                    p_ij.at(i,j)=vec_mult_float(factors);
                    sum_p_ij_addends.push_back(p_ij.at(i,j));
                    //calculate P'^{env}_{ij}
                    factors_env.push_back(gi_prime[k]);
                    factors_env.push_back(gj_prime[k]);
                    p_prime_ij_env.at(i,j)=vec_mult_float(factors_env);
                    sum_p_prime_ij_env_addends.push_back(p_prime_ij_env.at(i,j)*trial_current.w().at(i,j)); //restore divided factor to get z'
                }
            }
            //normalize P_{ij},P'^{env}_{ij}
            double sum_p_ij=vec_add_float(sum_p_ij_addends);
            double sum_p_prime_ij_env=vec_add_float(sum_p_prime_ij_env_addends);
            for(size_t i=0;i<p_ij.nx();i++){
                for(size_t j=0;j<p_ij.ny();j++){
                        p_ij.at(i,j)/=sum_p_ij;
                        p_prime_ij_env.at(i,j)/=sum_p_prime_ij_env;
                }
            }
            // std::cout<<(std::string)trial_current.w()<<"\n";
            // std::cout<<(std::string)p_ij<<"\n";
            // std::cout<<(std::string)p_prime_ij_env<<"\n";

            double sum=0;
            for(size_t i=0;i<p_ij.nx();i++){
                for(size_t j=0;j<p_ij.ny();j++){
                    if(lr==0){ //lr==0 means use iterative method based on stationarity condition
                        if(fabs(p_prime_ij_env.at(i,j))>1e-10){
                            trial_current.w().at(i,j)=p_ij.at(i,j)/p_prime_ij_env.at(i,j);
                            if(trial_current.w().at(i,j)<1e-10){
                                trial_current.w().at(i,j)=1e-10;
                            }
                        }
                    }
                    else{ //lr!=0 means use gradient descent with lr
                        if(fabs(trial_current.w().at(i,j))>1e-10){
                            // trial_current.w().at(i,j)-=lr*(p_prime_ij_env.at(i,j)-(p_ij.at(i,j)/trial_current.w().at(i,j)));
                            double grad=p_prime_ij_env.at(i,j)-(p_ij.at(i,j)/trial_current.w().at(i,j));
                            grad*=2*sqrt(trial_current.w().at(i,j)); //get gradient of sqrt(w_ijk)
                            g_current.at(i,j)=grad;
                            m_current.at(i,j)=(beta1*m_current.at(i,j))+((1-beta1)*g_current.at(i,j));
                            v_current.at(i,j)=(beta2*v_current.at(i,j))+((1-beta2)*pow(g_current.at(i,j),2.0));
                            double bias_corrected_m=m_current.at(i,j)/(1-pow(beta1,(double) t+1));
                            double bias_corrected_v=v_current.at(i,j)/(1-pow(beta2,(double) t+1));
                            // trial_current.w().at(i,j)=pow(sqrt(trial_current.w().at(i,j))-(alpha*(bias_corrected_m/(sqrt(bias_corrected_v)+epsilon))),2.0);
                            trial_current.w().at(i,j)=pow(((1-(alpha*0.01))*sqrt(trial_current.w().at(i,j)))-(alpha*(bias_corrected_m/(sqrt(bias_corrected_v)+epsilon))),2.0);
                            // trial_current.w().at(i,j)=pow(sqrt(trial_current.w().at(i,j))-(lr*grad),2.0);
                            if(trial_current.w().at(i,j)<1e-10){ //in case the weight is negative, force it to be nonnegative!
                                trial_current.w().at(i,j)=1e-10;
                            }
                            sum+=trial_current.w().at(i,j);
                        }
                    }
                }
            }
            if(lr!=0){ //lr==0 means use iterative method based on stationarity condition
                for(size_t i=0;i<trial_current.w().nx();i++){
                    for(size_t j=0;j<trial_current.w().ny();j++){
                        trial_current.w().at(i,j)/=sum;
                        if(trial_current.w().at(i,j)<1e-10){ //in case the weight is negative, force it to be nonnegative!
                            trial_current.w().at(i,j)=1e-10;
                        }
                    }
                }
            }
            trial_current.bmi(trial_current.w());
            // std::cout<<"trial_current.w():\n"<<(std::string)trial_current.w()<<"\n";
            
            //NOTE: right now, this uses a workaround for floating-point non-associativity, which involves storing addends in vectors and then sorting them before adding.
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
                
                //prep to calculate P^_{ki_\mu}, P'^{env}_{ki_\mu}
                std::vector<std::vector<double> > addends(p_ki.nx()*p_ki.ny());
                std::vector<std::vector<double> > addends_env(p_prime_ki_env.nx()*p_prime_ki_env.ny());
                std::vector<double> sum_p_ki_addends;
                std::vector<double> sum_p_prime_ki_env_addends;
                trial_cluster[n].bmi(trial_cluster[n].w());
                std::vector<double> gi_prime(r_k,1);
                std::vector<double> gj_prime(r_k,1);
                for(size_t i=0;i<current.w().nx();i++){
                    for(size_t j=0;j<current.w().ny();j++){
                        //calculate G'(f_k(S_i,S_j))
                        std::vector<double> gi_prime_factors;
                        std::vector<double> gj_prime_factors;
                        size_t k=current.f().at(i,j); //select where output is added to
                        for(size_t m=0;m<trial_cluster.size();m++){
                            if((old_cluster[m].v1()==old_current.v1())||(old_cluster[m].v2()==old_current.v1())){ //connected to site i
                                if((trial_cluster[m].w().sum_over_axis(0))[k]>1e-10){ //avoid 0 factors
                                    gi_prime_factors.push_back((trial_cluster[m].w().sum_over_axis(0))[k]);
                                }
                            }
                            else{ //connected to site j
                                if((trial_cluster[m].w().sum_over_axis(0))[k]>1e-10){ //avoid 0 factors
                                    gj_prime_factors.push_back((trial_cluster[m].w().sum_over_axis(0))[k]);
                                }
                            }
                        }
                        gi_prime[k]=vec_mult_float(gi_prime_factors);
                        gj_prime[k]=vec_mult_float(gj_prime_factors);
                    }
                }
                for(size_t i=0;i<current.w().nx();i++){
                    for(size_t j=0;j<current.w().ny();j++){
                        size_t k=current.f().at(i,j); //select where output is added to
                        //get addends
                        for(size_t imu=0;imu<p_ki.nx();imu++){
                            std::vector<double> factors;
                            std::vector<double> factors_env;
                            //calculate P_{ki_\mu} addends
                            factors.push_back(old_current.w().at(i,j));
                            if(source==current.v1()){ //connected to site i
                                if(old_cluster[n].v1()==current.v1()){ //site imu > site i
                                    factors.push_back(old_cluster[n].w().at(i,imu));
                                    factors.push_back(1/(old_cluster[n].w().sum_over_axis(1))[i]);
                                }
                                else{ //site imu < site i
                                    factors.push_back(old_cluster[n].w().at(imu,i));
                                    factors.push_back(1/(old_cluster[n].w().sum_over_axis(0))[i]);
                                }
                            }
                            else{ //connected to site j
                                if(old_cluster[n].v1()==current.v2()){ //site imu > site j
                                    factors.push_back(old_cluster[n].w().at(j,imu));
                                    factors.push_back(1/(old_cluster[n].w().sum_over_axis(1))[j]);
                                }
                                else{ //site imu < site j
                                    factors.push_back(old_cluster[n].w().at(imu,j));
                                    factors.push_back(1/(old_cluster[n].w().sum_over_axis(0))[j]);
                                }
                            }
                            factors.push_back(gi[i]);
                            factors.push_back(gj[j]);
                            addends[(k*p_ki.nx())+imu].push_back(vec_mult_float(factors));
                            sum_p_ki_addends.push_back(vec_mult_float(factors));
                            //calculate P'^{env}_{ki_\mu} addends
                            factors_env.push_back(current.w().at(i,j));
                            factors_env.push_back(gi_prime[k]);
                            factors_env.push_back(gj_prime[k]);
                            factors_env.push_back(1/(trial_cluster[n].w().sum_over_axis(0))[k]); //divide appropriate factor
                            addends_env[(k*p_prime_ki_env.nx())+imu].push_back(vec_mult_float(factors_env));
                            sum_p_prime_ki_env_addends.push_back(vec_mult_float(factors_env)*trial_cluster[n].w().at(imu,k)); //restore divided factor to get z'
                        }
                    }
                }
                //calculate P^_{ki_\mu}, P'^{env}_{ki_\mu}
                for(size_t imu=0;imu<p_ki.nx();imu++){
                    for(size_t k=0;k<p_ki.ny();k++){
                        p_ki.at(imu,k)=vec_add_float(addends[(k*p_ki.nx())+imu]);
                        p_prime_ki_env.at(imu,k)=vec_add_float(addends_env[(k*p_ki.nx())+imu]);
                    }
                }
                double sum_p_ki=vec_add_float(sum_p_ki_addends);
                double sum_p_prime_ki_env=vec_add_float(sum_p_prime_ki_env_addends);
                // std::cout<<"p_ki: "<<(std::string) p_ki<<"\n";
                // std::cout<<"p_ki_env: "<<(std::string) p_prime_ki_env<<"\n";
                //normalize P_{ki_\mu},P'^{env}_{ki_\mu}
                for(size_t i=0;i<p_ki.nx();i++){
                    for(size_t j=0;j<p_ki.ny();j++){
                        p_ki.at(i,j)/=sum_p_ki;
                        p_prime_ki_env.at(i,j)/=sum_p_prime_ki_env;
                    }
                }
                
                double sum=0;
                for(size_t imu=0;imu<p_ki.nx();imu++){
                    for(size_t k=0;k<p_ki.ny();k++){
                        if(lr==0){ //lr==0 means use iterative method based on stationarity condition
                            if(fabs(p_prime_ki_env.at(imu,k))>1e-10){
                                trial_cluster[n].w().at(imu,k)=p_ki.at(imu,k)/p_prime_ki_env.at(imu,k);
                                if(trial_cluster[n].w().at(imu,k)<1e-10){
                                    trial_cluster[n].w().at(imu,k)=1e-10;
                                }
                            }
                        }
                        else{ //lr!=0 means use gradient descent with lr
                            if(fabs(trial_cluster[n].w().at(imu,k))>1e-10){
                                // trial_cluster[n].w().at(imu,k)-=lr*(p_prime_ki_env.at(imu,k)-(p_ki.at(imu,k)/trial_cluster[n].w().at(imu,k)));
                                double grad=p_prime_ki_env.at(imu,k)-(p_ki.at(imu,k)/trial_cluster[n].w().at(imu,k));
                                grad*=2*sqrt(trial_cluster[n].w().at(imu,k)); //get gradient of sqrt(w_kimu)
                                g_cluster[n].at(imu,k)=grad;
                                m_cluster[n].at(imu,k)=(beta1*m_cluster[n].at(imu,k))+((1-beta1)*g_cluster[n].at(imu,k));
                                v_cluster[n].at(imu,k)=(beta2*v_cluster[n].at(imu,k))+((1-beta2)*pow(g_cluster[n].at(imu,k),2.0));
                                double bias_corrected_m=m_cluster[n].at(imu,k)/(1-pow(beta1,(double) t+1));
                                double bias_corrected_v=v_cluster[n].at(imu,k)/(1-pow(beta2,(double) t+1));
                                // trial_cluster[n].w().at(imu,k)=pow(sqrt(trial_cluster[n].w().at(imu,k))-(alpha*(bias_corrected_m/(sqrt(bias_corrected_v)+epsilon))),2.0);
                                trial_cluster[n].w().at(imu,k)=pow(((1-(alpha*0.01))*sqrt(trial_cluster[n].w().at(imu,k)))-(alpha*(bias_corrected_m/(sqrt(bias_corrected_v)+epsilon))),2.0);
                                // trial_cluster[n].w().at(imu,k)=pow(sqrt(trial_cluster[n].w().at(imu,k))-(lr*grad),2.0);
                                if(trial_cluster[n].w().at(imu,k)<1e-10){ //in case the weight is negative, force it to be nonnegative!
                                    trial_cluster[n].w().at(imu,k)=1e-10;
                                }
                                sum+=trial_cluster[n].w().at(imu,k);
                            }
                        }
                    }
                }
                // std::cout<<"trial_cluster[n]: "<<(std::string) trial_cluster[n].w()<<"\n";
                if(lr!=0){ //lr==0 means use iterative method based on stationarity condition
                    for(size_t imu=0;imu<trial_cluster[n].w().nx();imu++){
                        for(size_t k=0;k<trial_cluster[n].w().ny();k++){
                            trial_cluster[n].w().at(imu,k)/=sum;
                            if(trial_cluster[n].w().at(imu,k)<1e-10){ //in case the weight is negative, force it to be nonnegative!
                                trial_cluster[n].w().at(imu,k)=1e-10;
                            }
                        }
                    }
                }
                trial_cluster[n].bmi(trial_cluster[n].w());
            }
            
            //check for convergence if after first iteration (since trial_current must be resized)
            cost=kl_div(sites,old_current,old_cluster,current,cluster);
            // if(cost>1e3){ //if too large, reinitialize, doesn't count
                // std::cout<<"cost too large. discarding result.\n";
                // restarts--;
                // break;
            // }
            if(cost<1){ //if too small, reinitialize, doesn't count
                std::cout<<"cost less than 1. discarding result.\n";
                restarts--;
                break;
            }
            diff=0;
            if(t>0){
                for(size_t i=0;i<trial_current.w().nx();i++){
                    for(size_t j=0;j<trial_current.w().ny();j++){
                        diff+=fabs(trial_current.w().at(i,j)-prev_current.at(i,j));
                    }
                }
                for(size_t n=0;n<trial_cluster.size();n++){
                    for(size_t i=0;i<trial_cluster[n].w().nx();i++){
                        for(size_t j=0;j<trial_cluster[n].w().ny();j++){
                            diff+=fabs(trial_cluster[n].w().at(i,j)-prev_cluster[n].at(i,j));
                        }
                    }
                }
                if(diff<1e-6){
                    std::cout<<"converged after "<<(t+1)<<" iterations (diff==0)\n";
                    break;
                }
                if(fabs(cost-1)<1e-6){
                    std::cout<<"converged after "<<(t+1)<<" iterations (cost==1)\n";
                    break;
                }
                if(t==max_it-1){
                    std::cout<<"no convergence after "<<(max_it)<<" iterations\n";
                }
            }
            
            //update convergence check variables
            prev_current=trial_current.w();
            for(size_t n=0;n<trial_cluster.size();n++){
                prev_cluster[n]=trial_cluster[n].w();
            }
        }
        std::cout<<"final cost: "<<cost<<"\n";
        // std::cout<<"final diff: "<<diff<<"\n";
                    
        if(cost<best_cost){
            std::cout<<"cost improved. replacing...\n";
            best_cost=cost;
            best_current=trial_current;
            for(size_t n=0;n<best_cluster.size();n++){
                best_cluster[n]=trial_cluster[n];
            }
        }
        if(fabs(best_cost-1)<1e-6){
            std::cout<<"identical distribution obtained. stopping optimization.\n";
            break;
        }
    }
    std::cout<<"best cost: "<<best_cost<<"\n";
    current=best_current;
    for(size_t n=0;n<cluster.size();n++){
        cluster[n]=best_cluster[n];
    }
}

//cmd function mimicking alg. 1. this assumes and results in a fixed rank.
array2d<size_t> optimize::f_alg1(site v_i,site v_j){
    array2d<size_t> f_res(v_i.rank(),v_j.rank());
    for(size_t s_i=0;s_i<v_i.rank();s_i++){
        for(size_t s_j=0;s_j<v_j.rank();s_j++){
            f_res.at(s_i,s_j)=(v_i.vol()>v_j.vol())?s_i:s_j;
            // f_res.at(s_i,s_j)=s_i;
        }
    }
    return f_res;
}

//cmd function that maximizes the entropy
array2d<size_t> optimize::f_maxent(site v_i,site v_j,array2d<double> w,size_t r_k){
    array2d<size_t> f_res(v_i.rank(),v_j.rank());
    std::vector<double> w_sums(r_k);
    // std::cout<<v_i.rank()*v_j.rank()<<"\n";
    for(size_t n=0;n<v_i.rank()*v_j.rank();n++){
        auto max_it=std::max_element(w.e().begin(),w.e().end());
        double max=*max_it;
        size_t argmax=std::distance(w.e().begin(),max_it);
        size_t i=argmax%w.nx();
        size_t j=argmax/w.nx();
        size_t k=std::distance(w_sums.begin(),std::min_element(w_sums.begin(),w_sums.end())); //choose first case of smallest cumulative sum
        w_sums[k]+=max;
        // std::cout<<v_i.vol()<<" "<<v_j.vol()<<" "<<i<<" "<<j<<" "<<k<<" "<<w_sums[k]<<"\n";
        w.at(i,j)=-1; //weights can never be negative so this element will never be the max. also, w is passed by value so no change to original.
        f_res.at(i,j)=k;
    }
    // double S=0;
    // for(size_t n=0;n<w_sums.size();n++){
        // S-=w_sums[n]*log(w_sums[n]);
    // }
    // std::cout<<"entropy S="<<S<<"\n";
    return f_res;
}

//cmd function that maximizes the similarity between vector magnetizations of the downstream sites and the vector magnetization of the reference potts basis vectors upstream
array2d<size_t> optimize::f_mvec_sim(site v_i,site v_j,array2d<double> w,size_t r_k){
    size_t r_i=v_i.rank();
    size_t r_j=v_j.rank();
    array2d<size_t> f_res(r_i,r_j);
    //TODO
    //calculate dot product
    std::vector<double> w_sums(r_k,0);
    std::vector<std::pair<size_t,size_t> > done_ij_pairs; //for part that ensures surjectivity
    array3d<double> c(r_i,r_j,r_k);
    for(size_t i=0;i<r_i;i++){
        for(size_t j=0;j<r_j;j++){
            //get sum of m_vecs
            std::vector<double> m_i=v_i.m_vec()[i];
            std::vector<double> m_j=v_j.m_vec()[j];
            if(r_i>r_j){
                while(m_i.size()!=m_j.size()){
                    m_j.push_back(0);
                }
            }
            if(r_i<r_j){
                while(m_i.size()!=m_j.size()){
                    m_i.push_back(0);
                }
            }
            std::vector<double> m_i_m_j_sum((r_i>r_j)?r_i-1:r_j-1,0);
            for(size_t idx=0;idx<m_i_m_j_sum.size();idx++){
                m_i_m_j_sum[idx]=m_i[idx]+m_j[idx];
            }
            //get reference potts basis
            std::vector<std::vector<double> > v;
            try{
                v=observables::m_vec_ref_cache.at(r_k);
            }
            catch(const std::out_of_range& oor){
                v=potts_ref_vecs(r_k);
                observables::m_vec_ref_cache[r_k]=v;
            }
            //populate c
            for(size_t k=0;k<r_k;k++){
                std::vector<double> e_k=v[k];
                if(m_i_m_j_sum.size()>e_k.size()){
                    while(m_i_m_j_sum.size()!=e_k.size()){
                        e_k.push_back(0);
                    }
                }
                if(m_i_m_j_sum.size()<e_k.size()){
                    while(m_i_m_j_sum.size()!=e_k.size()){
                        m_i_m_j_sum.push_back(0);
                    }
                }
                double dot=0;
                std::vector<double> dot_addends;
                for(size_t idx=0;idx<m_i_m_j_sum.size();idx++){
                    // dot+=m_i_m_j_sum[idx]*e_k[idx];
                    dot_addends.push_back(m_i_m_j_sum[idx]*e_k[idx]);
                }
                dot=vec_add_float(dot_addends);
                c.at(i,j,k)=dot;
                // c.at(i,j,k)=dot*w.at(i,j);
            }
        }
    }
    //satisfy surjectivity
    /* for(size_t k=0;k<r_k;k++){
        size_t max_c_idx_i=0;
        size_t max_c_idx_j=0;
        double max_c=c.at(0,0,k)-(1e-10*w_sums[k]);
        // std::cout<<(c.at(0,0,k)-(1e-10*w_sums[k]))<<" "<<max_c<<"\n";
        for(size_t i=0;i<r_i;i++){
            for(size_t j=0;j<r_j;j++){
                //skip if i,j in done_ij_pairs
                auto it=std::find(done_ij_pairs.begin(),done_ij_pairs.end(),std::make_pair(i,j));
                if(it!=done_ij_pairs.end()){
                    continue;
                }
                // std::cout<<(c.at(i,j,k)-(1e-10*w_sums[k]))<<" "<<max_c<<"\n";
                if((c.at(i,j,k)-(1e-10*w_sums[k]))>max_c){
                    max_c_idx_i=i;
                    max_c_idx_j=j;
                    max_c=c.at(i,j,k)-(1e-10*w_sums[k]);
                }
            }
        }
        f_res.at(max_c_idx_i,max_c_idx_j)=k;
        w_sums[k]+=w.at(max_c_idx_i,max_c_idx_j);
        done_ij_pairs.push_back(std::make_pair(max_c_idx_i,max_c_idx_j));
    } */
    //determine rest of f
    for(size_t i=0;i<r_i;i++){
        for(size_t j=0;j<r_j;j++){
            //skip if i,j in done_ij_pairs
            auto it=std::find(done_ij_pairs.begin(),done_ij_pairs.end(),std::make_pair(i,j));
            if(it!=done_ij_pairs.end()){
                continue;
            }
            size_t max_c_idx=0;
            double max_c=c.at(i,j,0)-(1e-10*w_sums[0]);
            // std::cout<<(c.at(i,j,0)-(1e-10*w_sums[0]))<<" "<<max_c<<"\n";
            for(size_t k=1;k<r_k;k++){
                // std::cout<<(c.at(i,j,k)-(1e-10*w_sums[k]))<<" "<<max_c<<"\n";
                if((c.at(i,j,k)-(1e-10*w_sums[k]))>max_c){
                    max_c_idx=k;
                    max_c=c.at(i,j,k)-(1e-10*w_sums[k]);
                }
            }
            f_res.at(i,j)=max_c_idx;
            w_sums[max_c_idx]+=w.at(i,j);
            // for(size_t i=0;i<w_sums.size();i++){
                // std::cout<<w_sums[i]<<" ";
            // }
            // std::cout<<"\n";
        }
    }
    // std::cout<<(std::string)w<<"\n";
    // std::cout<<(std::string)c<<"\n";
    // std::cout<<"new: "<<(std::string)f_res<<"\n";
    // double S=0;
    // for(size_t n=0;n<w_sums.size();n++){
        // S-=w_sums[n]*log(w_sums[n]);
    // }
    // std::cout<<"entropy S="<<S<<"\n";
    return f_res;
}

//cmd function that uses maxent for mismatched triplets and alg1 when r_i=r_j=r_k
array2d<size_t> optimize::f_hybrid_maxent(site v_i,site v_j,array2d<double> w,size_t r_k){
    if((v_i.rank()==v_j.rank())&&(v_i.rank()==r_k)){
        return f_alg1(v_i,v_j);
    }
    else{
        return f_maxent(v_i,v_j,w,r_k);
    }
}

//cmd function that uses mvec_sim for mismatched triplets and alg1 when r_i=r_j=r_k
array2d<size_t> optimize::f_hybrid_mvec_sim(site v_i,site v_j,array2d<double> w,size_t r_k){
    if((v_i.rank()==v_j.rank())&&(v_i.rank()==r_k)){
        return f_alg1(v_i,v_j);
    }
    else{
        return f_mvec_sim(v_i,v_j,w,r_k);
    }
}

double optimize::kl_div(std::vector<site> sites,bond old_current,std::vector<bond> old_cluster,bond current,std::vector<bond> cluster){
    // std::cout<<(std::string)current<<"\n";
    // std::cout<<(std::string)current.f()<<"\n";
    // std::cout<<(std::string)current.w()<<"\n";
    
    std::vector<double> g_prime(*(std::max_element(current.f().e().begin(),current.f().e().end()))+1);
    for(size_t k=0;k<g_prime.size();k++){
        double g_prime_res=1;
        for(size_t n=0;n<cluster.size();n++){
            double spin_sum_prime=0;
            for(size_t imu=0;imu<cluster[n].w().nx();imu++){
                spin_sum_prime+=cluster[n].w().at(imu,k);
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
                        spin_sum_i+=old_cluster[n].w().at(i,imu);
                    }
                }
                else{ //site imu < site i
                    for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                        spin_sum_i+=old_cluster[n].w().at(imu,i);
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
                        spin_sum_j+=old_cluster[n].w().at(j,imu);
                    }
                }
                else{ //site imu < site j
                    for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                        spin_sum_j+=old_cluster[n].w().at(imu,j);
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
            double gi_tilde_res=0;
            double gj_tilde_res=0;
            for(size_t n=0;n<old_cluster.size();n++){
                double spin_sum_i_tilde=0;
                double spin_sum_j_tilde=0;
                double div_factor=0;
                size_t k=current.f().at(i,j);
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
                            div_factor+=old_cluster[n].w().at(i,imu);
                        }
                        for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                            if(cluster[n].w().at(imu,k)>0){
                                spin_sum_i_tilde+=old_cluster[n].w().at(i,imu)*log(cluster[n].w().at(imu,k));
                            }
                        }
                    }
                    else{ //site imu < site i
                        for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                            div_factor+=old_cluster[n].w().at(imu,i);
                        }
                        for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                            if(cluster[n].w().at(imu,k)>0){
                                spin_sum_i_tilde+=old_cluster[n].w().at(imu,i)*log(cluster[n].w().at(imu,k));
                            }
                        }
                    }
                    gi_tilde_res+=(div_factor>0)?(spin_sum_i_tilde*gi[i]/div_factor):0;
                }
                else{ //connected to site j
                    if(old_cluster[n].v1()==old_current.v2()){ //site imu > site j
                        for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                            div_factor+=old_cluster[n].w().at(j,imu);
                        }
                        for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                            if(cluster[n].w().at(imu,k)>0){
                                spin_sum_j_tilde+=old_cluster[n].w().at(j,imu)*log(cluster[n].w().at(imu,k));
                            }
                        }
                    }
                    else{ //site imu < site j
                        for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                            div_factor+=old_cluster[n].w().at(imu,j);
                        }
                        for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                            if(cluster[n].w().at(imu,k)>0){
                                spin_sum_j_tilde+=old_cluster[n].w().at(imu,j)*log(cluster[n].w().at(imu,k));
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
    
    double z_prime=0;
    double z=0;
    double a=0;
    double b=0;
    for(size_t i=0;i<sites[current.v1()].rank();i++){
        for(size_t j=0;j<sites[current.v2()].rank();j++){
            double contrib_z_prime=current.w().at(i,j)*g_prime[current.f().at(i,j)];
            double contrib_z=old_current.w().at(i,j)*gi[i]*gj[j];
            double contrib_a=(current.w().at(i,j)>0)?old_current.w().at(i,j)*gi[i]*gj[j]*log(current.w().at(i,j)):0;
            double contrib_b=old_current.w().at(i,j)*((gi_tilde.at(i,j)*gj[j])+(gi[i]*gj_tilde.at(i,j)));
            size_t k=current.f().at(i,j);
            double contrib_b_logs=0;
            z_prime+=contrib_z_prime;
            z+=contrib_z;
            a+=contrib_a;
            b+=contrib_b;
        }
    }
    // std::cout<<"z: "<<z<<"\nz': "<<z_prime<<"\nz': "<<z_prime2<<"\n";
    // std::cout<<"a: "<<a<<"\nb: "<<b<<"\n";
    double res=log(z_prime)-(a/z)-(b/z);
    return res;
}