#include <fstream>

#include "../mpi_utils.hpp"
#include "optimize.hpp"
#include "../observables.hpp"

double optimize::opt(size_t master,size_t slave,size_t r_k,std::vector<site> sites,bond& old_current,std::vector<bond>& old_cluster,bond& current,std::vector<bond>& cluster,size_t max_it,double lr,size_t max_restarts){
    std::uniform_real_distribution<> unif_dist(1e-10,1.0);
    double best_cost=1e49;
    bond best_current=current;
    std::vector<bond> best_cluster;
    for(size_t n=0;n<cluster.size();n++){
        best_cluster.push_back(cluster[n]);
    }
    
    //precompute intermediate quantities for weight optimization
    //calculate G_i(S_i) and G_j(S_j)
    std::vector<double> gi(old_current.w().nx(),1);
    std::vector<double> gj(old_current.w().ny(),1);
    for(size_t i=0;i<old_current.w().nx();i++){
        std::vector<double> gi_factors;
        for(size_t n=0;n<old_cluster.size();n++){
            if((old_cluster[n].v1()==old_current.v1())||(old_cluster[n].v2()==old_current.v1())){
                gi_factors.push_back((old_cluster[n].w().lse_over_axis((old_cluster[n].v1()==old_current.v1())?1:0))[i]);
            }
        }
        gi[i]=vec_add_float(gi_factors);
    }
    for(size_t j=0;j<old_current.w().ny();j++){
        std::vector<double> gj_factors;
        for(size_t n=0;n<old_cluster.size();n++){
            if((old_cluster[n].v1()==old_current.v2())||(old_cluster[n].v2()==old_current.v2())){
                gj_factors.push_back((old_cluster[n].w().lse_over_axis((old_cluster[n].v1()==old_current.v2())?1:0))[j]);
            }
        }
        gj[j]=vec_add_float(gj_factors);
    }
    //calculate Z stored as ln(Z)
    std::vector<double> z_addends;
    for(size_t i=0;i<old_current.w().nx();i++){
        for(size_t j=0;j<old_current.w().ny();j++){
            z_addends.push_back(old_current.w().at(i,j)+gi[i]+gj[j]);
        }
    }
    double z=lse(z_addends);
    
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
                        for(size_t j=0;j<new_w.ny();j++){
                            if((i<old_cluster[n].w().nx())&&(j<old_cluster[n].w().ny())){
                                new_w.at(i,j)=exp(old_cluster[n].w().at(i,j))*(old_cluster[n].w().nx()*old_cluster[n].w().ny())/(double) (new_w.nx()*new_w.ny());
                            }
                            else{
                                new_w.at(i,j)=1/(double) (new_w.nx()*new_w.ny());
                            }
                            sum+=new_w.at(i,j);
                        }
                    }
                    for(size_t i=0;i<new_w.nx();i++){
                        for(size_t j=0;j<new_w.ny();j++){
                            new_w.at(i,j)/=sum;
                            new_w.at(i,j)=log(new_w.at(i,j));
                        }
                    }
                    trial_cluster[n].w()=new_w;
                }
            }
        }
        else{
            array2d<double> new_w(trial_current.w().nx(),trial_current.w().ny());
            double sum=0;
            for(size_t i=0;i<new_w.nx();i++){
                for(size_t j=0;j<new_w.ny();j++){
                    new_w.at(i,j)=unif_dist(mpi_utils::prng);
                    sum+=new_w.at(i,j);
                }
            }
            for(size_t i=0;i<new_w.nx();i++){
                for(size_t j=0;j<new_w.ny();j++){
                    new_w.at(i,j)/=sum;
                    new_w.at(i,j)=log(new_w.at(i,j));
                }
            }
            trial_current.w()=new_w;
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
                        new_w.at(i,j)=log(new_w.at(i,j));
                    }
                }
                trial_cluster[n].w()=new_w;
            }
        }
        //cost convergence variables
        double prev_cost=1e50;
        double ewma_cost=prev_cost;
        size_t window_size=10;
        //win-adamw variables
        double alpha=lr;
        double beta1=0.9;
        double beta2=0.999;
        double epsilon=1e-10;
        double lambda=0.01; //weight decay
        double reckless_alpha=2*alpha; //win-adamw
        double tau=1/(alpha+reckless_alpha+(alpha*reckless_alpha*lambda)); //win-adamw
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
        array2d<double> x_current(trial_current.w().nx(),trial_current.w().ny());
        std::vector<array2d<double> > x_cluster;
        for(size_t n=0;n<trial_cluster.size();n++){
            x_cluster.push_back(array2d<double>(trial_cluster[n].w().nx(),trial_cluster[n].w().ny()));
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
                            gi_prime_factors.push_back((trial_cluster[m].w().lse_over_axis(0))[k]);
                        }
                        else{ //connected to site j
                            gj_prime_factors.push_back((trial_cluster[m].w().lse_over_axis(0))[k]);
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
                    factors.push_back(old_current.w().at(i,j));
                    factors.push_back(gi[i]);
                    factors.push_back(gj[j]);
                    p_ij.at(i,j)=vec_add_float(factors);
                    sum_p_ij_addends.push_back(p_ij.at(i,j));
                    //calculate P'^{env}_{ij}
                    size_t k=current.f().at(i,j); //select where output is added to
                    factors_env.push_back(gi_prime[k]);
                    factors_env.push_back(gj_prime[k]);
                    p_prime_ij_env.at(i,j)=vec_add_float(factors_env);
                    sum_p_prime_ij_env_addends.push_back(p_prime_ij_env.at(i,j)+trial_current.w().at(i,j)); //restore divided factor to get z'
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
            cost=kl_div(exp(z),exp(z_prime),gi,gj,sites,old_current,old_cluster,trial_current,trial_cluster);
            // std::cout<<sum_p_prime_ij_env<<"\n";
            // std::cout<<(std::string)trial_current.w()<<"\n";
            // std::cout<<(std::string)p_ij<<"\n";
            // std::cout<<(std::string)p_prime_ij_env<<"\n";

            std::vector<double> sum_addends;
            for(size_t i=0;i<p_ij.nx();i++){
                for(size_t j=0;j<p_ij.ny();j++){
                    if(lr==0){ //lr==0 means use iterative method based on stationarity condition
                        trial_current.w().at(i,j)=p_ij.at(i,j)-p_prime_ij_env.at(i,j);
                        sum_addends.push_back(trial_current.w().at(i,j));
                    }
                    else{ //lr!=0 means use gradient descent with lr
                        double a=p_prime_ij_env.at(i,j)+trial_current.w().at(i,j);
                        double b=p_ij.at(i,j);
                        if(a!=b){ //a==b means grad C is 0, so no change to params
                            double grad=(a>b)?exp(a+log(1-exp(b-a))):-exp(b+log(1-exp(a-b))); //gradient can be negative, must be done in normal space, of ln(w_ijk)
                            g_current.at(i,j)=grad;
                            m_current.at(i,j)=(t==0)?g_current.at(i,j):(beta1*m_current.at(i,j))+((1-beta1)*g_current.at(i,j));
                            v_current.at(i,j)=(t==0)?pow(g_current.at(i,j),2.0):(beta2*v_current.at(i,j))+((1-beta2)*pow(g_current.at(i,j),2.0));
                            double bias_corrected_m=m_current.at(i,j)/(1-pow(beta1,(double) t+1));
                            double bias_corrected_v=v_current.at(i,j)/(1-pow(beta2,(double) t+1));
                            // trial_current.w().at(i,j)=((1-(alpha*0.01))*trial_current.w().at(i,j))-(alpha*(bias_corrected_m/(sqrt(bias_corrected_v)+epsilon))); //adamw
                            double u=bias_corrected_m/(sqrt(bias_corrected_v)+epsilon); //win-adamw
                            x_current.at(i,j)=(1/(1+(alpha*0.01)))*(trial_current.w().at(i,j)-(alpha*u)); //win-adamw
                            trial_current.w().at(i,j)=(reckless_alpha*tau*x_current.at(i,j))+((alpha*tau)*(trial_current.w().at(i,j)-(reckless_alpha*u))); //win-adamw
                        }
                        sum_addends.push_back(trial_current.w().at(i,j));
                    }
                }
            }
            double sum=lse(sum_addends);
            for(size_t i=0;i<trial_current.w().nx();i++){
                for(size_t j=0;j<trial_current.w().ny();j++){
                    trial_current.w().at(i,j)-=sum;
                    if(trial_current.w().at(i,j)<log(1e-100)){ //in case the weight is negative, force it to be nonnegative!
                        trial_current.w().at(i,j)=log(1e-100);
                    }
                }
            }
            // std::cout<<"trial_current.w():\n"<<(std::string)trial_current.w()<<"\n";
            
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
                                    factors.push_back(-(old_cluster[n].w().lse_over_axis(1))[i]);
                                }
                                else{ //site imu < site i
                                    factors.push_back(old_cluster[n].w().at(imu,i));
                                    factors.push_back(-(old_cluster[n].w().lse_over_axis(0))[i]);
                                }
                            }
                            else{ //connected to site j
                                if(old_cluster[n].v1()==current.v2()){ //site imu > site j
                                    factors.push_back(old_cluster[n].w().at(j,imu));
                                    factors.push_back(-(old_cluster[n].w().lse_over_axis(1))[j]);
                                }
                                else{ //site imu < site j
                                    factors.push_back(old_cluster[n].w().at(imu,j));
                                    factors.push_back(-(old_cluster[n].w().lse_over_axis(0))[j]);
                                }
                            }
                            factors.push_back(gi[i]);
                            factors.push_back(gj[j]);
                            addends[(k*p_ki.nx())+imu].push_back(vec_add_float(factors));
                            sum_p_ki_addends.push_back(vec_add_float(factors));
                            //calculate P'^{env}_{ki_\mu} addends
                            factors_env.push_back(current.w().at(i,j));
                            factors_env.push_back(gi_prime[k]);
                            factors_env.push_back(gj_prime[k]);
                            factors_env.push_back(-(trial_cluster[n].w().lse_over_axis(0))[k]); //divide appropriate factor
                            addends_env[(k*p_prime_ki_env.nx())+imu].push_back(vec_add_float(factors_env));
                            sum_p_prime_ki_env_addends.push_back(vec_add_float(factors_env)+trial_cluster[n].w().at(imu,k)); //restore divided factor to get z'
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
                        if(lr==0){ //lr==0 means use iterative method based on stationarity condition
                            trial_cluster[n].w().at(imu,k)=p_ki.at(imu,k)-p_prime_ki_env.at(imu,k);
                            sum_addends.push_back(trial_cluster[n].w().at(imu,k));
                        }
                        else{ //lr!=0 means use gradient descent with lr
                            double a=p_prime_ki_env.at(imu,k)+trial_cluster[n].w().at(imu,k);
                            double b=p_ki.at(imu,k);
                            if(a!=b){ //a==b means grad C is 0, so no change to params
                                double grad=(a>b)?exp(a+log(1-exp(b-a))):-exp(b+log(1-exp(a-b))); //gradient can be negative, must be done in normal space, of ln(w_ijk)
                                g_cluster[n].at(imu,k)=grad;
                                m_cluster[n].at(imu,k)=(t==0)?g_cluster[n].at(imu,k):(beta1*m_cluster[n].at(imu,k))+((1-beta1)*g_cluster[n].at(imu,k));
                                v_cluster[n].at(imu,k)=(t==0)?pow(g_cluster[n].at(imu,k),2.0):(beta2*v_cluster[n].at(imu,k))+((1-beta2)*pow(g_cluster[n].at(imu,k),2.0));
                                double bias_corrected_m=m_cluster[n].at(imu,k)/(1-pow(beta1,(double) t+1));
                                double bias_corrected_v=v_cluster[n].at(imu,k)/(1-pow(beta2,(double) t+1));
                                // trial_cluster[n].w().at(imu,k)=((1-(alpha*0.01))*trial_cluster[n].w().at(imu,k))-(alpha*(bias_corrected_m/(sqrt(bias_corrected_v)+epsilon))); //adamw
                                double u=bias_corrected_m/(sqrt(bias_corrected_v)+epsilon); //win-adamw
                                x_cluster[n].at(imu,k)=(1/(1+(alpha*0.01)))*(trial_cluster[n].w().at(imu,k)-(alpha*u)); //win-adamw
                                trial_cluster[n].w().at(imu,k)=(reckless_alpha*tau*x_cluster[n].at(imu,k))+((alpha*tau)*(trial_cluster[n].w().at(imu,k)-(reckless_alpha*u))); //win-adamw
                            }
                            sum_addends.push_back(trial_cluster[n].w().at(imu,k));
                        }
                    }
                }
                double sum=lse(sum_addends);
                for(size_t imu=0;imu<trial_cluster[n].w().nx();imu++){
                    for(size_t k=0;k<trial_cluster[n].w().ny();k++){
                        trial_cluster[n].w().at(imu,k)-=sum;
                        if(trial_cluster[n].w().at(imu,k)<log(1e-100)){ //in case the weight is negative, force it to be nonnegative!
                            trial_cluster[n].w().at(imu,k)=log(1e-100);
                        }
                    }
                }
            }
            
            //check for convergence if after first iteration (since trial_current must be resized)
            // std::cout<<cost<<"\n";
            if((fabs(cost)>=1e-5)&&(cost<0)){ //if too small, reinitialize, doesn't count
                std::cout<<"cost less than 0. discarding result.\n";
                restarts--;
                exit(1);
                break;
            }
            if(t>0){
                ewma_cost=(((window_size-1)*ewma_cost)+(prev_cost-cost))/(double)window_size;
                // std::cout<<"ewma_cost: "<<ewma_cost<<"\n";
                // if(ewma_cost<0){
                    // std::cout<<"converged after "<<(t+1)<<" iterations (ewma_cost)\n";
                    // break;
                // }
                if(fabs(prev_cost-cost)<1e-5){
                    std::cout<<"converged after "<<(t+1)<<" iterations (cost)\n";
                    break;
                }
                if(t==max_it-1){
                    std::cout<<"no convergence after "<<(max_it)<<" iterations\n";
                }
                prev_cost=cost;
            }
            
            //update convergence check variables
            prev_current=trial_current.w();
            for(size_t n=0;n<trial_cluster.size();n++){
                prev_cluster[n]=trial_cluster[n].w();
            }
        }
        if((fabs(cost)>=1e-5)&&(cost<0)){
            continue;
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
        if(fabs(best_cost)<1e-5){
            std::cout<<"identical distribution obtained. stopping optimization.\n";
            break;
        }
    }
    std::cout<<"best cost: "<<best_cost<<"\n";
    current=best_current;
    current.bmi(current.w());
    for(size_t n=0;n<cluster.size();n++){
        cluster[n]=best_cluster[n];
        cluster[n].bmi(cluster[n].w());
    }
    return best_cost;
}

//cmd function mimicking alg. 1. this assumes and results in a fixed rank.
array2d<size_t> optimize::f_alg1(site v_i,site v_j){
    array2d<size_t> f_res(v_i.rank(),v_j.rank());
    for(size_t s_i=0;s_i<v_i.rank();s_i++){
        for(size_t s_j=0;s_j<v_j.rank();s_j++){
            f_res.at(s_i,s_j)=(v_i.vol()>v_j.vol())?s_i:s_j;
        }
    }
    return f_res;
}

//cmd function that maximizes the entropy
array2d<size_t> optimize::f_maxent(site v_i,site v_j,array2d<double> w,size_t r_k){
    array2d<size_t> f_res(v_i.rank(),v_j.rank());
    std::vector<double> w_sums(r_k);
    array2d<double> exp_w(w.nx(),w.ny());
    for(size_t i=0;i<exp_w.nx();i++){
        for(size_t j=0;j<exp_w.ny();j++){
            exp_w.at(i,j)=exp(w.at(i,j));
        }
    }
    for(size_t n=0;n<v_i.rank()*v_j.rank();n++){
        auto max_it=std::max_element(exp_w.e().begin(),exp_w.e().end());
        double max=*max_it;
        size_t argmax=std::distance(exp_w.e().begin(),max_it);
        size_t i=argmax%exp_w.nx();
        size_t j=argmax/exp_w.nx();
        size_t k=std::distance(w_sums.begin(),std::min_element(w_sums.begin(),w_sums.end())); //choose first case of smallest cumulative sum
        w_sums[k]+=max;
        // std::cout<<v_i.vol()<<" "<<v_j.vol()<<" "<<i<<" "<<j<<" "<<k<<" "<<w_sums[k]<<"\n";
        exp_w.at(i,j)=-1; //weights can never be negative so this element will never be the max. also, w is passed by value so no change to original.
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
                    dot_addends.push_back(m_i_m_j_sum[idx]*e_k[idx]);
                }
                dot=vec_add_float(dot_addends);
                c.at(i,j,k)=dot;
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
            for(size_t k=1;k<r_k;k++){
                if((c.at(i,j,k)-(1e-10*w_sums[k]))>max_c){
                    max_c_idx=k;
                    max_c=c.at(i,j,k)-(1e-10*w_sums[k]);
                }
            }
            f_res.at(i,j)=max_c_idx;
            w_sums[max_c_idx]+=exp(w.at(i,j));
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

double optimize::kl_div(double z,double z_prime,std::vector<double> gi,std::vector<double> gj,std::vector<site> sites,bond old_current,std::vector<bond> old_cluster,bond current,std::vector<bond> cluster){
    array2d<double> gi_tilde(sites[old_current.v1()].rank(),sites[old_current.v2()].rank());
    array2d<double> gj_tilde(sites[old_current.v1()].rank(),sites[old_current.v2()].rank());
    for(size_t i=0;i<sites[old_current.v1()].rank();i++){
        for(size_t j=0;j<sites[old_current.v2()].rank();j++){
            double gi_tilde_res=0;
            double gj_tilde_res=0;
            for(size_t n=0;n<old_cluster.size();n++){
                double spin_sum_i_tilde=0;
                double spin_sum_j_tilde=0;
                double div_factor=0;
                if(old_cluster[n].v1()==old_current.v1()||old_cluster[n].v2()==old_current.v1()){ //connected to site i
                    if(old_cluster[n].v1()==old_current.v1()){ //site imu > site i
                        for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                            div_factor+=exp(old_cluster[n].w().at(i,imu));
                            spin_sum_i_tilde+=exp(old_cluster[n].w().at(i,imu))*old_cluster[n].w().at(i,imu);
                        }
                    }
                    else{ //site imu < site i
                        for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                            div_factor+=exp(old_cluster[n].w().at(imu,i));
                            spin_sum_i_tilde+=exp(old_cluster[n].w().at(imu,i))*old_cluster[n].w().at(imu,i);
                        }
                    }
                    gi_tilde_res+=spin_sum_i_tilde/div_factor;
                }
                else{ //connected to site j
                    if(old_cluster[n].v1()==old_current.v2()){ //site imu > site j
                        for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                            div_factor+=exp(old_cluster[n].w().at(j,imu));
                            spin_sum_j_tilde+=exp(old_cluster[n].w().at(j,imu))*old_cluster[n].w().at(j,imu);
                        }
                    }
                    else{ //site imu < site j
                        for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                            div_factor+=exp(old_cluster[n].w().at(imu,j));
                            spin_sum_j_tilde+=exp(old_cluster[n].w().at(imu,j))*old_cluster[n].w().at(imu,j);
                        }
                    }
                    gj_tilde_res+=spin_sum_j_tilde/div_factor;
                }
            }
            gi_tilde.at(i,j)=gi_tilde_res*exp(gi[i]);
            gj_tilde.at(i,j)=gj_tilde_res*exp(gj[j]);
        }
    }
    
    array2d<double> gi_prime_tilde(sites[old_current.v1()].rank(),sites[old_current.v2()].rank());
    array2d<double> gj_prime_tilde(sites[old_current.v1()].rank(),sites[old_current.v2()].rank());
    for(size_t i=0;i<sites[current.v1()].rank();i++){
        for(size_t j=0;j<sites[current.v2()].rank();j++){
            double gi_prime_tilde_res=0;
            double gj_prime_tilde_res=0;
            for(size_t n=0;n<old_cluster.size();n++){
                double spin_sum_i_tilde=0;
                double spin_sum_j_tilde=0;
                double div_factor=0;
                size_t k=current.f().at(i,j);
                if(old_cluster[n].v1()==old_current.v1()||old_cluster[n].v2()==old_current.v1()){ //connected to site i
                    if(old_cluster[n].v1()==old_current.v1()){ //site imu > site i
                        for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                            div_factor+=exp(old_cluster[n].w().at(i,imu));
                            spin_sum_i_tilde+=exp(old_cluster[n].w().at(i,imu))*cluster[n].w().at(imu,k);
                        }
                    }
                    else{ //site imu < site i
                        for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                            div_factor+=exp(old_cluster[n].w().at(imu,i));
                            spin_sum_i_tilde+=exp(old_cluster[n].w().at(imu,i))*cluster[n].w().at(imu,k);
                        }
                    }
                    gi_prime_tilde_res+=spin_sum_i_tilde/div_factor;
                }
                else{ //connected to site j
                    if(old_cluster[n].v1()==old_current.v2()){ //site imu > site j
                        for(size_t imu=0;imu<old_cluster[n].w().ny();imu++){
                            div_factor+=exp(old_cluster[n].w().at(j,imu));
                            spin_sum_j_tilde+=exp(old_cluster[n].w().at(j,imu))*cluster[n].w().at(imu,k);
                        }
                    }
                    else{ //site imu < site j
                        for(size_t imu=0;imu<old_cluster[n].w().nx();imu++){
                            div_factor+=exp(old_cluster[n].w().at(imu,j));
                            spin_sum_j_tilde+=exp(old_cluster[n].w().at(imu,j))*cluster[n].w().at(imu,k);
                        }
                    }
                    gj_prime_tilde_res+=spin_sum_j_tilde/div_factor;
                }
            }
            gi_prime_tilde.at(i,j)=gi_prime_tilde_res*exp(gi[i]);
            gj_prime_tilde.at(i,j)=gj_prime_tilde_res*exp(gj[j]);
        }
    }
    
    double a=0;
    double b=0;
    for(size_t i=0;i<old_current.w().nx();i++){
        for(size_t j=0;j<old_current.w().ny();j++){
            double contrib_a=exp(old_current.w().at(i,j))*exp(gi[i])*exp(gj[j])*old_current.w().at(i,j);
            double contrib_b=exp(old_current.w().at(i,j))*((gi_tilde.at(i,j)*exp(gj[j]))+(exp(gi[i])*gj_tilde.at(i,j)));
            a+=contrib_a;
            b+=contrib_b;
        }
    }
    double a_prime=0;
    double b_prime=0;
    for(size_t i=0;i<current.w().nx();i++){
        for(size_t j=0;j<current.w().ny();j++){
            double contrib_a_prime=exp(old_current.w().at(i,j))*exp(gi[i])*exp(gj[j])*current.w().at(i,j);
            double contrib_b_prime=exp(old_current.w().at(i,j))*((gi_prime_tilde.at(i,j)*exp(gj[j]))+(exp(gi[i])*gj_prime_tilde.at(i,j)));
            a_prime+=contrib_a_prime;
            b_prime+=contrib_b_prime;
        }
    }
    
    
    double res=((a/z)+(b/z)-log(z))-((a_prime/z)+(b_prime/z)-log(z_prime));
    return res;
}