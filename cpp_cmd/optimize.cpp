#include <fstream>

#include "mpi_utils.hpp"
#include "optimize.hpp"
#include "observables.hpp"

void optimize::renyi_opt(size_t master,size_t slave,size_t r_k,std::vector<site> sites,bond& old_current,std::vector<bond>& old_cluster,bond& current,std::vector<bond>& cluster,size_t max_it,double lr=0){
    std::uniform_real_distribution<> unif_dist(1e-10,1.0);
    
    double best_cost=1e300;
    // std::cout<<"old_current.w():\n"<<(std::string)old_current.w()<<"\n";
    // std::cout<<"old_current.w() sum:\n"<<old_current.w().sum_over_all()<<"\n";
    bond best_current=current;
    std::vector<bond> best_cluster;
    for(size_t n=0;n<cluster.size();n++){
        // std::cout<<"old_cluster.w(): "<<(std::string)old_cluster[n].w()<<"\n";
        // std::cout<<"old_cluster.w() sum: "<<old_cluster[n].w().sum_over_all()<<"\n";
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
    
    for(size_t restarts=1;restarts<11;restarts++){
        bond trial_current=current;
        std::vector<bond> trial_cluster;
        for(size_t n=0;n<cluster.size();n++){
            trial_cluster.push_back(cluster[n]);
        }
        //reinitialize weight matrix to correct size if needed on first iteration
        array3d<double> new_w(trial_current.w().nx(),trial_current.w().ny(),r_k);
        double sum=0;
        for(size_t i=0;i<new_w.nx();i++){
            for(size_t j=i;j<new_w.ny();j++){
                for(size_t k=0;k<new_w.nz();k++){
                    new_w.at(i,j,k)=unif_dist(mpi_utils::prng);
                    sum+=new_w.at(i,j,k);
                    if(j!=i){
                        new_w.at(j,i,k)=new_w.at(i,j,k);
                        sum+=new_w.at(j,i,k);
                    }
                }
            }
        }
        for(size_t i=0;i<new_w.nx();i++){
            for(size_t j=0;j<new_w.ny();j++){
                for(size_t k=0;k<new_w.nz();k++){
                    new_w.at(i,j,k)/=sum;
                }
            }
        }
        trial_current.w()=new_w;
        // std::cout<<(std::string) new_w<<"\n";
        for(size_t n=0;n<trial_cluster.size();n++){
            array3d<double> new_w(sites[trial_cluster[n].v1()].rank(),r_k,1);
            for(size_t i=0;i<new_w.nx();i++){
                for(size_t j=0;j<new_w.ny();j++){
                    if((i<trial_cluster[n].w().nx())&&(j<trial_cluster[n].w().ny())){
                        new_w.at(i,j,0)=trial_cluster[n].w().at(i,j,0)*(trial_cluster[n].w().nx()*trial_cluster[n].w().ny())/(double) (new_w.nx()*new_w.ny());
                    }
                    else{
                        new_w.at(i,j,0)=1/(double) (new_w.nx()*new_w.ny());
                    }
                }
            }
            double sum=0;
            for(size_t i=0;i<new_w.nx();i++){
                for(size_t j=i;j<new_w.ny();j++){
                    new_w.at(i,j,0)=unif_dist(mpi_utils::prng);
                    sum+=new_w.at(i,j,0);
                    if(j!=i){
                        new_w.at(j,i,0)=new_w.at(i,j,0);
                        sum+=new_w.at(j,i,0);
                    }
                }
            }
            for(size_t i=0;i<new_w.nx();i++){
                for(size_t j=0;j<new_w.ny();j++){
                    new_w.at(i,j,0)/=sum;
                }
            }
            trial_cluster[n].w()=new_w;
            // std::cout<<(std::string) new_w<<"\n";
        }
        
        //nadam variables
        double alpha=0.001;
        double beta1=0.9;
        double beta2=0.999;
        double epsilon=1e-10;
        array3d<double> m_current(trial_current.w().nx(),trial_current.w().ny(),trial_current.w().nz());
        std::vector<array3d<double> > m_cluster;
        for(size_t n=0;n<trial_cluster.size();n++){
            m_cluster.push_back(array3d<double>(trial_cluster[n].w().nx(),trial_cluster[n].w().ny(),trial_cluster[n].w().nz()));
        }
        array3d<double> v_current(trial_current.w().nx(),trial_current.w().ny(),trial_current.w().nz());
        std::vector<array3d<double> > v_cluster;
        for(size_t n=0;n<trial_cluster.size();n++){
            v_cluster.push_back(array3d<double>(trial_cluster[n].w().nx(),trial_cluster[n].w().ny(),trial_cluster[n].w().nz()));
        }
        array3d<double> g_current(trial_current.w().nx(),trial_current.w().ny(),trial_current.w().nz());
        std::vector<array3d<double> > g_cluster;
        for(size_t n=0;n<trial_cluster.size();n++){
            g_cluster.push_back(array3d<double>(trial_cluster[n].w().nx(),trial_cluster[n].w().ny(),trial_cluster[n].w().nz()));
        }
        
        //convergence check variables
        array3d<double> prev_current=trial_current.w();
        std::vector<array3d<double> > prev_cluster;
        for(size_t n=0;n<trial_cluster.size();n++){
            prev_cluster.push_back(trial_cluster[n].w());
        }
        double diff=0;
        double cost=0;
        array3d<double> p_prime_ijk_env(trial_current.w().nx(),trial_current.w().ny(),r_k);
        for(size_t t=0;t<max_it;t++){
            // std::cout<<"old_current.w():\n"<<(std::string)old_current.w()<<"\n";
            //calculate G'_i(S_k) and G'_j(S_k)
            std::vector<double> gi_prime(r_k,1);
            std::vector<double> gj_prime(r_k,1);
            for(size_t k=0;k<r_k;k++){
                std::vector<double> gi_prime_factors;
                std::vector<double> gj_prime_factors;
                for(size_t m=0;m<trial_cluster.size();m++){
                    if((old_cluster[m].v1()==old_current.v1())||(old_cluster[m].v2()==old_current.v1())){ //connected to site i
                        if((trial_cluster[m].w().sum_over_axis(0,2))[k]>1e-10){ //avoid 0 factors
                            gi_prime_factors.push_back((trial_cluster[m].w().sum_over_axis(0,2))[k]);
                        }
                    }
                    else{ //connected to site j
                        if((trial_cluster[m].w().sum_over_axis(0,2))[k]>1e-10){ //avoid 0 factors
                            gj_prime_factors.push_back((trial_cluster[m].w().sum_over_axis(0,2))[k]);
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
            array3d<double> fi_prime(trial_current.w().nx(),r_k,r_k);
            array3d<double> fj_prime(trial_current.w().ny(),r_k,r_k);
            for(size_t k=0;k<r_k;k++){
                for(size_t k2=0;k2<r_k;k2++){
                    std::vector<std::vector<double> > fi_prime_factors;
                    std::vector<std::vector<double> > fj_prime_factors;
                    for(size_t i=0;i<trial_current.w().nx();i++){
                        fi_prime_factors.push_back(std::vector<double>());
                    }
                    for(size_t j=0;j<trial_current.w().ny();j++){
                        fj_prime_factors.push_back(std::vector<double>());
                    }
                    for(size_t m=0;m<trial_cluster.size();m++){
                        if((old_cluster[m].v1()==old_current.v1())||(old_cluster[m].v2()==old_current.v1())){ //connected to site i
                            for(size_t i=0;i<old_current.w().nx();i++){
                                double sum=0;
                                for(size_t imu=0;imu<trial_cluster[m].w().nx();imu++){
                                    double num=trial_cluster[m].w().at(imu,k,0)*trial_cluster[m].w().at(imu,k2,0);
                                    double denom;
                                    if(old_cluster[m].v1()==trial_current.v1()){ //site imu > site i
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
                                for(size_t imu=0;imu<trial_cluster[m].w().nx();imu++){
                                    double num=trial_cluster[m].w().at(imu,k,0)*trial_cluster[m].w().at(imu,k2,0);
                                    double denom;
                                    if(old_cluster[m].v1()==trial_current.v2()){ //site imu > site j
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
                    for(size_t i=0;i<trial_current.w().nx();i++){
                        fi_prime.at(i,k,k2)=vec_mult_float(fi_prime_factors[i]);
                    }
                    for(size_t j=0;j<trial_current.w().ny();j++){
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
                        sum_p_prime_ijk_env_addends.push_back(p_prime_ijk_env.at(i,j,k)*trial_current.w().at(i,j,k)); //restore divided factor to get z'
                    }
                }
            }
            //normalize P'^{env}_{ij} by Z' (as sum_p_prime_ijk_env)
            double sum_p_prime_ijk_env=vec_add_float(sum_p_prime_ijk_env_addends);
            for(size_t i=0;i<p_prime_ijk_env.nx();i++){
                for(size_t j=0;j<p_prime_ijk_env.ny();j++){
                    for(size_t k=0;k<p_prime_ijk_env.nz();k++){
                        p_prime_ijk_env.at(i,j,k)/=sum_p_prime_ijk_env;
                    }
                }
            }
            // std::cout<<"current:"<<(std::string)trial_current.w()<<"\n";
            // std::cout<<"p_prime_ijk_env:"<<(std::string)p_prime_ijk_env<<"\n";
            // std::cout<<"z':"<<sum_p_prime_ijk_env<<"\n";
            //calculate intermediate factors (and cost function as sum_ij_factors in the process)
            array3d<double> ij_factors(p_prime_ijk_env.nx(),p_prime_ijk_env.ny(),p_prime_ijk_env.nz());
            double sum_ij_factors=0;
            for(size_t i=0;i<p_prime_ijk_env.nx();i++){
                for(size_t j=0;j<p_prime_ijk_env.ny();j++){
                    for(size_t k=0;k<p_prime_ijk_env.nz();k++){
                        std::vector<double> k2_sum_addends;
                        for(size_t k2=0;k2<p_prime_ijk_env.nz();k2++){
                            k2_sum_addends.push_back(trial_current.w().at(i,j,k2)*fi_prime.at(i,k,k2)*fj_prime.at(j,k,k2));
                        }
                        ij_factors.at(i,j,k)=(trial_current.w().at(i,j,k)/old_current.w().at(i,j,0))*vec_add_float(k2_sum_addends);
                        ij_factors.at(i,j,k)*=(z/sum_p_prime_ijk_env)/sum_p_prime_ijk_env; //cancelled out in later division but needed in gradient descent
                        sum_ij_factors+=ij_factors.at(i,j,k);
                    }
                }
            }
            cost=sum_ij_factors;
            // std::cout<<"ij_factors:"<<(std::string)ij_factors<<"\n";
            // std::cout<<"sum_ij_factors:"<<sum_ij_factors<<"\n";
            
            double sum=0;
            for(size_t i=0;i<p_prime_ijk_env.nx();i++){
                for(size_t j=0;j<p_prime_ijk_env.ny();j++){
                    for(size_t k=0;k<p_prime_ijk_env.nz();k++){
                        if(lr==0){ //lr==0 means use iterative method based on stationarity condition
                            if(fabs(p_prime_ijk_env.at(i,j,k))>1e-10){
                                trial_current.w().at(i,j,k)=ij_factors.at(i,j,k)/(sum_ij_factors*p_prime_ijk_env.at(i,j,k));
                                if(trial_current.w().at(i,j,k)<1e-10){
                                    trial_current.w().at(i,j,k)=1e-10;
                                }
                            }
                        }
                        else{ //lr!=0 means use gradient descent with lr
                            if(fabs(trial_current.w().at(i,j,k))>1e-10){
                                // trial_current.w().at(i,j,k)-=lr*2*((ij_factors.at(i,j,k)/trial_current.w().at(i,j,k))-(p_prime_ijk_env.at(i,j,k)*sum_ij_factors));
                                double grad=2*((ij_factors.at(i,j,k)/trial_current.w().at(i,j,k))-(p_prime_ijk_env.at(i,j,k)*sum_ij_factors));
                                grad*=2*sqrt(trial_current.w().at(i,j,k)); //get gradient of sqrt(w_ijk)
                                g_current.at(i,j,k)=grad;
                                m_current.at(i,j,k)=(beta1*m_current.at(i,j,k))+((1-beta1)*g_current.at(i,j,k));
                                v_current.at(i,j,k)=(beta2*v_current.at(i,j,k))+((1-beta2)*pow(g_current.at(i,j,k),2.0));
                                double bias_corrected_m=m_current.at(i,j,k)/(1-pow(beta1,(double) t+1));
                                double bias_corrected_v=v_current.at(i,j,k)/(1-pow(beta2,(double) t+1));
                                // trial_current.w().at(i,j,k)=pow(sqrt(trial_current.w().at(i,j,k))-(alpha*(bias_corrected_m/(sqrt(bias_corrected_v)+epsilon))),2.0);
                                trial_current.w().at(i,j,k)=pow(((1-(alpha*0.01))*sqrt(trial_current.w().at(i,j,k)))-(alpha*(bias_corrected_m/(sqrt(bias_corrected_v)+epsilon))),2.0);
                                // trial_current.w().at(i,j,k)=pow(sqrt(trial_current.w().at(i,j,k))-(lr*grad),2.0);
                                if(trial_current.w().at(i,j,k)<1e-10){ //in case the weight is negative, force it to be nonnegative!
                                    trial_current.w().at(i,j,k)=1e-10;
                                }
                                sum+=trial_current.w().at(i,j,k);
                            }
                        }
                    }
                }
            }
            if(lr!=0){ //lr==0 means use iterative method based on stationarity condition
                for(size_t i=0;i<p_prime_ijk_env.nx();i++){
                    for(size_t j=0;j<p_prime_ijk_env.ny();j++){
                        for(size_t k=0;k<p_prime_ijk_env.nz();k++){
                            trial_current.w().at(i,j,k)/=sum;
                            if(trial_current.w().at(i,j,k)<1e-10){ //in case the weight is negative, force it to be nonnegative!
                                trial_current.w().at(i,j,k)=1e-10;
                            }
                        }
                    }
                }
            }
            trial_current.bmi(trial_current.w());
            // std::cout<<"updated current:"<<(std::string)trial_current.w()<<"\n";
            
            //NOTE: right now, this uses a workaround for floating-point non-associativity, which involves storing addends in vectors and then sorting them before adding.
            //p_prime_ki,imu if source==trial_current.v1(), p_prime_kj,jnu if source==trial_current.v2(). wlog, use p_prime_ki and imu.
            //k, the new index, is always second because virtual indices > physical indices.a
            for(size_t n=0;n<trial_cluster.size();n++){
                //determine whether this bond was connected to site i or to site j
                size_t source;
                if(old_cluster[n].v1()==master||old_cluster[n].v1()==slave){
                    source=(old_cluster[n].v1()==master)?master:slave;
                }
                else{
                    source=(old_cluster[n].v2()==master)?master:slave;
                }
                
                array3d<double> p_prime_ki_env(trial_cluster[n].w().nx(),trial_cluster[n].w().ny(),1);
                trial_cluster[n].bmi(trial_cluster[n].w());
                
                //calculate G'_i(S_k) and G'_j(S_k)
                std::vector<double> gi_prime(r_k,1);
                std::vector<double> gj_prime(r_k,1);
                for(size_t k=0;k<r_k;k++){
                    std::vector<double> gi_prime_factors;
                    std::vector<double> gj_prime_factors;
                    for(size_t m=0;m<trial_cluster.size();m++){
                        if((old_cluster[m].v1()==old_current.v1())||(old_cluster[m].v2()==old_current.v1())){ //connected to site i
                            if((trial_cluster[m].w().sum_over_axis(0,2))[k]>1e-10){ //avoid 0 factors
                                gi_prime_factors.push_back((trial_cluster[m].w().sum_over_axis(0,2))[k]);
                            }
                        }
                        else{ //connected to site j
                            if((trial_cluster[m].w().sum_over_axis(0,2))[k]>1e-10){ //avoid 0 factors
                                gj_prime_factors.push_back((trial_cluster[m].w().sum_over_axis(0,2))[k]);
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
                array3d<double> fi_prime(trial_current.w().nx(),r_k,r_k);
                array3d<double> fj_prime(trial_current.w().ny(),r_k,r_k);
                for(size_t m=0;m<trial_cluster.size();m++){
                    if((old_cluster[m].v1()==old_current.v1())||(old_cluster[m].v2()==old_current.v1())){ //connected to site i
                        f_factors.push_back(array3d<double>(trial_current.w().nx(),r_k,r_k));
                    }
                    else{ //connected to site j
                        f_factors.push_back(array3d<double>(trial_current.w().ny(),r_k,r_k));
                    }
                }
                for(size_t k=0;k<r_k;k++){
                    for(size_t k2=0;k2<r_k;k2++){
                        std::vector<std::vector<double> > fi_prime_factors;
                        std::vector<std::vector<double> > fj_prime_factors;
                        for(size_t i=0;i<trial_current.w().nx();i++){
                            fi_prime_factors.push_back(std::vector<double>());
                        }
                        for(size_t j=0;j<trial_current.w().ny();j++){
                            fj_prime_factors.push_back(std::vector<double>());
                        }
                        for(size_t m=0;m<trial_cluster.size();m++){
                            if((old_cluster[m].v1()==old_current.v1())||(old_cluster[m].v2()==old_current.v1())){ //connected to site i
                                for(size_t i=0;i<old_current.w().nx();i++){
                                    double sum=0;
                                    for(size_t imu=0;imu<trial_cluster[m].w().nx();imu++){
                                        double num=trial_cluster[m].w().at(imu,k,0)*trial_cluster[m].w().at(imu,k2,0);
                                        double denom;
                                        if(old_cluster[m].v1()==trial_current.v1()){ //site imu > site i
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
                                    for(size_t imu=0;imu<trial_cluster[m].w().nx();imu++){
                                        double num=trial_cluster[m].w().at(imu,k,0)*trial_cluster[m].w().at(imu,k2,0);
                                        double denom;
                                        if(old_cluster[m].v1()==trial_current.v2()){ //site imu > site j
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
                        for(size_t i=0;i<trial_current.w().nx();i++){
                            fi_prime.at(i,k,k2)=vec_mult_float(fi_prime_factors[i]);
                        }
                        for(size_t j=0;j<trial_current.w().ny();j++){
                            fj_prime.at(j,k,k2)=vec_mult_float(fj_prime_factors[j]);
                        }
                    }
                }
                // std::cout<<"fi_prime:"<<(std::string)fi_prime<<"\n";
                // std::cout<<"fj_prime:"<<(std::string)fj_prime<<"\n";
                //calculate P'^{env}_{ki_\mu} and Z' (as sum_p_prime_ki_env)
                std::vector<std::vector<double> > addends_env(p_prime_ki_env.nx()*p_prime_ki_env.ny());
                std::vector<double> sum_p_prime_ki_env_addends;
                for(size_t i=0;i<trial_current.w().nx();i++){
                    for(size_t j=0;j<trial_current.w().ny();j++){
                        for(size_t k=0;k<p_prime_ki_env.ny();k++){
                            for(size_t imu=0;imu<p_prime_ki_env.nx();imu++){
                                std::vector<double> factors_env;
                                factors_env.push_back(trial_current.w().at(i,j,k));
                                factors_env.push_back(gi_prime[k]);
                                factors_env.push_back(gj_prime[k]);
                                factors_env.push_back(1/(trial_cluster[n].w().sum_over_axis(0,2))[k]); //divide appropriate factor
                                addends_env[(k*p_prime_ki_env.nx())+imu].push_back(vec_mult_float(factors_env));
                                sum_p_prime_ki_env_addends.push_back(vec_mult_float(factors_env)*trial_cluster[n].w().at(imu,k,0)); //restore divided factor to get z'
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
                //normalize P'^{env}_{ki_\mu} by Z'
                for(size_t imu=0;imu<p_prime_ki_env.nx();imu++){
                    for(size_t k=0;k<p_prime_ki_env.ny();k++){
                        p_prime_ki_env.at(imu,k,0)/=sum_p_prime_ki_env;
                    }
                }
                // std::cout<<"cluster[n]: "<<(std::string) trial_cluster[n].w()<<"\n";
                // std::cout<<"p_prime_ki_env: "<<(std::string) p_prime_ki_env<<"\n";
                // std::cout<<"z': "<<sum_p_prime_ki_env<<"\n";
                //calculate intermediate factors (and cost function as sum_ki_factors in the process)
                array2d<double> ki_factors(p_prime_ki_env.nx(),p_prime_ki_env.ny());
                double sum_ki_factors=0;
                for(size_t k=0;k<p_prime_ki_env.ny();k++){
                    for(size_t imu=0;imu<p_prime_ki_env.nx();imu++){
                        std::vector<double> i2j2k2_sum_addends;
                        for(size_t i2=0;i2<trial_current.w().nx();i2++){
                            for(size_t j2=0;j2<trial_current.w().ny();j2++){
                                for(size_t k2=0;k2<p_prime_ki_env.ny();k2++){
                                    double res=(trial_current.w().at(i2,j2,k2)*trial_current.w().at(i2,j2,k)/old_current.w().at(i2,j2,0))*trial_cluster[n].w().at(imu,k2,0);
                                    double restored_f_factor;
                                    if(source==trial_current.v1()){ //connected to site i
                                        restored_f_factor=f_factors[n].at(i2,k,k2);
                                        if(old_cluster[n].v1()==trial_current.v1()){ //site imu > site i
                                            res/=old_cluster[n].w().at(i2,imu,0);
                                        }
                                        else{ //site imu < site i
                                            res/=old_cluster[n].w().at(imu,i2,0);
                                        }
                                    }
                                    else{ //connected to site j
                                        restored_f_factor=f_factors[n].at(j2,k,k2);
                                        if(old_cluster[n].v1()==trial_current.v2()){ //site imu > site j
                                            res/=old_cluster[n].w().at(j2,imu,0);
                                        }
                                        else{ //site imu < site j
                                            res/=old_cluster[n].w().at(imu,j2,0);
                                        }
                                    }
                                    res*=fi_prime.at(i2,k,k2)*fj_prime.at(j2,k,k2)/restored_f_factor;
                                    i2j2k2_sum_addends.push_back(res);
                                }
                            }
                        }
                        ki_factors.at(imu,k)=trial_cluster[n].w().at(imu,k,0)*vec_add_float(i2j2k2_sum_addends);
                        ki_factors.at(imu,k)*=(z/sum_p_prime_ki_env)/sum_p_prime_ki_env; //cancelled out in later division
                        sum_ki_factors+=ki_factors.at(imu,k);
                    }
                }
                // std::cout<<"ki_factors:"<<(std::string)ki_factors<<"\n";
                // std::cout<<"sum_ki_factors:"<<sum_ki_factors<<"\n";
                cost=sum_ki_factors;
                
                double sum=0;
                for(size_t imu=0;imu<p_prime_ki_env.nx();imu++){
                    for(size_t k=0;k<p_prime_ki_env.ny();k++){
                        if(lr==0){ //lr==0 means use iterative method based on stationarity condition
                            if(fabs(p_prime_ki_env.at(imu,k,0))>1e-10){
                                trial_cluster[n].w().at(imu,k,0)=ki_factors.at(imu,k)/(sum_ki_factors*p_prime_ki_env.at(imu,k,0));
                                if(trial_cluster[n].w().at(imu,k,0)<1e-10){
                                    trial_cluster[n].w().at(imu,k,0)=1e-10;
                                }
                            }
                        }
                        else{ //lr!=0 means use gradient descent with lr
                            if(fabs(trial_cluster[n].w().at(imu,k,0))>1e-10){
                                // trial_cluster[n].w().at(imu,k,0)-=lr*2*((ki_factors.at(imu,k)/trial_cluster[n].w().at(imu,k,0))-(p_prime_ki_env.at(imu,k,0)*sum_ki_factors));
                                double grad=2*((ki_factors.at(imu,k)/trial_cluster[n].w().at(imu,k,0))-(p_prime_ki_env.at(imu,k,0)*sum_ki_factors));
                                grad*=2*sqrt(trial_cluster[n].w().at(imu,k,0)); //get gradient of sqrt(w_kimu)
                                g_cluster[n].at(imu,k,0)=grad;
                                m_cluster[n].at(imu,k,0)=(beta1*m_cluster[n].at(imu,k,0))+((1-beta1)*g_cluster[n].at(imu,k,0));
                                v_cluster[n].at(imu,k,0)=(beta2*v_cluster[n].at(imu,k,0))+((1-beta2)*pow(g_cluster[n].at(imu,k,0),2.0));
                                double bias_corrected_m=m_cluster[n].at(imu,k,0)/(1-pow(beta1,(double) t+1));
                                double bias_corrected_v=v_cluster[n].at(imu,k,0)/(1-pow(beta2,(double) t+1));
                                // trial_cluster[n].w().at(imu,k,0)=pow(sqrt(trial_cluster[n].w().at(imu,k,0))-(alpha*(bias_corrected_m/(sqrt(bias_corrected_v)+epsilon))),2.0);
                                trial_cluster[n].w().at(imu,k,0)=pow(((1-(alpha*0.01))*sqrt(trial_cluster[n].w().at(imu,k,0)))-(alpha*(bias_corrected_m/(sqrt(bias_corrected_v)+epsilon))),2.0);
                                // trial_cluster[n].w().at(imu,k,0)=pow(sqrt(trial_cluster[n].w().at(imu,k,0))-(lr*grad),2.0);
                                if(trial_cluster[n].w().at(imu,k,0)<1e-10){ //in case the weight is negative, force it to be nonnegative!
                                    trial_cluster[n].w().at(imu,k,0)=1e-10;
                                }
                                sum+=trial_cluster[n].w().at(imu,k,0);
                            }
                        }
                    }
                }
                if(lr!=0){ //lr==0 means use iterative method based on stationarity condition
                    for(size_t imu=0;imu<p_prime_ki_env.nx();imu++){
                        for(size_t k=0;k<p_prime_ki_env.ny();k++){
                            trial_cluster[n].w().at(imu,k,0)/=sum;
                            if(trial_cluster[n].w().at(imu,k,0)<1e-10){
                                trial_cluster[n].w().at(imu,k,0)=1e-10;
                            }
                        }
                    }
                }
                // std::cout<<"updated cluster[n]: "<<(std::string) trial_cluster[n].w()<<"\n";
                trial_cluster[n].bmi(trial_cluster[n].w());
            }
            
            //check for convergence if after first iteration (since trial_current must be resized)
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
                        for(size_t k=0;k<trial_current.w().nz();k++){
                            diff+=fabs(trial_current.w().at(i,j,k)-prev_current.at(i,j,k));
                        }
                    }
                }
                for(size_t n=0;n<trial_cluster.size();n++){
                    for(size_t i=0;i<trial_cluster[n].w().nx();i++){
                        for(size_t j=0;j<trial_cluster[n].w().ny();j++){
                            diff+=fabs(trial_cluster[n].w().at(i,j,0)-prev_cluster[n].at(i,j,0));
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