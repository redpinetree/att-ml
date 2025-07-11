/*
Copyright 2025 Katsuya O. Akamatsu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <getopt.h>

#include "omp.h"
#include "sampling.hpp"
#include "stopwatch.hpp"
#include "graph.hpp"
#include "graph_utils.hpp"
#include "algorithm_nll.hpp"
#include "observables.hpp"

std::mt19937_64 prng; //initialize prng

void print_usage(){
    std::cerr<<"usage: att_ml [--options]\n";
    std::cerr<<"options:\n";
    std::cerr<<"\t-h,--help: display this message\n";
    std::cerr<<"\t-v,--verbose:\n\t\t0->nothing printed to stdout\n\t\t1->sample number and aggregate timing data\n\t\t2->per-instance timing\n\t\t3->more detailed timing breakdown\n\t\t4->graph contents, debug observable data\n";
    std::cerr<<"\t-i,--input: REQUIRED -- input file containing training data.\n";
    std::cerr<<"\t-o,--output: prefix for output files. please omit the file extension.\n";
    std::cerr<<"\t-L,--label-file: input file containing training labels\n";
    std::cerr<<"\t-V,--test-file: input file containing test data\n";
    std::cerr<<"\t-B,--test-label-file: input file containing test labels\n";
    std::cerr<<"\t-T,--init-tree-type: initial tree type: mps or pbttn or rand\n";
    std::cerr<<"\t-r,--r-max: maximum rank of spins in the approximation\n";
    std::cerr<<"\t-N,--nll-iter-max: maximum number of NLL optimization iterations\n";
    std::cerr<<"\t-S,--seed: seed for PRNG\n";
    std::cerr<<"\t--born: use double-layer BMATT instead of single-layer nnTTN\n";
    std::cerr<<"\t--hybrid: pretrain with double-layer BMATT then use single-layer nnTTN, with pretrained structure. Overrides --born.\n";
    std::cerr<<"\t--compress-r: enable rank compression\n";
    std::cerr<<"\t--struct-opt: enable structural optimization\n";
}

template<typename cmp>
graph<cmp> gen_graph(int idim,int tdim,int r_max,int num_vs,std::string init_tree_type){
    graph<cmp> g;
    if(init_tree_type=="mps"){
        std::normal_distribution<double> dist(0,0);
        g=graph_utils::init_mps<cmp>(idim,tdim,r_max,num_vs);
    }
    else if(init_tree_type=="pbttn"){
        std::normal_distribution<double> dist{0,0};
        g=graph_utils::init_pbttn<cmp>(idim,tdim,r_max,num_vs);
    }
    else if(init_tree_type=="rand"){
        std::normal_distribution<double> dist{0,0};
        g=graph_utils::init_rand<cmp>(idim,tdim,r_max,num_vs);
    }
    return g;
}

int main(int argc,char **argv){
    //omp init
    // omp_set_num_threads(1);
    //argument handling
    long int seed=0;
    // int seed=(int) std::chrono::system_clock::now().time_since_epoch().count();
    int born_flag=0;
    int hybrid_flag=0;
    int train_type=0;
    int compress_r=0;
    int struct_opt=0;
    int verbose=0;
    bool input_set=false;
    bool output_set=false;
    bool label_set=false;
    bool test_set=false;
    bool test_label_set=false;
    bool tdim_set=false;
    std::string input,output,label_file,test_file,test_label_file;
    std::string init_tree_type="mps";
    int r_max=0;
    int n_config_samples=1000;
    int n_nll_iter_max=10000;
    double lr=0.001;
    int batch_size=1000;
    //option arguments
    while(1){
        static struct option long_opts[]={
            {"born",no_argument,&born_flag,1},
            {"hybrid",no_argument,&hybrid_flag,1},
            {"compress-r",no_argument,&compress_r,1},
            {"struct-opt",no_argument,&struct_opt,1},
            {"help",no_argument,0,'h'},
            {"verbose",required_argument,0,'v'},
            {"input",required_argument,0,'i'},
            {"output",required_argument,0,'o'},
            {"label-file",required_argument,0,'L'},
            {"test-file",required_argument,0,'V'},
            {"test-label-file",required_argument,0,'B'},
            {"init-tree-type",required_argument,0,'T'},
            {"r-max",required_argument,0,'r'},
            {"nll-iter-max",required_argument,0,'N'},
            {"learning-rate",required_argument,0,'l'},
            {"batch_size",required_argument,0,'b'},
            {"seed",required_argument,0,'S'},
            {0, 0, 0, 0}
        };
        int opt_idx=0;
        int c=getopt_long(argc,argv,"hv:i:o:L:V:B:T:r:N:l:b:S:",long_opts,&opt_idx);
        if(c==-1){break;} //end of options
        switch(c){
            //handle long option flags
            case 0:
            break;
            //handle standard options
            case 'h': print_usage(); exit(1);
            case 'v': verbose=(int) atoi(optarg); break;
            case 'i': input=std::string(optarg); input_set=true; break;
            case 'o': output=std::string(optarg); output_set=true; break;
            case 'L': label_file=std::string(optarg); label_set=true; break;
            case 'V': test_file=std::string(optarg); test_set=true; break;
            case 'B': test_label_file=std::string(optarg); test_label_set=true; break;
            case 'T': init_tree_type=std::string(optarg); break;
            case 'r': r_max=(int) atoi(optarg); break;
            case 'N': n_nll_iter_max=(int) atoi(optarg); break;
            case 'l': lr=atof(optarg); break;
            case 'b': batch_size=(int) atoi(optarg); break;
            case 'S': seed=strtoul(optarg,0,16); break;
            case '?':
            //error printed
            exit(1);
            default:
            std::cerr<<"Error parsing arguments. Aborting...\n";
            exit(1);
        }
    }
    //seed prng
    std::cout<<"PRNG seed: "<<seed<<"\n";
    prng.seed(seed);
    //select training type
    if(hybrid_flag){
        train_type=2;
        std::cout<<"Hybrid training type.\n";
    }
    else{
        if(born_flag){
            train_type=1;
            std::cout<<"BMATT training type.\n";
        }
        else{
            train_type=0;
            std::cout<<"nnTTN training type.\n";
        }
    }
    //positional arguments
    if(((argc-optind)>0)||!input_set){
        print_usage();
        exit(1);
    }
    //check if input specified
    if(!input_set){
        std::cerr<<"Error: Must specify input file with training data.\n";
        print_usage();
        exit(1);
    }
    //check if test data specified properly, if needed
    if((!test_set)&&test_label_set){
        std::cerr<<"Error: Must specify input file with test data when specifying test label data.\n";
        print_usage();
        exit(1);
    }
    if(label_set&&test_set&&(!test_label_set)){
        std::cerr<<"Error: Must specify input file with test label data when specifying training label data and test data.\n";
        print_usage();
        exit(1);
    }
    //check initial tree options
    if(!((init_tree_type=="mps")||(init_tree_type=="pbttn")||(init_tree_type=="rand"))){
        std::cerr<<"Error: -T must be one of \"mps\" or \"pbttn\" or \"rand\".\n";
        print_usage();
        exit(1);
    }
    //check if exactly one of top dim and label file is specified
    if(tdim_set&&label_set){
        std::cerr<<"Error: -t cannot be set together with -L.\n";
        print_usage();
        exit(1);
    }
    std::vector<double> times;
    stopwatch sw,sw_total;
    sw_total.start();
    
    std::string output_fn=output+".txt";
    std::string output_fn_born=output+"_born.txt";
    std::string output_fn_hybrid=output+"_hybrid.txt";
    std::string data_fn=output+".dat";
    std::string data_fn_born=output+"_born.dat";
    std::string data_fn_hybrid=output+"_hybrid.dat";
    std::string samples_fn=output+".smp";
    std::string samples_fn_hybrid=output+"_hybrid.smp";
    
    std::stringstream header1_ss,header1_vals_ss,header2_ss,header2_mc_ss;
    observables::output_lines.clear(); //flush output lines
    
    double trial_time=0; //not including init time
    
    int train_data_idim,train_n_samples,train_data_total_length;
    std::vector<std::vector<array1d<double> > > train_data=algorithm::load_data_from_file(input,train_n_samples,train_data_total_length,train_data_idim);
    int test_data_idim,test_n_samples,test_data_total_length;
    std::vector<std::vector<array1d<double> > > test_data;
    if(test_set){
        test_data=algorithm::load_data_from_file(test_file,test_n_samples,test_data_total_length,test_data_idim);
    }
    int train_data_labels_tdim,train_labels_n_samples;
    std::vector<int> train_data_labels;
    if(label_set){
        train_data_labels=algorithm::load_data_labels_from_file(label_file,train_labels_n_samples,train_data_labels_tdim);
    }
    int test_data_labels_tdim,test_labels_n_samples;
    std::vector<int> test_data_labels;
    if(test_label_set){
        test_data_labels=algorithm::load_data_labels_from_file(test_label_file,test_labels_n_samples,test_data_labels_tdim);
    }
    int idim=train_data_idim;
    int tdim=label_set?train_data_labels_tdim:1;
    int num_vs=train_data_total_length;
    graph<bmi_comparator> g=gen_graph<bmi_comparator>(idim,tdim,r_max,num_vs,init_tree_type);
    
    if(r_max==0){
        std::cout<<"Maximum rank not specified/is zero. Will default to r_max=input_dim.\n";
        r_max=idim;
    }
    
    //should be impossible to trigger since these values are based on input file
    if(g.n_phys_sites()!=train_data_total_length){
        std::cout<<"Mismatch in input site count between training data ("<<train_data_total_length<<") and model ("<<g.n_phys_sites()<<").\n";
        exit(1);
    }
    if(idim!=train_data_idim){
        std::cout<<"Mismatch in input dimension between training data ("<<train_data_idim<<") and model ("<<idim<<").\n";
        exit(1);
    }
    //may trigger when test file is incorrectly specified
    if(test_set){
        if(g.n_phys_sites()!=test_data_total_length){
            std::cout<<"Mismatch in input site count between test data ("<<test_data_total_length<<") and model ("<<g.n_phys_sites()<<").\n";
            exit(1);
        }
        if(idim!=train_data_idim){
            std::cout<<"Mismatch in input dimension between test data ("<<test_data_idim<<") and model ("<<idim<<").\n";
            exit(1);
        }
    }
    
    if((train_type==1)||(train_type==2)){
        std::cout<<"Training details (BMATT):\n\ttrain data: "<<input<<"\n";
        std::cout<<"\ttrain data size: "<<train_data.size()<<"\n";
        if(label_set){
            std::cout<<"\ttrain labels: "<<label_file<<"\n";
        }
        if(test_set){
            std::cout<<"\ttest data: "<<test_file<<"\n";
            std::cout<<"\ttest data size: "<<test_data.size()<<"\n";
        }
        if(test_label_set){
            std::cout<<"\ttest labels: "<<test_label_file<<"\n";
        }
        std::cout<<"\tinit tree type: "<<init_tree_type<<"\n";
        std::cout<<"\ttop dim: "<<tdim<<"\n";
        std::cout<<"\tr max: "<<r_max<<"\n";
        std::cout<<"\tinput length: "<<num_vs<<"\n";
        std::cout<<"\tnll iter max: "<<n_nll_iter_max<<"\n";
        std::cout<<"\tlearning rate: "<<lr<<"\n";
        std::cout<<"\tbatch size: "<<batch_size<<"\n";
        std::cout<<"\tstruct opt: "<<(struct_opt?"true":"false")<<"\n";
        std::cout<<"\tcompress r: "<<(compress_r?"true":"false")<<"\n";
        sw.start();
        std::map<int,double> train_nll_history,test_nll_history;
        std::map<int,int> sweep_history;
        
        if(label_set&&test_set&&test_label_set){
            algorithm::train_nll_born(g,train_data,train_data_labels,test_data,test_data_labels,n_nll_iter_max,r_max,compress_r,lr,batch_size,train_nll_history,test_nll_history,sweep_history,struct_opt); //nll training with labels and test data
        }
        else if(label_set&&(!test_set)){
            algorithm::train_nll_born(g,train_data,train_data_labels,n_nll_iter_max,r_max,compress_r,lr,batch_size,train_nll_history,sweep_history,struct_opt); //nll training with labels
        }
        else if((!label_set)&&test_set){
            algorithm::train_nll_born(g,train_data,test_data,n_nll_iter_max,r_max,compress_r,lr,batch_size,train_nll_history,test_nll_history,sweep_history,struct_opt); //nll training with test data
        }
        else{
            algorithm::train_nll_born(g,train_data,n_nll_iter_max,r_max,compress_r,lr,batch_size,train_nll_history,sweep_history,struct_opt); //nll training
        }
        sw.split();
        if(verbose>=3){std::cout<<"nll training time: "<<(double) sw.elapsed()<<"ms\n";}
        trial_time+=sw.elapsed();
        sw.reset();
        
        observables::output_lines.push_back((std::string) g);
        std::string train_nll_string="train nlls: [";
        for(auto it=train_nll_history.begin();it!=train_nll_history.end();++it){
            train_nll_string+="("+std::to_string((*it).first)+", "+std::to_string((*it).second)+")";
            if(it!=--train_nll_history.end()){
                train_nll_string+=", ";
            }
        }
        train_nll_string+="]\n";
        observables::output_lines.push_back(train_nll_string);
        if(test_set){
            std::string test_nll_string="test nlls: [";
            for(auto it=test_nll_history.begin();it!=test_nll_history.end();++it){
                test_nll_string+="("+std::to_string((*it).first)+", "+std::to_string((*it).second)+")";
                if(it!=--test_nll_history.end()){
                    test_nll_string+=", ";
                }
            }
            test_nll_string+="]\n";
            observables::output_lines.push_back(test_nll_string);
        }
        std::string sweep_string="sweeps: [";
        for(auto it=sweep_history.begin();it!=sweep_history.end();++it){
            sweep_string+="("+std::to_string((*it).first)+", "+std::to_string((*it).second)+")";
            if(it!=--sweep_history.end()){
                sweep_string+=", ";
            }
        }
        sweep_string+="]\n";
        observables::output_lines.push_back(sweep_string);
        
        double total_mi=0;
        double total_ee=0;
        for(auto it=g.es().begin();it!=--g.es().end();++it){
            total_mi+=(*it).bmi();
            total_ee+=(*it).ee();
        }
        std::string total_mi_string="total mi: ";
        total_mi_string+=std::to_string(total_mi);
        total_mi_string+="\n";
        observables::output_lines.push_back(total_mi_string);
        std::string total_ee_string="total ee: ";
        total_ee_string+=std::to_string(total_ee);
        total_ee_string+="\n";
        observables::output_lines.push_back(total_ee_string);
        
        // std::vector<sample_data> generated_samples=sampling::tree_sample(g,10);
        // for(int i=0;i<generated_samples.size();i++){
            // for(int j=0;j<generated_samples[i].n_phys_sites();j++){
                // std::cout<<generated_samples[i].s()[j]<<" ";
            // }
            // std::cout<<"\n";
        // }
        //compute output quantities
        // if(verbose>=4){std::cout<<std::string(g);}
        
        if(output_set){
            observables::write_output(output_fn_born,observables::output_lines);
            std::ofstream ofs(data_fn_born,std::ios::binary);
            g.save(ofs);
            ofs.close();
        }
        else{
            observables::write_output(observables::output_lines);
        }
    }
    
    if(train_type==2){ //reset elements in network when doing hybrid training
        observables::output_lines.clear();
        std::uniform_real_distribution<> unif_dist(1e-10,1.0);
        for(auto it=g.es().begin();it!=g.es().end();++it){
            bond b=*it;
            double sum=0;
            for(int i=0;i<b.w().nx();i++){
                for(int j=0;j<b.w().ny();j++){
                    for(int k=0;k<b.w().nz();k++){
                        b.w().at(i,j,k)=unif_dist(prng);
                        sum+=b.w().at(i,j,k);
                    }
                }
            }
            for(int i=0;i<b.w().nx();i++){
                for(int j=0;j<b.w().ny();j++){
                    for(int k=0;k<b.w().nz();k++){
                        b.w().at(i,j,k)/=sum;
                    }
                }
            }
            g.vs()[b.order()].p_bond()=b;
            //use bmi of bmatt, so no need to update bmi
            g.es().erase(it);
            it=g.es().insert(b);
        }
        init_tree_type="bmatt";
    }

    if((train_type==0)||(train_type==2)){
        std::cout<<"Training details (nnTTN):\n\ttrain data: "<<input<<"\n";
        std::cout<<"\ttrain data size: "<<train_data.size()<<"\n";
        if(label_set){
            std::cout<<"\ttrain labels: "<<label_file<<"\n";
        }
        if(test_set){
            std::cout<<"\ttest data: "<<test_file<<"\n";
            std::cout<<"\ttest data size: "<<test_data.size()<<"\n";
        }
        if(test_label_set){
            std::cout<<"\ttest labels: "<<test_label_file<<"\n";
        }
        std::cout<<"\tinit tree type: "<<init_tree_type<<"\n";
        std::cout<<"\ttop dim: "<<tdim<<"\n";
        std::cout<<"\tr max: "<<r_max<<"\n";
        std::cout<<"\tinput length: "<<num_vs<<"\n";
        std::cout<<"\tnll iter max: "<<n_nll_iter_max<<"\n";
        std::cout<<"\tlearning rate: "<<lr<<"\n";
        std::cout<<"\tbatch size: "<<batch_size<<"\n";
        std::cout<<"\tstruct opt: "<<(struct_opt?"true":"false")<<"\n";
        std::cout<<"\tcompress r: "<<(compress_r?"true":"false")<<"\n";
        sw.start();
        std::map<int,double> train_nll_history,test_nll_history;
        std::map<int,int> sweep_history;
        
        if(label_set&&test_set&&test_label_set){
            algorithm::train_nll(g,train_data,train_data_labels,test_data,test_data_labels,n_nll_iter_max,r_max,compress_r,lr,batch_size,train_nll_history,test_nll_history,sweep_history,struct_opt); //nll training with labels and test data
        }
        else if(label_set&&(!test_set)){
            algorithm::train_nll(g,train_data,train_data_labels,n_nll_iter_max,r_max,compress_r,lr,batch_size,train_nll_history,sweep_history,struct_opt); //nll training with labels
        }
        else if((!label_set)&&test_set){
            algorithm::train_nll(g,train_data,test_data,n_nll_iter_max,r_max,compress_r,lr,batch_size,train_nll_history,test_nll_history,sweep_history,struct_opt); //nll training with test data
        }
        else{
            algorithm::train_nll(g,train_data,n_nll_iter_max,r_max,compress_r,lr,batch_size,train_nll_history,sweep_history,struct_opt); //nll training
        }
        sw.split();
        if(verbose>=3){std::cout<<"nll training time: "<<(double) sw.elapsed()<<"ms\n";}
        trial_time+=sw.elapsed();
        sw.reset();
        
        observables::output_lines.push_back((std::string) g);
        std::string train_nll_string="train nlls: [";
        for(auto it=train_nll_history.begin();it!=train_nll_history.end();++it){
            train_nll_string+="("+std::to_string((*it).first)+", "+std::to_string((*it).second)+")";
            if(it!=--train_nll_history.end()){
                train_nll_string+=", ";
            }
        }
        train_nll_string+="]\n";
        observables::output_lines.push_back(train_nll_string);
        if(test_set){
            std::string test_nll_string="test nlls: [";
            for(auto it=test_nll_history.begin();it!=test_nll_history.end();++it){
                test_nll_string+="("+std::to_string((*it).first)+", "+std::to_string((*it).second)+")";
                if(it!=--test_nll_history.end()){
                    test_nll_string+=", ";
                }
            }
            test_nll_string+="]\n";
            observables::output_lines.push_back(test_nll_string);
        }
        std::string sweep_string="sweeps: [";
        for(auto it=sweep_history.begin();it!=sweep_history.end();++it){
            sweep_string+="("+std::to_string((*it).first)+", "+std::to_string((*it).second)+")";
            if(it!=--sweep_history.end()){
                sweep_string+=", ";
            }
        }
        sweep_string+="]\n";
        observables::output_lines.push_back(sweep_string);
        
        double total_mi=0;
        for(auto it=g.es().begin();it!=--g.es().end();++it){
            total_mi+=(*it).bmi();
        }
        std::string total_mi_string="total mi: ";
        total_mi_string+=std::to_string(total_mi);
        total_mi_string+="\n";
        observables::output_lines.push_back(total_mi_string);
        
        std::vector<sample_data> generated_samples=sampling::tree_sample(g,10);
        // for(int i=0;i<generated_samples.size();i++){
            // for(int j=0;j<generated_samples[i].n_phys_sites();j++){
                // std::cout<<generated_samples[i].s()[j]<<" ";
            // }
            // std::cout<<"\n";
        // }
        //compute output quantities
        // if(verbose>=4){std::cout<<std::string(g);}
        
        if(output_set){
            if(train_type==0){
                observables::write_output(output_fn,observables::output_lines);
                std::ofstream ofs(data_fn,std::ios::binary);
                g.save(ofs);
                ofs.close();
            }
            else if(train_type==2){
                observables::write_output(output_fn_hybrid,observables::output_lines);
                std::ofstream ofs(data_fn_hybrid,std::ios::binary);
                g.save(ofs);
                ofs.close();
            }
        }
        else{
            observables::write_output(observables::output_lines);
        }
    }
    
    if(verbose>=2){std::cout<<"Time elapsed: "<<trial_time<<"ms\n";}
    times.push_back(trial_time);
    sw_total.split();
    //time statistics
    double timer_mean=0;
    double timer_std=0;
    for(int i=0;i<times.size();i++){
        timer_mean+=times[i];
    }
    timer_mean/=times.size();
    for(int i=0;i<times.size();i++){
        timer_std+=pow(times[i]-timer_mean,2);
    }
    timer_std/=(times.size()-1);
    timer_std=sqrt(timer_std);
    if(verbose>=1){std::cout<<"Avg. time/trial: "<<timer_mean<<"ms+/-"<<timer_std<<"ms\n";}
    if(verbose>=1){std::cout<<"Total time: "<<(double) sw_total.elapsed()<<"ms\n";}
    return 0;
}