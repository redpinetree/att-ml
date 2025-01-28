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
#include "mpi.h"
#include "sampling.hpp"
#include "stopwatch.hpp"
#include "mpi_utils.hpp"
#include "graph.hpp"
#include "graph_utils.hpp"
#include "algorithm_nll.hpp"
#include "observables.hpp"

void print_usage(){
    std::cerr<<"usage: tree_ml [--options] <input_dim> <d> <{l|l0,l1,...}>\n";
    std::cerr<<"options:\n";
    std::cerr<<"\t-h,--help: display this message\n";
    std::cerr<<"\t-v,--verbose:\n\t\t0->nothing printed to stdout (forced for MPI)\n\t\t1->sample number and aggregate timing data\n\t\t2->per-instance timing\n\t\t3->more detailed timing breakdown\n\t\t4->graph contents, debug observable data\n";
    std::cerr<<"\t-i,--input: input file containing training data.\n";
    std::cerr<<"\t-o,--output: prefix for output files. please omit the file extension.\n";
    std::cerr<<"\t-t,--top-dim: top dimension\n";
    std::cerr<<"\t-L,--label-file: input file containing training labels\n";
    std::cerr<<"\t-V,--test-file: input file containing test data\n";
    std::cerr<<"\t-B,--test-label-file: input file containing test labels\n";
    std::cerr<<"\t-T,--init-tree-type: initial tree type: mps or pbttn or rand\n";
    std::cerr<<"\t-r,--r-max: maximum rank of spins in the approximation\n";
    std::cerr<<"\t-N,--nll-iter-max: maximum number of NLL optimization iterations\n";
    std::cerr<<"\t--born: use double-layer TTNBM instead of single-layer nnTTN\n";
    std::cerr<<"\t--hybrid: pretrain with double-layer TTNBM then use single-layer nnTTN, with pretrained structure. Overrides --born.\n";
    std::cerr<<"\t--compress-r: enable rank compression\n";
    std::cerr<<"\t--struct-opt: enable structural optimization\n";
}

template<typename cmp>
graph<cmp> gen_graph(size_t idim,size_t tdim,size_t r_max,std::vector<size_t> ls,std::string init_tree_type){
    graph<cmp> g;
    if(init_tree_type=="mps"){
        std::normal_distribution<double> dist(0,0);
        g=graph_utils::init_mps<cmp>(idim,tdim,r_max,ls);
    }
    else if(init_tree_type=="pbttn"){
        std::normal_distribution<double> dist{0,0};
        g=graph_utils::init_pbttn<cmp>(idim,tdim,r_max,ls);
    }
    else if(init_tree_type=="rand"){
        std::normal_distribution<double> dist{0,0};
        g=graph_utils::init_rand<cmp>(idim,tdim,r_max,ls);
    }
    return g;
}

int main(int argc,char **argv){
    //mpi init
    mpi_utils::init();
    //argument handling
    int born_flag=0;
    int hybrid_flag=0;
    int train_type=0;
    int compress_r=0;
    int struct_opt=0;
    size_t verbose=0;
    bool input_set=false;
    bool output_set=false;
    bool label_set=false;
    bool test_set=false;
    bool test_label_set=false;
    bool tdim_set=false;
    std::string input,output,label_file,test_file,test_label_file;
    std::string init_tree_type="mps";
    size_t r_max=0;
    size_t tdim=1;
    size_t n_config_samples=1000;
    size_t n_nll_iter_max=10000;
    double lr=0.001;
    size_t batch_size=1000;
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
            {"top-dim",required_argument,0,'t'},
            {"label-file",required_argument,0,'L'},
            {"test-file",required_argument,0,'V'},
            {"test-label-file",required_argument,0,'B'},
            {"init-tree-type",required_argument,0,'T'},
            {"r-max",required_argument,0,'r'},
            {"nll-iter-max",required_argument,0,'N'},
            {"learning-rate",required_argument,0,'l'},
            {"batch_size",required_argument,0,'b'},
            {0, 0, 0, 0}
        };
        int opt_idx=0;
        int c=getopt_long(argc,argv,"hv:i:o:t:L:V:B:T:r:N:l:b:",long_opts,&opt_idx);
        if(c==-1){break;} //end of options
        switch(c){
            //handle long option flags
            case 0:
            break;
            //handle standard options
            case 'h': print_usage(); exit(1);
            case 'v': verbose=(size_t) atoi(optarg); break;
            case 'i': input=std::string(optarg); input_set=true; break;
            case 'o': output=std::string(optarg); output_set=true; break;
            case 't': tdim=(size_t) atoi(optarg); tdim_set=true; break;
            case 'L': label_file=std::string(optarg); label_set=true; break;
            case 'V': test_file=std::string(optarg); test_set=true; break;
            case 'B': test_label_file=std::string(optarg); test_label_set=true; break;
            case 'T': init_tree_type=std::string(optarg); break;
            case 'r': r_max=(size_t) atoi(optarg); break;
            case 'N': n_nll_iter_max=(size_t) atoi(optarg); break;
            case 'l': lr=atof(optarg); break;
            case 'b': batch_size=(size_t) atoi(optarg); break;
            case '?':
            //error printed
            exit(1);
            default:
            if(mpi_utils::root){std::cerr<<"Error parsing arguments. Aborting...\n";}
            exit(1);
        }
    }
    //select training type
    if(hybrid_flag){
        train_type=2;
        std::cout<<"Hybrid training type.\n";
    }
    else{
        if(born_flag){
            train_type=1;
            std::cout<<"TTNBM training type.\n";
        }
        else{
            train_type=0;
            std::cout<<"nnTTN training type.\n";
        }
    }
    //positional arguments
    if((argc-optind)<2){
        if(mpi_utils::root){print_usage();}
        exit(1);
    }
    size_t idim=(size_t) atoi(argv[optind++]);
    std::vector<size_t> ls;
    //check pos arg counts
    size_t d;
    if((argc-optind)>0){
        d=(size_t) atoi(argv[optind++]);
    }
    else{
        if(mpi_utils::root){print_usage();}
        exit(1);
    }
    if((argc-optind)==1){
        size_t l=(size_t) atoi(argv[optind++]);
        if((init_tree_type=="pbttn")&&(!(((l&(l-1))==0)&&(l!=0)))){
            std::cout<<"Dimensions must be a power of 2.\n";
            exit(1);
        }
        for(size_t i=0;i<d;i++){
            ls.push_back(l);
        }
    }
    else if((argc-optind)==d){
        for(size_t i=0;i<d;i++){
            size_t l=(size_t) atoi(argv[optind++]);
            if((init_tree_type=="pbttn")&&(!(((l&(l-1))==0)&&(l!=0)))){
                std::cout<<"Dimensions must be a power of 2.\n";
                exit(1);
            }
            ls.push_back(l);
        }
    }
    else{
        if(mpi_utils::root){print_usage();}
        exit(1);
    }
    //check if input specified
    if(!input_set){
        if(mpi_utils::root){
            std::cerr<<"Error: Must specify input file with training data.\n";
            print_usage();
        }
        exit(1);
    }
    //check if test data specified properly, if needed
    if((!test_set)&&test_label_set){
        if(mpi_utils::root){
            std::cerr<<"Error: Must specify input file with test data when specifying test label data.\n";
            print_usage();
        }
        exit(1);
    }
    if(label_set&&test_set&&(!test_label_set)){
        if(mpi_utils::root){
            std::cerr<<"Error: Must specify input file with test label data when specifying training label data and test data.\n";
            print_usage();
        }
        exit(1);
    }
    //override verbosity if proc_num>1
    if(mpi_utils::proc_num>1){
        if(mpi_utils::root && verbose>0){
            std::cerr<<"Since MPI is being used with more than 1 process, verbosity forced to 0.\n";
        }
        verbose=0;
    }
    //check initial tree options
    if(!((init_tree_type=="mps")||(init_tree_type=="pbttn")||(init_tree_type=="rand"))){
        if(mpi_utils::root){
            std::cerr<<"Error: -T must be one of \"mps\" or \"pbttn\" or \"rand\".\n";
            print_usage();
        }
        exit(1);
    }
    //check if exactly one of top dim and label file is specified
    if(tdim_set&&label_set){
        if(mpi_utils::root){
            std::cerr<<"Error: -t cannot be set together with -L.\n";
            print_usage();
        }
        exit(1);
    }
    if(r_max==0){
        std::cout<<"Maximum rank not specified/is zero. Will default to r_max=input_dim.\n";
        r_max=idim;
    }
    std::vector<double> times;
    stopwatch sw,sw_total;
    sw_total.start();
    
    std::string output_fn=output+"_born";
    std::string output_fn2=output;
    std::string samples_fn=output+"_samples";
    output_fn+=".txt";
    output_fn2+=".txt";
    samples_fn+=".txt";
    std::stringstream header1_ss,header1_vals_ss,header2_ss,header2_mc_ss;
    observables::output_lines.clear(); //flush output lines
    
    double trial_time=0; //not including init time
    
    size_t train_data_idim,train_n_samples,train_data_total_length;
    std::vector<sample_data> train_data=algorithm::load_data_from_file(input,train_n_samples,train_data_total_length,train_data_idim);
    size_t test_data_idim,test_n_samples,test_data_total_length;
    std::vector<sample_data> test_data;
    if(test_set){
        test_data=algorithm::load_data_from_file(test_file,test_n_samples,test_data_total_length,test_data_idim);
    }
    std::vector<size_t> train_data_labels;
    if(label_set){
        train_data_labels=algorithm::load_data_labels_from_file(label_file,train_n_samples,tdim);
    }
    std::vector<size_t> test_data_labels;
    if(test_label_set){
        train_data_labels=algorithm::load_data_labels_from_file(test_label_file,test_n_samples,tdim);
    }
    graph<bmi_comparator> g=gen_graph<bmi_comparator>(idim,tdim,r_max,ls,init_tree_type);
    
    if(g.n_phys_sites()!=train_data_total_length){
        std::cout<<"Mismatch in input site count between training data ("<<train_data_total_length<<") and model ("<<g.n_phys_sites()<<").\n";
        exit(1);
    }
    if(idim!=train_data_idim){
        std::cout<<"Mismatch in input dimension between training data ("<<train_data_idim<<") and model ("<<idim<<").\n";
        exit(1);
    }
    if(test_set){
        if(g.n_phys_sites()!=test_data_total_length){
            std::cout<<"Mismatch in input site count between test data ("<<test_data_total_length<<") and model ("<<g.n_phys_sites()<<").\n";
            exit(1);
        }
    }
    if((train_type==1)||(train_type==2)){
        std::cout<<"Training details (TTNBM):\n\ttrain data: "<<input<<"\n";
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
        std::cout<<"\tnll iter max: "<<n_nll_iter_max<<"\n";
        std::cout<<"\tlearning rate: "<<lr<<"\n";
        std::cout<<"\tbatch size: "<<batch_size<<"\n";
        std::cout<<"\tstruct opt: "<<(struct_opt?"true":"false")<<"\n";
        std::cout<<"\tcompress r: "<<(compress_r?"true":"false")<<"\n";
        sw.start();
        std::map<size_t,double> train_nll_history,test_nll_history;
        std::map<size_t,size_t> sweep_history;
        
        // std::cout<<"center_idx: "<<g.center_idx()<<"\n";
        // std::cout<<"dz: "<<(std::string)calc_dz_born(g)<<"\n";
        // std::cout<<"z: "<<exp(calc_z_born(g))<<"\n";
        
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
        // for(size_t i=0;i<generated_samples.size();i++){
            // for(size_t j=0;j<generated_samples[i].n_phys_sites();j++){
                // std::cout<<generated_samples[i].s()[j]<<" ";
            // }
            // std::cout<<"\n";
        // }
        //compute output quantities
        // if(verbose>=4){std::cout<<std::string(g);}
        
        if(output_set){
            observables::write_output(output_fn,observables::output_lines);
            // observables::write_output(mc_output_fn,observables::mc_output_lines);
        }
        else{
            observables::write_output(observables::output_lines);
            // observables::write_output(observables::mc_output_lines);
        }
    }
    
    if(train_type==2){ //reset elements in network when doing hybrid training
        observables::output_lines.clear();
        std::uniform_real_distribution<> unif_dist(1e-10,1.0);
        for(auto it=g.es().begin();it!=g.es().end();++it){
            bond b=*it;
            std::vector<double> sum_addends;
            for(size_t i=0;i<b.w().nx();i++){
                for(size_t j=0;j<b.w().ny();j++){
                    for(size_t k=0;k<b.w().nz();k++){
                        b.w().at(i,j,k)=unif_dist(mpi_utils::prng);
                        sum_addends.push_back(b.w().at(i,j,k));
                    }
                }
            }
            double sum=vec_add_float(sum_addends);
            for(size_t i=0;i<b.w().nx();i++){
                for(size_t j=0;j<b.w().ny();j++){
                    for(size_t k=0;k<b.w().nz();k++){
                        b.w().at(i,j,k)/=sum;
                    }
                }
            }
            g.vs()[b.order()].p_bond()=b;
            //use bmi of ttnbm, so no need to update bmi
            g.es().erase(it);
            it=g.es().insert(b);
        }
        init_tree_type="ttnbm";
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
        std::cout<<"\tnll iter max: "<<n_nll_iter_max<<"\n";
        std::cout<<"\tlearning rate: "<<lr<<"\n";
        std::cout<<"\tbatch size: "<<batch_size<<"\n";
        std::cout<<"\tstruct opt: "<<(struct_opt?"true":"false")<<"\n";
        std::cout<<"\tcompress r: "<<(compress_r?"true":"false")<<"\n";
        sw.start();
        std::map<size_t,double> train_nll_history,test_nll_history;
        std::map<size_t,size_t> sweep_history;
        
        // std::cout<<"center_idx: "<<g.center_idx()<<"\n";
        // std::cout<<"dz: "<<(std::string)calc_dz_born(g)<<"\n";
        // std::cout<<"z: "<<exp(calc_z_born(g))<<"\n";
        
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
        for(size_t i=0;i<generated_samples.size();i++){
            for(size_t j=0;j<generated_samples[i].n_phys_sites();j++){
                std::cout<<generated_samples[i].s()[j]<<" ";
            }
            std::cout<<"\n";
        }
        //compute output quantities
        // if(verbose>=4){std::cout<<std::string(g);}
        
        if(output_set){
            observables::write_output(output_fn2,observables::output_lines);
            // observables::write_output(mc_output_fn,observables::mc_output_lines);
        }
        else{
            observables::write_output(observables::output_lines);
            // observables::write_output(observables::mc_output_lines);
        }
    }
    
    if(verbose>=2){std::cout<<"Time elapsed: "<<trial_time<<"ms\n";}
    times.push_back(trial_time);
    sw_total.split();
    //time statistics
    double timer_mean=0;
    double timer_std=0;
    for(size_t i=0;i<times.size();i++){
        timer_mean+=times[i];
    }
    timer_mean/=times.size();
    for(size_t i=0;i<times.size();i++){
        timer_std+=pow(times[i]-timer_mean,2);
    }
    timer_std/=(times.size()-1);
    timer_std=sqrt(timer_std);
    if(verbose>=1){std::cout<<"Avg. time/trial: "<<timer_mean<<"ms+/-"<<timer_std<<"ms\n";}
    MPI_Barrier(MPI_COMM_WORLD);
    if(mpi_utils::root&&(verbose>=1)){std::cout<<"Total time: "<<(double) sw_total.elapsed()<<"ms\n";}
    MPI_Finalize();
    return 0;
}