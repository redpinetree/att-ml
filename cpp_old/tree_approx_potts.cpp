#include <iostream>
#include <string>
#include <random>
#include <chrono>
#include <getopt.h>

#include "stopwatch.hpp"
#include "graph.hpp"
#include "graph_utils.hpp"
#include "algorithm.hpp"

void print_usage(){
    std::cerr<<"usage: tree_approx_potts_old [--options] <q> <n_samples> <d> <{l|l1,l2,...}> <min_beta> <max_beta> <step_beta>\n";
    std::cerr<<"usage: tree_approx_potts_old [--options] <q> 0 <min_beta> <max_beta> <step_beta>\n";
    std::cerr<<"\tif <n_samples>!=0, required: -d, -1, -2, forbidden: -i,\n";
    std::cerr<<"\tif <n_samples>==0, reguired: -i, forbidden: -d, -1, -2,\n";
}

int main(int argc,char **argv){
    //argument handling
    int sort_by_coupling=0;
    int open_bc=0;
    size_t verbose=0;
    bool input_set=false;
    bool dist_set=false;
    bool dist_param1_set=false;
    bool dist_param2_set=false;
    std::string input,dist;
    double dist_param1,dist_param2;
    //option arguments
    while(1){
        static struct option long_opts[]={
            {"sort-by-coupling",no_argument,&sort_by_coupling,1},
            {"open-bc",no_argument,&open_bc,1},
            {"verbose",required_argument,0,'v'},
            {"input",required_argument,0,'i'},
            {"distribution",required_argument,0,'d'},
            {"dist_param1",required_argument,0,'1'},
            {"dist_param2",required_argument,0,'2'},
            {0, 0, 0, 0}
        };
        int opt_idx=0;
        int c=getopt_long(argc,argv,"v:i:d:1:2:",long_opts,&opt_idx);
        if(c==-1){break;} //end of options
        switch(c){
            //handle long option flags
            case 0:
            break;
            //handle standard options
            case 'v': verbose=(size_t) atoi(optarg); break;
            case 'i': input=std::string(optarg); input_set=true; break;
            case 'd': dist=std::string(optarg); dist_set=true; break;
            case '1': dist_param1=(double) atof(optarg); dist_param1_set=true; break;
            case '2': dist_param2=(double) atof(optarg); dist_param2_set=true; break;
            case '?':
            //error printed
            exit(1);
            default:
            std::cerr<<"Error parsing arguments. Aborting...\n";
            exit(1);
        }
    }
    //positional arguments
    if((argc-optind)<2){
        print_usage();
        exit(1);
    }
    size_t q=(size_t) atoi(argv[optind++]);
    size_t n_samples=(size_t) atoi(argv[optind++]);
    std::vector<size_t> ls;
    if(n_samples==0){ //input file mode
        if((argc-optind)!=3){
            print_usage();
            exit(1);
        }
    }
    else{
        //check pos arg counts
        size_t d;
        if((argc-optind)>0){
            d=(size_t) atoi(argv[optind++]);
        }
        else{
            print_usage();
            exit(1);
        }
        if((argc-optind)==4){
            size_t l=(size_t) atoi(argv[optind++]);
            for(size_t i=0;i<d;i++){
                ls.push_back(l);
            }
        }
        else if((argc-optind)==(d+3)){
            for(size_t i=0;i<d;i++){
                ls.push_back((size_t) atoi(argv[optind++]));
            }
        }
        else{
            print_usage();
            exit(1);
        }
    }
    double min_beta=(double) atof(argv[optind++]);
    double max_beta=(double) atof(argv[optind++]);
    double step_beta=(double) atof(argv[optind++]);
    //check presence of input file/dist options
    if(n_samples==0){
        if((!input_set)||dist_set){
            std::cerr<<"Error: if <n_samples> is 0, -i must be supplied, while -d must not be supplied.\n";
            print_usage();
            exit(1);
        }
    }
    if(n_samples!=0){
        if(input_set||(!dist_set)){
            std::cerr<<"Error: if <n_samples> is not 0, -d must be supplied, while -i must not be supplied.\n";
            print_usage();
            exit(1);
        }
    }
    if(dist_set){
        if(!(dist_param1_set&&dist_param2_set)){
            std::cerr<<"Error: if -d is supplied, -1 and -2 must also be supplied.\n";
            print_usage();
            exit(1);
        }
    }
    graph_old g;
    if(input_set){
        g=graph_utils::load_graph_old(input,q);
    }
    else{
        //transformations are done to counteract the transformations in gen_hypercubic
        if(dist=="gaussian"){
            //dist_param1=mean, dist_param2=std
            std::normal_distribution<double> dist((dist_param1+1)/2,dist_param2/2);
            g=graph_utils::gen_hypercubic_old(q,ls,!open_bc,dist);
        }
        else if(dist=="bimodal"){
            //dist_param1=p, dist_param2=N/A
            std::discrete_distribution<int> dist{1-dist_param1,dist_param1};
            g=graph_utils::gen_hypercubic_old(q,ls,!open_bc,dist);
        }
        else if(dist=="uniform"){
            //dist_param1=min, dist_param2=max
            std::uniform_real_distribution<double> dist{(dist_param1+1)/2,(dist_param2+1)/2};
            g=graph_utils::gen_hypercubic_old(q,ls,!open_bc,dist);
        }
        else{
            std::cerr<<"Error: -d must be one of \"gaussian\", \"bimodal\", or \"uniform\".\n";
            print_usage();
            exit(1);
        }
    }
    stopwatch sw;
    sw.start();
    algorithm::tree_approx_old(q,g,sort_by_coupling);
    sw.split();
    std::cout<<std::string(g);
    std::cout<<(double) (sw.elapsed()/1e3)<<"ms \n";
    sw.reset();
}