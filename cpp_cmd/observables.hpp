#ifndef OBSERVABLES
#define OBSERVABLES

#include <string>
#include <complex>
#include <vector>
#include <tuple>
#include <map>

#include "graph.hpp"
#include "ndarray.hpp"

namespace observables{
    extern std::vector<std::string> output_lines;
    extern std::vector<std::vector<double> > probs;
    extern std::map<std::tuple<size_t,size_t,size_t,std::vector<size_t> >,double> known_factors;
    extern std::map<std::tuple<size_t,size_t,size_t,std::vector<size_t>,std::vector<double> >,std::complex<double> > known_factors_complex;
    template<typename cmp>
    void cmd_treeify(graph<cmp>&);
    //calculate real observables
    template<typename cmp>
    double m(graph<cmp>&,size_t,size_t,size_t,size_t);
    template<typename cmp>
    double m(graph<cmp>&,size_t,size_t,size_t,std::vector<size_t>,size_t); //top-down
    template<typename cmp>
    double m(graph<cmp>&,size_t,size_t,size_t);
    template<typename cmp>
    double m(graph<cmp>&,size_t,size_t,std::vector<size_t>); //bottom-up
    //calculate complex observables (fourier space)
    template<typename cmp>
    std::complex<double> m(graph<cmp>&,size_t,size_t,size_t,size_t,std::vector<double>);
    template<typename cmp>
    std::complex<double> m(graph<cmp>&,size_t,size_t,size_t,std::vector<size_t>,std::vector<double>,size_t); //top-down
    template<typename cmp>
    std::complex<double> m(graph<cmp>&,size_t,size_t,size_t,std::vector<double>);
    template<typename cmp>
    std::complex<double> m(graph<cmp>&,size_t,size_t,std::vector<size_t>,std::vector<double>); //bottom-up
    
    template<typename cmp>
    void print_moments(graph<cmp>&,size_t,size_t); //debug
    void write_output(std::string,std::vector<std::string>&);
    void write_output(std::vector<std::string>&);
    
    
}

#endif
