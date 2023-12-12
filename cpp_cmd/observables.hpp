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
    extern std::map<std::tuple<size_t,size_t,size_t>,std::vector<double> > m_vec_cache;
    extern std::map<size_t,std::vector<std::vector<double> > > m_vec_ref_cache;
    extern std::map<std::tuple<size_t,size_t,size_t,size_t,std::vector<size_t>,std::vector<size_t> >,double> m_known_factors;
    extern std::map<std::tuple<size_t,size_t,size_t,size_t,std::vector<size_t>,std::vector<size_t>,std::vector<double> >,std::complex<double> > m_known_factors_complex;
    extern std::map<std::tuple<size_t,size_t,size_t,size_t,std::vector<size_t>,std::vector<size_t> >,double> q_known_factors;
    extern std::map<std::tuple<size_t,size_t,size_t,size_t,std::vector<size_t>,std::vector<size_t>,std::vector<double> >,std::complex<double> > q_known_factors_complex;
    //calculate real observables
    template<typename cmp>
    std::vector<double> m_vec(graph<cmp>&,size_t,size_t,size_t,size_t); //top-down, for the vector magnetization
    template<typename cmp>
    double m(graph<cmp>&,size_t,size_t,size_t,size_t,std::vector<size_t>,std::vector<size_t>,size_t); //top-down
    template<typename cmp>
    double m(graph<cmp>&,size_t,size_t,size_t,size_t,bool);
    template<typename cmp>
    double m(graph<cmp>&,size_t,size_t,size_t,std::vector<size_t>,std::vector<size_t>); //bottom-up
    template<typename cmp>
    double q(graph<cmp>&,size_t,size_t,size_t,size_t,std::vector<size_t>,std::vector<size_t>,size_t); //top-down
    template<typename cmp>
    double q(graph<cmp>&,size_t,size_t,size_t,size_t,bool);
    template<typename cmp>
    double q(graph<cmp>&,size_t,size_t,size_t,std::vector<size_t>,std::vector<size_t>); //bottom-up
    //calculate complex observables (fourier space)
    template<typename cmp>
    std::complex<double> m(graph<cmp>&,size_t,size_t,size_t,size_t,std::vector<size_t>,std::vector<size_t>,std::vector<double>,size_t); //top-down
    template<typename cmp>
    std::complex<double> m(graph<cmp>&,size_t,size_t,size_t,size_t,std::vector<double>,bool);
    template<typename cmp>
    std::complex<double> m(graph<cmp>&,size_t,size_t,size_t,std::vector<size_t>,std::vector<size_t>,std::vector<double>); //bottom-up
    template<typename cmp>
    std::complex<double> q(graph<cmp>&,size_t,size_t,size_t,size_t,std::vector<size_t>,std::vector<size_t>,std::vector<double>,size_t); //top-down
    template<typename cmp>
    std::complex<double> q(graph<cmp>&,size_t,size_t,size_t,size_t,std::vector<double>,bool);
    template<typename cmp>
    std::complex<double> q(graph<cmp>&,size_t,size_t,size_t,std::vector<size_t>,std::vector<size_t>,std::vector<double>); //bottom-up
    
    template<typename cmp>
    void print_moments(graph<cmp>&,size_t); //debug
    void write_output(std::string,std::vector<std::string>&);
    void write_output(std::vector<std::string>&);
}

#endif
