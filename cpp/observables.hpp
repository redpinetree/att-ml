#ifndef OBSERVABLES
#define OBSERVABLES

#include <string>
#include <utility>
#include <vector>

namespace observables{
    extern std::vector<std::string> output_lines;
    void write_output(std::string,std::vector<std::string>&);
    void write_output(std::vector<std::string>&);
    void write_binary_output(std::string,std::vector<std::pair<double,std::vector<double> > >&);
    void write_binary_output(std::vector<std::pair<double,std::vector<double> > >&);
}

#endif
