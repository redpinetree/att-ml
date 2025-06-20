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

#ifndef SAMPLING
#define SAMPLING

#include <string>
#include <complex>
#include <vector>
#include <tuple>
#include <map>

#include "graph.hpp"
#include "ndarray.hpp"

class sample_data{
public:
    sample_data();
    sample_data(int,std::vector<int>);
    int n_phys_sites() const;
    std::vector<int> s() const;
    int& n_phys_sites();
    std::vector<int>& s();
private:
    int n_phys_sites_;
    std::vector<int> s_;
};

namespace sampling{
    template<typename cmp>
    std::vector<sample_data> tree_sample(int,graph<cmp>&,int);
    template<typename cmp>
    std::vector<sample_data> tree_sample(graph<cmp>&,int);
}

#endif
