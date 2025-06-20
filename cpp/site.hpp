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

#ifndef SITE
#define SITE

#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <utility>

#include "ndarray.hpp"
#include "bond.hpp"
#include "utils.hpp"

class site{
public:
    site();
    site(int,int);
    site(int,int,int,int,int);
    site(int,int,int,std::pair<int,int>);
    operator std::string() const;
    friend std::ostream& operator<<(std::ostream&,const site&);
    int rank() const;
    int vol() const;
    int depth() const;
    std::vector<int> coords() const;
    bool virt() const;
    std::pair<int,int> p() const;
    int l_idx() const;
    int r_idx() const;
    int u_idx() const;
    double bmi() const;
    double ee() const;
    int& rank();
    int& vol();
    int& depth();
    bool& virt();
    int& l_idx();
    int& r_idx();
    int& u_idx();
    double& bmi();
    double& ee();
    bond& p_bond();
    std::vector<double>& p_k();
    array3d<double>& p_ijk();
    array2d<double>& p_ik();
    array2d<double>& p_jk();
    void save(std::ostream& os);
    static site load(std::istream& is);
private:
    bool virt_;
    int rank_;
    int vol_;
    int depth_;
    int l_idx_;
    int r_idx_;
    int u_idx_;
    double bmi_;
    double ee_;
    std::vector<double> p_k_;
    array3d<double> p_ijk_;
    array2d<double> p_ik_;
    array2d<double> p_jk_;
    bond p_bond_;
};

#endif
