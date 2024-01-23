#ifndef BOND
#define BOND

#include <iostream>
#include <string>
#include <utility>

#include "../ndarray.hpp"

class bond{
public:
    bond();
    bond(size_t,size_t,array3d<double>);
    bond(std::pair<size_t,size_t>,array3d<double>);
    operator std::string() const;
    friend std::ostream& operator<<(std::ostream&,const bond&);
    std::pair<size_t,size_t> v() const;
    std::pair<size_t,size_t> v_orig() const;
    size_t v1() const;
    size_t v2() const;
    size_t v1_orig() const;
    size_t v2_orig() const;
    size_t virt_count() const;
    array3d<double> w() const;
    double bmi() const;
    double cost() const;
    size_t order() const;
    bool todo() const;
    std::pair<size_t,size_t>& v();
    std::pair<size_t,size_t>& v_orig();
    size_t& v1();
    size_t& v2();
    size_t& v1_orig();
    size_t& v2_orig();
    size_t& virt_count();
    array3d<double>& w();
    double& bmi();
    double& cost();
    size_t& order();
    bool& todo();
    void bmi(array3d<double>&);
private:
    std::pair<size_t,size_t> v_;
    std::pair<size_t,size_t> v_orig_;
    size_t virt_count_;
    array3d<double> w_; //if not yet processed, todo=1
    double bmi_;
    double cost_;
    size_t order_; //in observable computation, this is the upstream site
    bool todo_;
};

#endif