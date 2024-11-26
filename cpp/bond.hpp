#ifndef BOND
#define BOND

#include <iostream>
#include <string>
#include <utility>

#include "ndarray.hpp"

class bond{
public:
    bond();
    bond(size_t,size_t,size_t,array3d<double>);
    bond(std::pair<size_t,size_t>,size_t,array3d<double>);
    operator std::string() const;
    friend std::ostream& operator<<(std::ostream&,const bond&);
    std::pair<size_t,size_t> v() const;
    std::pair<size_t,size_t> v_orig() const;
    size_t v1() const;
    size_t v2() const;
    size_t v1_orig() const;
    size_t v2_orig() const;
    size_t virt_count() const;
    size_t depth() const;
    array3d<double> w() const;
#ifdef MODEL_CMD
    array2d<size_t> f() const;
#endif
    double bmi() const;
#ifdef MODEL_TREE_ML_BORN
    double ee() const;
#endif
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
    size_t& depth();
    array3d<double>& w();
#ifdef MODEL_CMD
    array2d<size_t>& f();
#endif
    double& bmi();
#ifdef MODEL_TREE_ML_BORN
    double& ee();
#endif
    double& cost();
    size_t& order();
    bool& todo(); //if not yet processed, todo=1
    void bmi(array3d<double>&);
private:
    std::pair<size_t,size_t> v_;
    std::pair<size_t,size_t> v_orig_;
    size_t virt_count_;
    size_t depth_;
    array3d<double> w_;
#ifdef MODEL_CMD
    array2d<size_t> f_;
#endif
    double bmi_;
#ifdef MODEL_TREE_ML_BORN
    double ee_;
#endif
    double cost_;
    size_t order_; //in observable computation, this is the upstream site
    bool todo_;
};

#endif