#include <cmath>

#include "bond.hpp"

bond::bond(size_t q,size_t v1,size_t v2,double j){
    this->q_=q;
    this->v_=(v1<v2)?std::pair<size_t,size_t>(v1,v2):std::pair<size_t,size_t>(v2,v1);
    this->j_=j;
    this->bmi(q,j);
    this->order_=0;
    this->todo_=true;
}

bond::bond(size_t q,std::pair<size_t,size_t> v,double j){
    this->q_=q;
    this->v_=v;
    this->j_=j;
    this->bmi(q,j);
    this->order_=0;
    this->todo_=true;
}

bond::operator std::string() const{
    return "[("+std::to_string(this->v().first)+","+std::to_string(this->v().second)+"),"+std::to_string(this->j())+","+std::to_string(this->bmi())+","+std::to_string(this->order())+","+std::to_string(this->todo())+"]";
}

std::ostream& operator<<(std::ostream& os,const bond& e){
    os<<std::string(e);
    return os;
}

size_t bond::q() const{
    return this->q_;
}

std::pair<size_t,size_t> bond::v() const{
    return this->v_;
}

size_t bond::v1() const{
    return this->v_.first;
}

size_t bond::v2() const{
    return this->v_.second;
}

double bond::j() const{
    return this->j_;
}

double bond::bmi() const{
    return this->bmi_;
}

size_t bond::order() const{
    return this->order_;
}

bool bond::todo() const{
    return this->todo_;
}

size_t& bond::q(){
    return this->q_;
}

std::pair<size_t,size_t>& bond::v(){
    return this->v_;
}

size_t& bond::v1(){
    return this->v_.first;
}

size_t& bond::v2(){
    return this->v_.second;
}

double& bond::j(){
    return this->j_;
}

double& bond::bmi(){
    return this->bmi_;
}

size_t& bond::order(){
    return this->order_;
}

bool& bond::todo(){
    return this->todo_;
}

void bond::bmi(size_t q,double k){
    double z=(q*exp(k))+(q*(q-1));
    this->bmi_=(2*log(q))+(q*((exp(k)*(k-log(z)))-((q-1)*log(z)))/z);
}