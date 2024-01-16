#ifndef STOPWATCH
#define STOPWATCH

#include <chrono>

class stopwatch{
public:
  stopwatch():start_(std::chrono::high_resolution_clock::now()),total_(0){};
  void start(){
    this->start_=std::chrono::high_resolution_clock::now();
  };
  void split(){
    this->total_+=std::chrono::high_resolution_clock::now()-this->start_;
  };
  void reset(){
    this->total_=std::chrono::high_resolution_clock::duration(0);
  };
  std::chrono::high_resolution_clock::duration::rep elapsed() const{
    return std::chrono::duration_cast<std::chrono::milliseconds>(this->total_).count();
  };
private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  std::chrono::high_resolution_clock::duration total_;
};

#endif
