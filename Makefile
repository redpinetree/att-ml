MPICXX = mpicxx
CXX = g++
CXXFLAGS = -O3 -march=native -pipe -std=c++17 -fopenmp -fno-trapping-math -fno-math-errno
# CXXFLAGS = -O0

TARGET = ./bin/att_ml

SRC = ./cpp/main_att_ml.cpp ./cpp/observables.cpp ./cpp/graph_utils.cpp ./cpp/algorithm_nll.cpp ./cpp/optimize_nll.cpp ./cpp/optimize_nll_born.cpp ./cpp/ttn_ops.cpp ./cpp/ttn_ops_born.cpp ./cpp/mat_ops.cpp ./cpp/sampling.cpp ./cpp/site.cpp ./cpp/bond.cpp

OBJ = $(SRC:%.cpp=%.o)

.PHONY: clean graph

all: $(TARGET)

$(TARGET): $(OBJ)
	mkdir -p ./bin
	$(CXX) $(OBJ) -llapack -lblas -o $(TARGET) $(CXXFLAGS)

$(TARGET_DEBUG): $(OBJ)
	mkdir -p ./bin
	$(CXX) -g $(OBJ) -llapack -lblas -o $(TARGET_DEBUG) $(CXXFLAGS)

$(OBJ): %.o: %.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)

clean:
	@rm -f ./cpp/*.o 2>/dev/null || true
	@rm $(TARGET) 2>/dev/null || true

graph: 
	@make -dn MAKE=: all | sed -rn "s/^(\s+)Considering target file '(.*)'\.$$/\1\2/p"