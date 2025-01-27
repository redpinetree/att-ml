MPICXX = mpicxx
CXX = g++
CXXFLAGS = -O3 -ffloat-store -ffp-contract=off -march=native -pipe -std=c++17 -fopenmp
# CXXFLAGS = -O0

TARGET = ./bin/tree_ml

SRC = ./cpp/main_tree_ml.cpp ./cpp/observables.cpp ./cpp/graph_utils.cpp ./cpp/algorithm_nll.cpp ./cpp/optimize_nll.cpp ./cpp/optimize_nll_born.cpp ./cpp/ttn_ops.cpp ./cpp/ttn_ops_born.cpp ./cpp/mat_ops.cpp ./cpp/sampling.cpp ./cpp/site.cpp ./cpp/bond.cpp ./cpp/mpi_utils.cpp

OBJ = $(SRC:%.cpp=%.o)

.PHONY: clean graph

all: $(TARGET)

$(TARGET): $(OBJ)
	$(MPICXX) $(OBJ) -llapack -lopenblas -o $(TARGET) $(CXXFLAGS)

$(OBJ): %.o: %.cpp
	$(MPICXX) -c $< -o $@ $(CXXFLAGS)

clean:
	@rm -f ./cpp/*.o 2>/dev/null || true
	@rm $(TARGET) 2>/dev/null || true

graph: 
	@make -dn MAKE=: all | sed -rn "s/^(\s+)Considering target file '(.*)'\.$$/\1\2/p"