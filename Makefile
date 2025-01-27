MPICXX = mpicxx
CXX = g++
CXXFLAGS = -O3 -ffloat-store -ffp-contract=off -march=native -pipe -std=c++17 -fopenmp
# CXXFLAGS = -O0

TARGET_TREE_ML = ./bin/tree_ml
TARGET_TREE_ML_BORN = ./bin/tree_ml_born
TARGET_TREE_ML_HYBRID = ./bin/tree_ml_hybrid

SRC_COMMON_TREE_ML = ./cpp/observables.cpp ./cpp/graph_utils.cpp ./cpp/algorithm_nll.cpp ./cpp/optimize_nll.cpp ./cpp/ttn_ops.cpp ./cpp/mat_ops.cpp ./cpp/sampling.cpp ./cpp/site.cpp ./cpp/bond.cpp ./cpp/mpi_utils.cpp
SRC_COMMON_TREE_ML_BORN = ./cpp/observables.cpp ./cpp/graph_utils.cpp ./cpp/algorithm_nll.cpp ./cpp/optimize_nll.cpp ./cpp/ttn_ops.cpp ./cpp/mat_ops.cpp ./cpp/sampling.cpp ./cpp/site.cpp ./cpp/bond.cpp ./cpp/mpi_utils.cpp
SRC_COMMON_TREE_ML_HYBRID = ./cpp/observables.cpp ./cpp/graph_utils.cpp ./cpp/algorithm_nll.cpp ./cpp/optimize_nll.cpp ./cpp/ttn_ops.cpp ./cpp/mat_ops.cpp ./cpp/sampling.cpp ./cpp/site.cpp ./cpp/bond.cpp ./cpp/mpi_utils.cpp
SRC_TREE_ML = ./cpp/main_tree_ml.cpp
SRC_TREE_ML_BORN = ./cpp/optimize_nll_born.cpp ./cpp/ttn_ops_born.cpp ./cpp/main_tree_ml_born.cpp
SRC_TREE_ML_HYBRID = ./cpp/optimize_nll_born.cpp ./cpp/ttn_ops_born.cpp ./cpp/main_tree_ml_hybrid.cpp

OBJ_COMMON_TREE_ML = $(SRC_COMMON_TREE_ML:%.cpp=%_tree_ml.o)
OBJ_COMMON_TREE_ML_BORN = $(SRC_COMMON_TREE_ML_BORN:%.cpp=%_tree_ml_born.o)
OBJ_COMMON_TREE_ML_HYBRID = $(SRC_COMMON_TREE_ML_HYBRID:%.cpp=%_tree_ml_hybrid.o)
OBJ_TREE_ML = $(SRC_TREE_ML:%.cpp=%.o)
OBJ_TREE_ML_BORN = $(SRC_TREE_ML_BORN:%.cpp=%.o)
OBJ_TREE_ML_HYBRID = $(SRC_TREE_ML_HYBRID:%.cpp=%.o)

.PHONY: clean graph

all: $(TARGET_TREE_ML) $(TARGET_TREE_ML_BORN) $(TARGET_TREE_ML_HYBRID)
tree_ml: $(TARGET_TREE_ML)
tree_ml_born: $(TARGET_TREE_ML_BORN)
tree_ml_hybrid: $(TARGET_TREE_ML_HYBRID)

$(TARGET_TREE_ML): $(OBJ_COMMON_TREE_ML) $(OBJ_TREE_ML)
	$(MPICXX) $(OBJ_COMMON_TREE_ML) $(OBJ_TREE_ML) -llapack -lopenblas -o $(TARGET_TREE_ML) $(CXXFLAGS)

$(TARGET_TREE_ML_BORN): $(OBJ_COMMON_TREE_ML_BORN) $(OBJ_TREE_ML_BORN)
	$(MPICXX) $(OBJ_COMMON_TREE_ML_BORN) $(OBJ_TREE_ML_BORN) -llapack -lopenblas -o $(TARGET_TREE_ML_BORN) $(CXXFLAGS)

$(TARGET_TREE_ML_HYBRID): $(OBJ_COMMON_TREE_ML_HYBRID) $(OBJ_TREE_ML_HYBRID)
	$(MPICXX) $(OBJ_COMMON_TREE_ML_HYBRID) $(OBJ_TREE_ML_HYBRID) -llapack -lopenblas -o $(TARGET_TREE_ML_HYBRID) $(CXXFLAGS)

$(OBJ_COMMON_TREE_ML): %_tree_ml.o: %.cpp
	$(MPICXX) -D MODEL_TREE_ML -c $< -o $@ $(CXXFLAGS)

$(OBJ_COMMON_TREE_ML_BORN): %_tree_ml_born.o: %.cpp
	$(MPICXX) -D MODEL_TREE_ML_BORN -c $< -o $@ $(CXXFLAGS)

$(OBJ_COMMON_TREE_ML_HYBRID): %_tree_ml_hybrid.o: %.cpp
	$(MPICXX) -D MODEL_TREE_ML_BORN -c $< -o $@ $(CXXFLAGS)

$(OBJ_TREE_ML): %.o: %.cpp
	$(MPICXX) -D MODEL_TREE_ML -c $< -o $@ $(CXXFLAGS)

$(OBJ_TREE_ML_BORN): %.o: %.cpp
	$(MPICXX) -D MODEL_TREE_ML_BORN -c $< -o $@ $(CXXFLAGS)

$(OBJ_TREE_ML_HYBRID): %.o: %.cpp
	$(MPICXX) -D MODEL_TREE_ML_BORN -c $< -o $@ $(CXXFLAGS)

clean:
	@rm -f ./cpp/*.o 2>/dev/null || true
	@rm -f ./cpp/*/*.o 2>/dev/null || true
	@rm $(TARGET_TREE_ML) 2>/dev/null || true
	@rm $(TARGET_TREE_ML_BORN) 2>/dev/null || true
	@rm $(TARGET_TREE_ML_HYBRID) 2>/dev/null || true

graph: 
	@make -dn MAKE=: all | sed -rn "s/^(\s+)Considering target file '(.*)'\.$$/\1\2/p"