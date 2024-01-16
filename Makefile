MPICXX = mpicxx
CXX = g++
CXXFLAGS = -O3 -ffloat-store -ffp-contract=off -march=native -pipe -std=c++17
# CXXFLAGS = -O0

TARGET = ./bin/tree_approx_potts
TARGET_RENYI = ./bin/renyi_approx
TARGET_OLD = ./bin/tree_approx_potts_old
SRC =./cpp/tree_approx_potts.cpp ./cpp/observables.cpp ./cpp/algorithm.cpp ./cpp/graph_utils.cpp ./cpp/site.cpp ./cpp/bond.cpp ./cpp/bond_utils.cpp ./cpp/mpi_utils.cpp
SRC_RENYI =./cpp_cmd/cmd_approx.cpp ./cpp_cmd/observables.cpp ./cpp_cmd/algorithm.cpp ./cpp_cmd/renyi/optimize.cpp ./cpp_cmd/graph_utils.cpp ./cpp_cmd/site.cpp ./cpp_cmd/bond.cpp ./cpp_cmd/mpi_utils.cpp
SRC_OLD = ./cpp_old/tree_approx_potts.cpp ./cpp_old/algorithm.cpp ./cpp_old/graph.cpp ./cpp_old/graph_utils.cpp ./cpp_old/site.cpp ./cpp_old/bond.cpp ./cpp_old/bond_utils.cpp
OBJ = $(SRC:%.cpp=%.o)
OBJ_RENYI = $(SRC_RENYI:%.cpp=%.o)
OBJ_OLD = $(SRC_OLD:%.cpp=%.o)

.PHONY: clean
.SUFFIXES: .cpp .hpp .o

all: $(TARGET_RENYI)
all: $(TARGET) $(TARGET_RENYI)
all: $(TARGET) $(TARGET_OLD) $(TARGET_RENYI)

$(TARGET): $(OBJ)
	$(MPICXX) $(OBJ) -o $(TARGET) $(CXXFLAGS)

$(TARGET_RENYI): $(OBJ_RENYI)
	$(MPICXX) $(OBJ_RENYI) -o $(TARGET_RENYI) $(CXXFLAGS)

$(TARGET_OLD): $(OBJ_OLD)
	$(CXX) $(OBJ_OLD) -o $(TARGET_OLD) $(CXXFLAGS)

%.o: %.cpp
	$(MPICXX) -I ./cpp_cmd/renyi -c $< -o $@ $(CXXFLAGS)

clean:
	#@rm -f ./cpp/*.o 2>/dev/null || true
	@rm -f ./cpp_cmd/*.o 2>/dev/null || true
	@rm -f ./cpp_cmd/*/*.o 2>/dev/null || true
	#@rm -f ./cpp_old/*.o 2>/dev/null || true
	#@rm $(TARGET) 2>/dev/null || true
	@rm $(TARGET_RENYI) 2>/dev/null || true
	#@rm $(TARGET_OLD) 2>/dev/null || true