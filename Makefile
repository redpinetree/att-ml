CXX = g++
CXXFLAGS = -O3 -march=native -pipe -std=c++17

TARGET = ./bin/tree_approx_potts
TARGET_OLD = ./bin/tree_approx_potts_old
SRC = ./cpp/tree_approx_potts.cpp ./cpp/observables.cpp ./cpp/algorithm.cpp ./cpp/graph_utils.cpp ./cpp/site.cpp ./cpp/bond.cpp ./cpp/bond_utils.cpp
SRC_OLD = ./cpp_old/tree_approx_potts.cpp ./cpp_old/algorithm.cpp ./cpp_old/graph.cpp ./cpp_old/graph_utils.cpp ./cpp_old/site.cpp ./cpp_old/bond.cpp ./cpp_old/bond_utils.cpp
OBJ = $(SRC:%.cpp=%.o)
OBJ_OLD = $(SRC_OLD:%.cpp=%.o)

.PHONY: clean
.SUFFIXES: .cpp .hpp .o

all: $(TARGET)
# all: $(TARGET) $(TARGET_OLD)

$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $(TARGET) $(CXXFLAGS)

$(TARGET_OLD): ${OBJ_OLD}
	$(CXX) $(OBJ_OLD) -o $(TARGET_OLD) $(CXXFLAGS)

%.o: %.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)

clean:
	@rm -f ./cpp/*.o 2>/dev/null || true
	# @rm -f ./cpp_old/*.o 2>/dev/null || true
	@rm $(TARGET) 2>/dev/null || true
	# @rm $(TARGET_OLD) 2>/dev/null || true