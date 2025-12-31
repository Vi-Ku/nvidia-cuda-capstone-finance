NVCC = nvcc

# Flags:
# -O3: Optimize for speed
# -std=c++14: Required for modern Thrust features
# -lcurand: Link Random Number library
# -lnvToolsExt: Link NVTX Profiling library
NVCC_FLAGS = -O3 -std=c++14 -I./include -lcurand -lnvToolsExt

BIN_DIR = bin
SRC_DIR = src

TARGET = $(BIN_DIR)/risk_engine

# Automatically find all .cu files in src/
SRCS = $(wildcard $(SRC_DIR)/*.cu)

all: clean $(TARGET)

$(TARGET): $(SRCS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

clean:
	rm -f $(BIN_DIR)/* data/*.csv data/*.png