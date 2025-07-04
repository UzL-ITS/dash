IDIR = ../../../dash/include
ONNXDIR = ../../../dash/onnx
EIGENDIR = /usr/include/eigen3
MAIN_INCLUDES = -I/usr/local/cuda/include -ldlib -lpthread -lcudart -lcuda -L/usr/local/cuda/lib64 `pkg-config --cflags --libs protobuf` -lopenblas
CPPFLAGS = -I$(IDIR) -I$(ONNXDIR) -I$(EIGENDIR)
ALL_CXXFLAGS = $(CXXFLAGS) $(CPPFLAGS) -msse4.2 -maes -fopenmp -std=c++17 -mrdrnd -I../lib #-DLABEL_TENSOR_USE_EIGEN -DEIGEN_USE_BLAS -DEIGEN_VECTORIZE
NVCC_FLAGS = -std=c++17
OBJDIR = ./obj



SRCS = $(wildcard *.cpp)
DEPENDS = $(SRCS:%.cpp=$(OBJDIR)/%.o)

.PHONY: debug release clean $(OBJDIR)

debug: ALL_CXXFLAGS += -O2 -g -Wall
debug: NVCC_FLAGS += -G
profile: ALL_CXXFLAGS += -g -O3 -shared-libgcc
release: ALL_CXXFLAGS += -O3 -DNDEBUG -march=native -ffast-math
debug profile release: main

main: $(OBJDIR) $(OBJDIR)/onnx.proto3.o $(DEPENDS)
	g++ $(ALL_CXXFLAGS) -o $@ $(filter %.o %.cpp, $^) $(MAIN_INCLUDES)
	@echo "Linked $@"

$(OBJDIR)/%.o: %.cpp
	nvcc -x cu -Xcompiler "$(ALL_CXXFLAGS)" $(NVCC_FLAGS) -c -o $@ $<
	@echo "NVCC => $<"

$(OBJDIR)/onnx.proto3.o: $(ONNXDIR)/onnx.proto3.pb.cc
	pkg-config --cflags protobuf
	g++ $(ALL_CXXFLAGS) -c $^ -o $(OBJDIR)/onnx.proto3.o
	@echo "CC => $(OBJDIR)/onnx.proto3.o"

$(ONNXDIR)/onnx.proto3.pb.cc:
	protoc --proto_path=$(ONNXDIR) --cpp_out=$(ONNXDIR) onnx.proto3

$(OBJDIR):
	mkdir -p $@

clean:
	rm -f main
	rm -f gmon.out
	rm -r -f $(OBJDIR)
	rm -f $(ONNXDIR)/*.pb.*
