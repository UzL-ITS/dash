CPPFLAGS = -I../include -I/usr/include/eigen3 -I../onnx
ALL_CXXFLAGS = $(CXXFLAGS) $(CPPFLAGS) -msse4.2 -maes -mrdrnd -fopenmp -std=c++17 -I/usr/local/cuda/include -lcudart -lcuda -g #-DLABEL_TENSOR_USE_EIGEN -DEIGEN_USE_BLAS
LDFLAGS = -lgtest -lpthread -lopenblas
OBJDIR = obj
LIBS = -L/usr/local/cuda/lib64
ONNXDIR = ../onnx

SRCS = $(wildcard *.cpp)
DEPENDS = $(SRCS:%.cpp=$(OBJDIR)/%.o)

.PHONY: clean $(OBJDIR)

test: $(OBJDIR) $(OBJDIR)/onnx.proto3.o $(DEPENDS)
	g++ $(LIBS) $(ALL_CXXFLAGS) -o $@ $(filter %.o %.cpp, $^) $(LDFLAGS) -I/usr/local/cuda/include -lcudart -lcuda -L/usr/local/cuda/lib64 `pkg-config --cflags --libs protobuf`

$(OBJDIR)/%.o: %.cpp
	nvcc -x cu -G -Xcompiler "$(ALL_CXXFLAGS) $(LIBS) $(LDFLAGS)" -std=c++17 -c -o $@ $<

$(OBJDIR)/onnx.proto3.o: $(ONNXDIR)/onnx.proto3.pb.cc
	pkg-config --cflags protobuf
	g++ $(ALL_CXXFLAGS) -c ../onnx/onnx.proto3.pb.cc -o $(OBJDIR)/onnx.proto3.o

$(ONNXDIR)/onnx.proto3.pb.cc:
	protoc --proto_path=$(ONNXDIR) --cpp_out=$(ONNXDIR) onnx.proto3

$(OBJDIR):
	mkdir -p $@

clean:
	rm -f test
	rm -r -f $(OBJDIR)
	rm -f $(ONNXDIR)/*.pb.*


