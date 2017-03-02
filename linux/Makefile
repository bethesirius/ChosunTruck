OUT_O_DIR = build
TENSORBOX_UTILS_DIR = tensorbox/utils

OBJS = $(OUT_O_DIR)/getScreen_linux.o $(OUT_O_DIR)/main2.o $(OUT_O_DIR)/IPM.o $(OUT_O_DIR)/linefinder.o $(OUT_O_DIR)/uinput.o

CC = gcc
CFLAGS = -std=c11 -Wall -O3 -march=native
CPP = g++
CPPFLAGS = `pkg-config opencv --cflags --libs` -std=c++11 -lX11 -Wall -fopenmp -O3 -march=native

TARGET = ChosunTruck

$(TARGET) : $(OBJS)
	$(CPP) $(OBJS) $(CPPFLAGS) -o $@

$(OUT_O_DIR)/main2.o : src/main2.cc
	mkdir -p $(@D)
	$(CPP) -c $< $(CPPFLAGS) -o $@
$(OUT_O_DIR)/getScreen_linux.o : src/getScreen_linux.cc
	mkdir -p $(@D)
	$(CPP) -c $< $(CPPFLAGS) -o $@
$(OUT_O_DIR)/IPM.o : src/IPM.cc
	mkdir -p $(@D)
	$(CPP) -c $< $(CPPFLAGS) -o $@
$(OUT_O_DIR)/linefinder.o : src/linefinder.cc
	mkdir -p $(@D)
	$(CPP) -c $< $(CPPFLAGS) -o $@
$(OUT_O_DIR)/uinput.o : src/uinput.c
	mkdir -p $(@D)
	$(CC) -c $< $(CFLAGS) -o $@

clean : 
	rm -f $(OBJS) ./$(TARGET)

.PHONY: Drive

Drive:
	pip install runcython
	makecython++ $(TENSORBOX_UTILS_DIR)/stitch_wrapper.pyx "" "$(TENSORBOX_UTILS_DIR)/stitch_rects.cpp $(TENSORBOX_UTILS_DIR)/hungarian/hungarian.cpp"

hungarian: $(TENSORBOX_UTILS_DIR)/hungarian/hungarian.so

$(TENSORBOX_UTILS_DIR)/hungarian/hungarian.so:
	cd $(TENSORBOX_UTILS_DIR)/hungarian && \
	TF_INC=$$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())') && \
	if [ `uname` == Darwin ];\
	then g++ -std=c++11 -shared hungarian.cc -o hungarian.so -fPIC -I -D_GLIBCXX_USE_CXX11_ABI=0$$TF_INC;\
	else g++ -std=c++11 -shared hungarian.cc -o hungarian.so -fPIC -I  $$TF_INC; fi

	
