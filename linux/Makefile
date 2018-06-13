OUT_O_DIR = build

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
