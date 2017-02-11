SHELL := /bin/bash

.PHONY: all
all:
	pip install runcython
	makecython++ stitch_wrapper.pyx "" "stitch_rects.cpp ./hungarian/hungarian.cpp"

hungarian: hungarian/hungarian.so

hungarian/hungarian.so:
	cd hungarian && \
	TF_INC=$$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())') && \
	if [ `uname` == Darwin ];\
	then g++ -std=c++11 -shared hungarian.cc -o hungarian.so -fPIC -I -D_GLIBCXX_USE_CXX11_ABI=0$$TF_INC;\
	else g++ -std=c++11 -shared hungarian.cc -o hungarian.so -fPIC -I  $$TF_INC; fi
