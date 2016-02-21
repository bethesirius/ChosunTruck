SHELL := /bin/bash

.PHONY: all
all:
	pip install runcython
	makecython++ stitch_wrapper.pyx "" "stitch_rects.cpp ./hungarian/hungarian.cpp"

