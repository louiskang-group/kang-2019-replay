#!/bin/bash
# Sample compilation script in which the FFTW3 single-precision library has
# been installed and the ziggurat random number generation package by John
# Burkardt has been compiled and installed as a library
gcc -o gc_dynamics gc_dynamics.c \
  -march=native -Ofast -lziggurat -lfftw3f -lm
