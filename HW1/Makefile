.PHONY: all p1 p2 p3 clean

all: p1 p2 p3

p1: p1.cpp
	g++ -fopenmp -Ofast -o p1 p1.cpp

p2: p2.cpp
	g++ -fopenmp -Ofast -o p2 p2.cpp	

p3: p3.cpp
	g++ -fopenmp -Ofast -o p3 p3.cpp

clean:
	rm -f p1 p2 p3