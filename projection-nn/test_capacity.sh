#!/bin/sh
for i in $(seq 1 100)
do
	for u1 in $(seq 2 8) # number of neurons in first hidden layer
	do
		for u2 in $(seq 2 8) # number of neurons in second hidden layer
		do
			python3 main.py nunits=2,$u1,$u2,2 datafilename=test_capacity_$u1$u2_$i randseed=$i nEpochs=30 Ktrain=4096 Kval=50 Ktest=10000
		done
	done
done
