#!/bin/sh
# test for loop
for i in $(seq 1 4)
do
	for u1 in $(seq 1 2)
	do
		for u2 in $(seq 1 2)
		do
			echo "In loop: index $i, u1u2 = $u1$u2, formatted name: data-test-$u1$u2-$i"
		done
	done
done
