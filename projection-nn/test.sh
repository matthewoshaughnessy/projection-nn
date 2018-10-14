#!/bin/sh
# test for loop
for i in {1..10}
do
	for u1 in {1..3}
	do
		for u2 in {1..3}
		do
			echo "In loop: index $i, u1u2 = $u1$u25"
		done
	done
done