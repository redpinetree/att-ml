# for l in 4 8
# do
	# for i in {0..9}
	# do
		# ./rudy/rudy -spinglass2pm $l $l 50 $i > "bimodal${l}x${l}_${i}.txt"
		# ./rudy/rudy -spinglass2g $l $l $i > "gaussian${l}x${l}_${i}.txt"
	# done
# done

# for l in 3 4 5
# do
	# for i in {0..99}
	# do
		# ./rudy/rudy -spinglass2g $l $l $i > "gaussian${l}x${l}_${i}.txt"
	# done
# done

# for l in 3 4
# do
	# for i in {0..1023}
	# do
		# ./rudy/rudy -spinglass2g $l $l $i > "gaussian${l}x${l}_${i}.txt"
	# done
# done

# for n in 4 8 16 32 64 128
# do
	# ./rudy/rudy -toroidal_grid $n 1 > "ferro_ring${n}.txt"
# done

# for l in 4 8 16 32 64
# do
	# for p in 0.1 0.25 0.5 0.75 1
	# do
		# python3 ./gen_hier2d.py ${l} ${l} ${p} hier2d/hier2d_l${l}_p${p}.txt
	# done
# done

# for l in 4 8 16 32 64
# do
	# for i in {0..9}
	# do
		# python3 ./gen_randgauge2d.py ${l} ${l} randgauge2d/l${l}/randgauge2d_l${l}_${i}.txt
	# done
# done
