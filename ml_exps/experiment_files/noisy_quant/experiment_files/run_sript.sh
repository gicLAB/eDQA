for s in 0 1 2 3 4  
do
	for b in 3 4 5 
	do
		python3 experiment_files/noisy_quant.py --model resnet32 --bit $b --seed $s; python3 experiment_files/noisy_quant.py --model mobilev2 --bit $b --seed $s
        done
done
