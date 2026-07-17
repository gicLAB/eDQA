for s in 0 1 2 3 4
do
	for t in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9  
	do
		python3 experiment_files/DQA_ablation.py --model resnet32 --bit 3 --m 3 --imp_ratio $t --seed $s
		python3 experiment_files/DQA_ablation.py --model mobilev2 --bit 3 --m 3 --imp_ratio $t --seed $s
        done
done

