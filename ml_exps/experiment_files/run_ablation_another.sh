for s in 0 1 2 3 4
do
	for y in 1 2 
	do
		python3 experiment_files/DQA_ablation.py --model resnet32 --bit 3 --m $y --imp_ratio 0.4 --seed $s
		python3 experiment_files/DQA_ablation.py --model mobilev2 --bit 3 --m $y --imp_ratio 0.4 --seed $s
    done
done
