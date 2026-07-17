for b in 3 4 5 6 8
do
	for m in pot baseline
	do
		python3 experiment_files/DQA_ext_ext.py --model resnet18 --method $m --bit $b --seed 0
		python3 experiment_files/DQA_ext_ext.py --model vit --method $m --bit $b --seed 0
		python3 experiment_files/DQA_ext_ext.py --model resnet32 --method $m --bit $b --seed 0
		python3 experiment_files/DQA_ext_ext.py --model mobilev2 --method $m --bit $b --seed 0
	done
done

for s in 0 1 2 3 4
do
	for b in 3 4 5 6 8
	do
		python3 experiment_files/DQA_ext_ext.py --model resnet18 --method dqa --bit $b --seed $s
		python3 experiment_files/DQA_ext_ext.py --model vit --method dqa --bit $b --seed $s
		python3 experiment_files/DQA_ext_ext.py --model resnet32 --method dqa --bit $b --seed $s
		python3 experiment_files/DQA_ext_ext.py --model mobilev2 --method dqa --bit $b --seed $s	
    done
done