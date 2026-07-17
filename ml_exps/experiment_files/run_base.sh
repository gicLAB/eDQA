for b in 3 4 5 6 8
do
	python3 experiment_files/DQA_ext_ext.py --model resnet18 --method baseline --bit $b --seed 0
	python3 experiment_files/DQA_ext_ext.py --model vit --method baseline --bit $b --seed 0
	python3 experiment_files/DQA_ext_ext.py --model resnet32 --method baseline --bit $b --seed 0
	python3 experiment_files/DQA_ext_ext.py --model mobilev2 --method baseline --bit $b --seed 0
done

for b in 6 8
do 
	python3 experiment_files/DQA_ext.py --model resnet18 --method baseline --bit $b --seed 0
	python3 experiment_files/DQA_ext.py --model vit --method baseline --bit $b --seed 0
	python3 experiment_files/DQA_ext.py --model resnet32 --method baseline --bit $b --seed 0
	python3 experiment_files/DQA_ext.py --model mobilev2 --method baseline --bit $b --seed 0
done