for b in 3 4 5
do
	python3 experiment_files/noisy_quant_TMLR.py --model resnet18 --bit $b --seed 0
done