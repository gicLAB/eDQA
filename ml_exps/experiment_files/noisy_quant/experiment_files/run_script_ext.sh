for b in 3 4 5 6 8
do
	python3 experiment_files/noisy_quant_ext_ext.py --model resnet32 --bit $b --seed 0
	python3 experiment_files/noisy_quant_ext_ext.py --model mobilev2 --bit $b --seed 0
	python3 experiment_files/noisy_quant_ext_ext.py --model vit --bit $b --seed 0
	python3 experiment_files/noisy_quant_ext_ext.py --model resnet18 --bit $b --seed 0
done