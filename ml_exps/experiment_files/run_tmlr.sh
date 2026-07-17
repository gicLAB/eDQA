#for s in 0 1 2 3 4
#do
#	for b in 3 4 5 
#	do
#		python3 experiment_files/DQA_ext_TMLR.py --model resnet18 --bit $b --seed $s
#        done
#done

for b in 5 
do
	#python3 experiment_files/DQA_ext_TMLR.py --model resnet18 --bit $b --seed 0 --method baseline
	#python3 experiment_files/DQA_ext_TMLR.py --model resnet18 --bit $b --seed 0 --method pot
	python3 experiment_files/DQA_ext_TMLR.py --model resnet18 --bit $b --seed 0 --method easyquant
done
