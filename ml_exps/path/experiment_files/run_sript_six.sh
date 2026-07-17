python3 experiment_files/pad_saw_path_ext.py --model resnet18 --seed 0
for s in 1 2 3 4 
do
	python3 experiment_files/pad_saw_path_ext.py --model resnet32 --seed $s 
	python3 experiment_files/pad_saw_path_ext.py --model mobilev2 --seed $s
	python3 experiment_files/pad_saw_path_ext.py --model resnet18 --seed $s
	#python3 experiment_files/pad_saw_path_ext.py --model vit --seed $s
done

