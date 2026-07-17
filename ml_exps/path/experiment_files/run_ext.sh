for s in 3 4 
do
	#python3 experiment_files/pad_saw_path.py --model vit --seed $s; 
	python3 experiment_files/pad_saw_path.py --model resnet18 --seed $s
done
