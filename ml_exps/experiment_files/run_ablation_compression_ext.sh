for s in 0 
do
	for c in 'huffman' 
	do
		#python3 experiment_files/DQA_ablation_comp_ext.py --model resnet32 --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
		#python3 experiment_files/DQA_ablation_comp_ext.py --model mobilev2 --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
        #python3 experiment_files/DQA_ablation_comp_ext.py --model vit_cifar --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
        python3 experiment_files/DQA_ablation_comp_ext.py --model resnet18_cifar --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
        #python3 experiment_files/DQA_ablation_comp_ext.py --model vit_im --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
        python3 experiment_files/DQA_ablation_comp_ext.py --model resnet18_im --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
    done
done
wait

for s in 1
do
	for c in 'huffman' 
	do
		#python3 experiment_files/DQA_ablation_comp_ext.py --model resnet32 --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
		#python3 experiment_files/DQA_ablation_comp_ext.py --model mobilev2 --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
        #python3 experiment_files/DQA_ablation_comp_ext.py --model vit_cifar --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
        python3 experiment_files/DQA_ablation_comp_ext.py --model resnet18_cifar --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
        #python3 experiment_files/DQA_ablation_comp_ext.py --model vit_im --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
        python3 experiment_files/DQA_ablation_comp_ext.py --model resnet18_im --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
    done
done
wait

for s in 2 
do
	for c in 'huffman' 
	do
		#python3 experiment_files/DQA_ablation_comp_ext.py --model resnet32 --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
		#python3 experiment_files/DQA_ablation_comp_ext.py --model mobilev2 --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
        #python3 experiment_files/DQA_ablation_comp_ext.py --model vit_cifar --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
        python3 experiment_files/DQA_ablation_comp_ext.py --model resnet18_cifar --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
        #python3 experiment_files/DQA_ablation_comp_ext.py --model vit_im --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
        python3 experiment_files/DQA_ablation_comp_ext.py --model resnet18_im --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
    done
done
wait

for s in 3 
do
	for c in 'huffman'
	do
		#python3 experiment_files/DQA_ablation_comp_ext.py --model resnet32 --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
		#python3 experiment_files/DQA_ablation_comp_ext.py --model mobilev2 --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
        #python3 experiment_files/DQA_ablation_comp_ext.py --model vit_cifar --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
        python3 experiment_files/DQA_ablation_comp_ext.py --model resnet18_cifar --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
        #python3 experiment_files/DQA_ablation_comp_ext.py --model vit_im --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
        python3 experiment_files/DQA_ablation_comp_ext.py --model resnet18_im --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
    done
done
wait

for s in 4
do
	for c in 'huffman' 
	do
		#python3 experiment_files/DQA_ablation_comp_ext.py --model resnet32 --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
		#python3 experiment_files/DQA_ablation_comp_ext.py --model mobilev2 --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
        #python3 experiment_files/DQA_ablation_comp_ext.py --model vit_cifar --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
        python3 experiment_files/DQA_ablation_comp_ext.py --model resnet18_cifar --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
        #python3 experiment_files/DQA_ablation_comp_ext.py --model vit_im --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
        python3 experiment_files/DQA_ablation_comp_ext.py --model resnet18_im --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s &
    done
done
wait