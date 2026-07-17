for s in 0 1 2 3 4
do
	for c in 'huffman' 'deflate' 'lzma' 'zstd' 
	do
		python3 experiment_files/DQA_ablation_comp.py --model resnet32 --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s
		python3 experiment_files/DQA_ablation_comp.py --model mobilev2 --bit 3 --m 3 --imp_ratio 0.4 --comp $c --seed $s
        done
done

