for s in 0 1 2 3 4
do
    for b in 3 4 5 6 8
    do
        python3 experiment_files/DQA_ext_easyquant.py --model resnet32_cifar --bit $b --seed $s
        python3 experiment_files/DQA_ext_easyquant.py --model mobilev2_cifar --bit $b --seed $s
        python3 experiment_files/DQA_ext_easyquant.py --model vit_cifar --bit $b --seed $s
        python3 experiment_files/DQA_ext_easyquant.py --model resnet18_cifar --bit $b --seed $s
        python3 experiment_files/DQA_ext_easyquant.py --model resnet32_im --bit $b --seed $s
        python3 experiment_files/DQA_ext_easyquant.py --model mobilev2_im --bit $b --seed $s
        python3 experiment_files/DQA_ext_easyquant.py --model vit_im --bit $b --seed $s
        python3 experiment_files/DQA_ext_easyquant.py --model resnet18_im --bit $b --seed $s
    done
done