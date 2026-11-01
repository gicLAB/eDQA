# eDQA

This repo is the offical repo of paper [eDQA: Efficient Deep Quantization of DNN Activations on
Edge Devices](https://openreview.net/pdf?id=SEIBCdgE5W).

## Introduction
This repo is composed by two parts: algorithmic part (ml_exps) and hardware part (hw_exps). The hardware part correspond to the experiments in the section 4.4 of the paper and the algorithmic part correspond to the rest experiments. 

## Algorithmic Part
In this part, all the algorithms are embedded in the model files (ml_exps/experiment_files/models_with_algorithm).

To run the main experiments in section 4.2, refer to the scripts:<br> 
ml_exps/experiment_files/run_tmlr.sh,<br> 
ml_exps/experiment_files/run_ext.sh,<br>
ml_exps/experiment_files/run_sript.sh,<br>
ml_exps/experiment_files/run_sript_six.sh<br>
ml_exps/experiment_files/run_easy_quant.sh. 

For the NoisyQuant, refer to scripts:
ml_exps/experiment_files/noisy_quant/experiment_files/run_tmlr.sh. (It contains part of the cases in our paper, but can be an example for how to implement and run the NoisyQuant in our research.)

To run the ablation study in section 4.3, refer to the script: ml_exps/experiment_files/run_ablation_ext.sh.

To run the experiments in section 3.2.2, refer to the script:<br>
ml_exps/experiment_files/run_ablation_compresson.sh,<br>
ml_exps/experiment_files/run_ablation_compression_ext.sh.

### Get the channel ranks
Note that for DQA, we need to get the ranks of activation channels first. To get the channels, refer to the scripts:<br>
ml_exps/path/experiment_files/run_ext.sh,<br>
ml_exps/path/experiment_files/run_im.sh,<br>
ml_exps/path/experiment_files/run_sript.sh<br>
ml_exps/path/experiment_files/run_sript_six.sh.<br>






