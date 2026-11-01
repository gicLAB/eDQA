# eDQA

This repo is the offical repo of paper [eDQA: Efficient Deep Quantization of DNN Activations on
Edge Devices](https://openreview.net/pdf?id=SEIBCdgE5W).

## Introduction
This repo is composed by two parts: algorithmic part (ml_exps) and hardware part (hw_exps). The hardware part corresponds to the experiments in the section 4.4 of the paper and the algorithmic part corresponds to the rest experiments. 

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

### Namings
Note that 'im' means the ImageNet and 'pot' means power-of-two quantization.

## Hardware Part

The hardware part (`hw_exps/`) covers the FPGA experiments in Section 4.4 of the paper. It holds
two accelerators, written in SystemC and synthesised with HLS using the
[SECDA](https://github.com/gicLAB) methodology, for the PYNQ-Z1 board:

- **v1 (eDQA):** `hw_exps/dqa_exp/v1/accelerator` quantizes an activation channel and compresses
  its low-order bits with an on-chip canonical Huffman encoder, then decodes and dequantizes it.
- **v2 (NoisyQuant baseline):** `hw_exps/dqa_exp/v2/accelerator` searches for the activation
  scale and noise bias, then quantizes and dequantizes the channel.

Each accelerator comes with a host driver (`accelerator/driver/`) and an experiment program
(`experiment.cc`). The program runs channel sizes from 7×7 to 56×56 and records quantization and
dequantization cycles, data transfer times and the compression ratio.

Other contents:
- `hw_exps/hw_files/`: prebuilt bitstreams (`.bit` + `.hwh`) for both designs, their hardware
  generation configs and the Vivado block-design template.
- `hw_exps/data/`: input activation data and the golden outputs used to check the results
  (`dqa_exp/gen_data.ipynb` generates the inputs).

Note that the experiments are built inside a SECDA framework workspace (Bazel), not as a
standalone project.




