#ifndef ACCNAME_H
#define ACCNAME_H

#include "acc_config.sc.h"
#include <systemc.h>

SC_MODULE(ACCNAME) {
  sc_in<bool> clock;
  sc_in<bool> reset;

  // ================================================= //
  // Global ports
  // ================================================= //

  // Control ports
  CTRL_Define_Ports;

  // Data ports
  AXI4M_Bus_Port(float, input);
  AXI4M_Bus_Port(float, output);
  sc_in<unsigned int> channel_size_in;
  sc_in<unsigned int> num_channels_in;
  sc_out<unsigned int> channel_size_out;
  sc_out<unsigned int> num_channels_out;
  sc_out<unsigned int> compression_ratio_out;

  // ================================================= //
  // Global variables
  // ================================================= //

  // int channel_count;
  // int channel_size;
  // int length;
  float max_ele;

  unsigned int encoded_diff_size;
  unsigned int diff_size;

  // ================================================= //
  // Global buffers
  // ================================================= //
  float channels[INPUT_SIZE];
  sc_uint<N_Bits> quantized_channels[INPUT_SIZE];
  float output_channels[INPUT_SIZE];

  // Diff Channel
  sc_uint<M_Bits> diff[MAX_CHANNEL_SIZE];
  sc_uint<1> encoded_diff[MAX_CHANNEL_SIZE * 3];
  sc_uint<M_Bits> decoded_diff[MAX_CHANNEL_SIZE];

  // Huffman encoding variables
  unsigned int freq[M_Range];
  SymFreq filtered[M_Range];

  SymFreq sorted[M_Range];
  SymFreq sorted_copy1[M_Range];
  SymFreq sorted_copy2[M_Range];

  SYMBOL parent[M_Range - 1];
  SymNode left[M_Range - 1];
  SymNode right[M_Range - 1];

  unsigned int length_histogram[TREE_DEPTH];
  unsigned int truncated_length_histogram1[TREE_DEPTH];
  unsigned int truncated_length_histogram2[TREE_DEPTH];
  CodewordLength symbol_bits[M_Range];

  sc_uint<TREE_DEPTH_BITS> child_depth[MAX_CHANNEL_SIZE - 1];
  unsigned int internal_length_histogram[TREE_DEPTH];
  unsigned int tree_frequency[TreeM_Range];

  sc_uint<CODEWORD_LENGTH_BITS> encoding_length[M_Range];
  Codeword encoding_cwords[M_Range];

  // Decompression variables
  int de_parent[TreeM_Range];
  int de_left[TreeM_Range];
  int de_right[TreeM_Range];
  int de_node_symbol[TreeM_Range];

  // ================================================= //
  // Global signals
  // ================================================= //

  DEFINE_SC_SIGNAL(bool, start_quantize);
  DEFINE_SC_SIGNAL(bool, done_quantize);
  DEFINE_SC_SIGNAL(bool, start_dequantize);
  DEFINE_SC_SIGNAL(bool, done_dequantize);

  DEFINE_SC_SIGNAL(unsigned int, channel_size);
  DEFINE_SC_SIGNAL(unsigned int, channel_count);
  DEFINE_SC_SIGNAL(unsigned int, data_length);

  // ================================================= //
  // Profiling variable
  // ================================================= //

  // ================================================= //
  // Functions
  // ================================================= //

  // DQA_Quantization
  float floor(float);

  float round(float);

  float round_half_to_even(float);

  int clamp(int val, int min_val, int max_val);

  void quant_DQA();

  // Huffman Compression
  void encode(int num_symbols);

  void create_codeword(CodewordLength symbol_bits[M_Range],
                       unsigned int codeword_length_histogram[TREE_DEPTH]);

  void canonize_tree(SymFreq sorted[M_Range], int num_symbols,
                     unsigned int codeword_length_histogram[TREE_DEPTH],
                     CodewordLength symbol_bits[M_Range]);

  void truncate_tree(unsigned int input_length_histogram[TREE_DEPTH],
                     unsigned int output_length_histogram_0[TREE_DEPTH],
                     unsigned int output_length_histogram_1[TREE_DEPTH]);

  void compute_bit_length(SYMBOL parent[M_Range - 1], SymNode left[M_Range - 1],
                          SymNode right[M_Range - 1], int num_symbols,
                          unsigned int length_histogram[TREE_DEPTH]);

  void create_tree(SymFreq in[M_Range], int num_symbols,
                   SYMBOL parent[M_Range - 1], SymNode left[M_Range - 1],
                   SymNode right[M_Range - 1]);

  void count_diff_frequency();

  void filter(int *num_symbols);

  void sort(int num_symbols);

  void huffman_encode();

  // DQA_DeQuantization

  void dequant_DQA();

  void huffman_reconstruct();

  void huffman_decode();

  // ================================================= //
  // HWC
  // ================================================= //

  HWC_Reset;

  HWC_CTHREAD(Compute);

  HWC_CTHREAD(Quantize);

  HWC_CTHREAD(DeQuantize);

  void HW_MAIN() {
    wait();
    while (true) {
      {
#pragma HLS LATENCY max = 0 min = 0
#pragma HLS protocol fixed
        HWC_Logic(Compute);
        HWC_Logic(Quantize);
        HWC_Logic(DeQuantize);
        DWAIT();
      }
    }
  }
  // ================================================= //

  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(sc_module_name name_)
      : sc_module(name_), input_port("input_port"), output_port("output_port") {

    SC_CTHREAD(Compute, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(Quantize, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(DeQuantize, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(HW_MAIN, clock);
    reset_signal_is(reset, true);

    CTRL_PragGroup;
    AXI4M_PragAddr(input);
    AXI4M_PragAddr(output);
    CTRL_Prag(channel_size_in);
    CTRL_Prag(num_channels_in);
    CTRL_Prag(channel_size_out);
    CTRL_Prag(num_channels_out);
    CTRL_Prag(compression_ratio_out);

    HWC_PragReset;
    HWC_PragGroup(Compute);
    HWC_PragGroup(Quantize);
    HWC_PragGroup(DeQuantize);
  }
};

#endif