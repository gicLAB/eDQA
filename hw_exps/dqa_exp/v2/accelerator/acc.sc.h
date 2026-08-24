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
  sc_in<unsigned int> dim_in;
  sc_out<unsigned int> channel_size_out;
  sc_out<unsigned int> dim_out;

  // ================================================= //
  // Global variables
  // ================================================= //

  float max_ele;
  float quant_activation_candidate;
  float noise_global;
  float act_scale_global;

  // ================================================= //
  // Global buffers
  // ================================================= //
  float channels[INPUT_SIZE];
  sc_uint<N_Bits> quantized_channels[INPUT_SIZE];
  float output_channels[INPUT_SIZE];

  float data[INPUT_SIZE];
  float weights[INPUT_SIZE];
  float output[INPUT_SIZE];

  float all_quant_input[INPUT_SIZE];
  float Out0[INPUT_SIZE];
  float Out1[INPUT_SIZE];

  // ================================================= //
  // Global signals
  // ================================================= //

  DEFINE_SC_SIGNAL(bool, start_quantize);
  DEFINE_SC_SIGNAL(bool, done_quantize);
  DEFINE_SC_SIGNAL(bool, start_dequantize);
  DEFINE_SC_SIGNAL(bool, done_dequantize);
  DEFINE_SC_SIGNAL(bool, start_matmul);
  DEFINE_SC_SIGNAL(bool, done_matmul);

  DEFINE_SC_SIGNAL(unsigned int, height);
  DEFINE_SC_SIGNAL(unsigned int, width);
  DEFINE_SC_SIGNAL(unsigned int, depth);
  DEFINE_SC_SIGNAL(unsigned int, channel_size);

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

  float float_clamp(float val, float min_val, float max_val);

  void all_quant_activation_sub();

  void all_quant_activation_add();

  void all_quant_activation_clamp(float min_val, float max_val);

  void all_quant_activation(float act_scale);

  void all_quant_load();

  // void all_quant_print();

  // void all_quant_print_Out1();

  void all_quant_activation_matmul();

  float mse_loss();

  float percentile_search();

  float search_mean(float act_scale);

  // float search_bias(float act_scale);

  void add_noise(float noise);

  void sub_noise(float noise);

  void forward_pass();

  void quant_activation();
  

  // DQA_DeQuantization

  void dequant_activation();

  // ================================================= //
  // HWC
  // ================================================= //

  HWC_Reset;

  HWC_CTHREAD(Compute);

  HWC_CTHREAD(Quantize);

  HWC_CTHREAD(DeQuantize);

  HWC_CTHREAD(MatMul);

  void HW_MAIN() {
    wait();
    while (true) {
      {
#pragma HLS LATENCY max = 0 min = 0
#pragma HLS protocol fixed
        HWC_Logic(Compute);
        HWC_Logic(Quantize);
        HWC_Logic(DeQuantize);
        HWC_Logic(MatMul);
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

    SC_CTHREAD(MatMul, clock);
    reset_signal_is(reset, true);

    SC_CTHREAD(HW_MAIN, clock);
    reset_signal_is(reset, true);

    CTRL_PragGroup;
    AXI4M_PragAddr(input);
    AXI4M_PragAddr(output);
    CTRL_Prag(channel_size_in);
    CTRL_Prag(dim_in);
    CTRL_Prag(channel_size_out);
    CTRL_Prag(dim_out);

    HWC_PragReset;
    HWC_PragGroup(Compute);
    HWC_PragGroup(Quantize);
    HWC_PragGroup(DeQuantize);
    HWC_PragGroup(MatMul);
  }
};

#endif