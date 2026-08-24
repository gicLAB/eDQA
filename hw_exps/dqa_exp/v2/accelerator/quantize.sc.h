#include <algorithm>
#include <cmath>

int ACCNAME::clamp(int val, int min_val, int max_val) {
  return std::max(min_val, std::min(val, max_val));
}

float ACCNAME::float_clamp(float val, float min_val, float max_val) {
  float clamped_val = std::max(min_val, std::min(val, max_val));
  if (clamped_val == 0) return 0.0f; // Handle -0 case
  return std::max(min_val, std::min(val, max_val));
}

float ACCNAME::floor(float vav) { return floor(vav); }

float ACCNAME::round(float vav) { return ceil(vav - 0.5); }

float ACCNAME::round_half_to_even(float value) {
  if (round(value) == -0.0f) {
    return 0.0f; // Handle -0 case
  }
  return round(value);
}

float ACCNAME::mse_loss() {
  float sum = 0.0f;
  for (int i = 0; i < channel_size; ++i) {
    float diff = Out0[i] - Out1[i];
    sum += (diff * diff);
  }
  float loss = sum / channel_size;
  return loss;
}

void ACCNAME::all_quant_activation_sub() {
  for (int i = 0; i < channel_size; ++i) {
    all_quant_input[i] = all_quant_input[i] - quant_activation_candidate;
  }
}

void ACCNAME::all_quant_activation_add() {
  for (int i = 0; i < channel_size; ++i) {
    all_quant_input[i] = all_quant_input[i] + quant_activation_candidate;
  }
}

// void ACCNAME::all_quant_print() {
//   cout << "===========================" << endl;
//   for (int i = 0; i < height; ++i) {
//     for (int j = 0; j < width; ++j) {
//       std::cout << all_quant_input[height * i + j] << " ";
//     }
//     std::cout << std::endl;
//   }
//   cout << "===========================" << endl;
// }

// void ACCNAME::all_quant_print_Out1() {
//   cout << "===========================" << endl;
//   for (int i = 0; i < height; ++i) {
//     for (int j = 0; j < width; ++j) {
//       std::cout << Out1[height * i + j] << " ";
//     }
//     std::cout << std::endl;
//   }
//   cout << "===========================" << endl;
// }

void ACCNAME::all_quant_activation_clamp(float min_val, float max_val) {
  for (int i = 0; i < channel_size; ++i) {
    all_quant_input[i] = float_clamp(all_quant_input[i], min_val, max_val);
  }
}
void ACCNAME::all_quant_activation(float act_scale) {
  for (int i = 0; i < channel_size; ++i) {
    float val = all_quant_input[i];
    float q = round_half_to_even(val / act_scale);
    all_quant_input[i] = q * act_scale;
  }
}

void ACCNAME::all_quant_load() {
  for (int i = 0; i < channel_size; ++i) {
    all_quant_input[i] = channels[i];
  }
}

void ACCNAME::all_quant_activation_matmul() {
  for (int i = 0; i < channel_size; ++i) data[i] = all_quant_input[i];
  Sig_Start(start_matmul, done_matmul);
  for (int i = 0; i < channel_size; ++i) Out1[i] = output[i];
  wait();
}

float ACCNAME::percentile_search() {
#pragma HLS inline OFF

  float absmax = max_ele;
  int search_space = PERCENTILE_SEARCH_SPACE;
  float min_loss = 1e6;
  float best_act_scale = 0.0f;
  Quantize_si.write(30);
  wait();
  for (int i = search_space; i > 0; --i) {
    float clip_value = (absmax / search_space) * i;
    float act_scale = clip_value / (BitWidth_Max);
    Quantize_si.write(31);
    wait();
    all_quant_load();
    Quantize_si.write(32);
    wait();
    all_quant_activation_clamp(-clip_value, clip_value);
    Quantize_si.write(33);
    wait();
    all_quant_activation(act_scale);
    Quantize_si.write(34);
    wait();
    all_quant_activation_matmul();
    Quantize_si.write(35);
    wait();
    float loss = mse_loss();
    Quantize_si.write(36);
    wait();
    // ALOG("Percentile Search | Loss: " << loss << " | "
    //                                   << "Act Scale: " << act_scale << " | "
    //                                   << "Clip Value: " << clip_value <<
    //                                   endl);
    if (loss < min_loss) {
      min_loss = loss;
      best_act_scale = act_scale;
    }
  }
  return best_act_scale;
}

float ACCNAME::search_mean(float act_scale) {
#pragma HLS inline OFF

  int search_space = MEAN_SEARCH_SPACE;
  float min_loss = 1e6;
  float best_candidate = 0.0f;
  Quantize_si.write(40);
  wait();
  for (int i = (-search_space); i < search_space; i++) {
    float candidate = (act_scale * i) / search_space;
    quant_activation_candidate = candidate;
    Quantize_si.write(41);
    wait();
    all_quant_load();
    // all_quant_print();
    Quantize_si.write(42);
    wait();
    all_quant_activation_add();
    // all_quant_print();
    Quantize_si.write(43);
    wait();
    all_quant_activation(act_scale);
    // all_quant_print();
    Quantize_si.write(44);
    wait();
    all_quant_activation_sub();
    // all_quant_print();
    Quantize_si.write(45);
    wait();
    all_quant_activation_matmul();
    // all_quant_print_Out1();
    Quantize_si.write(46);
    wait();
    float loss = mse_loss();
    Quantize_si.write(47);
    wait();
    // ALOG("Search Mean | Loss: " << loss << " | "
    //                             << "Act Scale: " << act_scale << " | "
    //                             << "Candidate: " << candidate << " | " <<
    //                             endl);
    if (loss < min_loss) {
      min_loss = loss;
      best_candidate = candidate;
    }
  }
  return best_candidate;
}

void ACCNAME::forward_pass() {
#pragma HLS inline OFF

  for (int i = 0; i < channel_size; ++i) data[i] = channels[i];
  Sig_Start(start_matmul, done_matmul);
  for (int i = 0; i < channel_size; ++i) Out0[i] = output[i];
  wait();
}

void ACCNAME::add_noise(float noise) {
#pragma HLS inline OFF

  for (int i = 0; i < channel_size; ++i) {
    ALOG("Adding noise: " << noise << " to channel[" << i
                          << "] = " << channels[i] << endl);
    channels[i] += noise;
  }
}

void ACCNAME::quant_activation() {
#pragma HLS inline OFF
  for (int i = 0; i < channel_size; ++i) {
    float val = channels[i];
    float q = round_half_to_even(val / act_scale_global);
    char q_int = clamp(q, BitWidth_Min, BitWidth_Max);
    sc_uint<BitWidth> q_fixed = q_int;
    quantized_channels[i] = q_fixed;
    ALOG("QA | " << i << " | "
                 << "val: " << val << " | "
                 << "q: " << q << " | "
                 << "q_int: " << (int)q_int << " | "
                 << "q_fixed: " << quantized_channels[i] << endl);
  }
}

void ACCNAME::Quantize() {
  done_quantize.write(false);
  start_matmul.write(false);
  Quantize_si.write(0);
  wait();
  while (true) {
    Quantize_si.write(1);
    Sig_Wait(start_quantize);
    Quantize_si.write(2);
    wait();

    forward_pass();
    Quantize_si.write(3);
    wait();

    act_scale_global = percentile_search();
    Quantize_si.write(4);
    wait();

    noise_global = search_mean(act_scale_global);
    Quantize_si.write(5);
    wait();

    add_noise(noise_global);
    Quantize_si.write(6);
    wait();
    ALOG("Quantization | "
         << "act_scale_global: " << act_scale_global << " | "
         << "noise_global: " << noise_global << endl);

    quant_activation();
    Quantize_si.write(7);
    wait();

    Sig_Done(start_quantize, done_quantize);
  }
}

// ALOG("Quantization | "
//      << "act_scale_global: " << act_scale_global << " | "
//      << "noise_global: " << noise_global << endl);
