#include <algorithm>
#include <cmath>

// Helper function to clamp a value between min and max
int ACCNAME::clamp(int val, int min_val, int max_val) {
  return std::max(min_val, std::min(val, max_val));
}

float ACCNAME::floor(float vav) {
  // Implement floor logic
  return std::floor(vav);
}

float ACCNAME::round(float vav) {
  // Implement round logic
  return std::ceil(vav - 0.5);
}

// Round half to even (banker's rounding)
float ACCNAME::round_half_to_even(float value) { return round(value); }

// Quantization function
void ACCNAME::quant_DQA() {
  // float delta = max_ele / std::pow(2, bitwidth - 1);
  float delta =
      max_ele / BitWidth_IntRange; // delta = max_ele / (2 ** (bitwidth - 1))
  ALOG("Delta: " << delta << endl);
  ALOG("max_ele: " << max_ele << endl);
  // Quantize_si.write(3);
  // wait();

  if (delta == 0) {
    for (int i = 0; i < channel_size; ++i) {
      quantized_channels[i] = 0;
      diff[i] = 0; // diff = torch.zeros(channel.shape).long()
    }
  } else {
    for (int i = 0; i < channel_size; ++i) {
      float val = channels[i];
      float q = round_half_to_even(
          val / delta); // sign * roundup(abs[q])) // banker's round
      char q_int = clamp(q, BitWidth_Min, BitWidth_Max);
      sc_uint<BitWidth> q_fixed = q_int;
      quantized_channels[i] = (q_fixed >> M_Bits);
      diff[i] = q_fixed.range(M_Bits - 1, 0);
      ALOG("Quantized[" << i << "]: " << quantized_channels[i] << ", "
                        << diff[i] << ", " << q_fixed << ", " << q << endl);
    }
  }
  wait();
  // Quantize_si.write(4);
  // wait();

  huffman_encode();
}

void ACCNAME::Quantize() {
  done_quantize.write(false);
  Quantize_si.write(0);
  wait();
  while (true) {
    Quantize_si.write(1);
    wait();
    while (!start_quantize.read()) wait();
    Quantize_si.write(2);
    wait();
    quant_DQA();
    wait();
    Quantize_si.write(3);
    done_quantize.write(true);
    while (start_quantize.read()) wait();
    done_quantize.write(false);
    Quantize_si.write(4);
    wait();
  }
}
