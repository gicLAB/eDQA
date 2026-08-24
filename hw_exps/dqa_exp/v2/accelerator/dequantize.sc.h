
void ACCNAME::dequant_activation() {
  for (int i = 0; i < channel_size; ++i) {
    output_channels[i] = quantized_channels[i] * (act_scale_global);
  }
}

void ACCNAME::sub_noise(float noise) {
  for (int i = 0; i < channel_size; ++i) {
    output_channels[i] -= noise;
  }
}

void ACCNAME::DeQuantize() {
  done_dequantize.write(false);
  DeQuantize_si.write(0);
  wait();
  while (true) {
    DeQuantize_si.write(1);
    Sig_Wait(start_dequantize);
    DeQuantize_si.write(2);
    wait();

    dequant_activation();
    wait();
    sub_noise(noise_global);
    wait();

    Sig_Done(start_dequantize, done_dequantize);
  }
}