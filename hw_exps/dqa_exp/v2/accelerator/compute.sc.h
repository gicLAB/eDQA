void ACCNAME::Compute() {
  done.write(false);
  start_quantize.write(false);
  start_dequantize.write(false);
  Compute_si.write(0);

  wait();
  while (true) {
    Compute_si.write(1);
    Sig_Wait(start);
    Compute_si.write(2);
    wait();

    channel_size.write(channel_size_in.read());
    height.write(dim_in.read());
    width.write(dim_in.read());
    depth.write(dim_in.read());
    wait();

    channel_size_out.write(channel_size);
    dim_out.write(dim_in.read());
    Compute_si.write(3);
    wait();

    unsigned int mem_base = input_addr.read();
    input_port->burst_read(mem_base, channel_size, (float *)&channels[0]);
    for (int i = 0; i < channel_size; i++) {
      float val = channels[i];
      weights[i] = val;
      if (val < 0) val = -val;
      if (val > max_ele) max_ele = val;
    }

    // Start the quantization
    Compute_si.write(6);
    Sig_Start(start_quantize, done_quantize);

    // Start the dequantization
    Compute_si.write(7);
    Sig_Start(start_dequantize, done_dequantize);
    wait();

    //  Output data
    mem_base = output_addr.read();
    output_port->burst_write(mem_base, channel_size,
                             (float *)&output_channels[0]);

    Sig_Done(start, done);
  }
}
