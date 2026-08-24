void ACCNAME::Compute() {
  done.write(false);
  Compute_si.write(0);
  start_quantize.write(false);
  start_dequantize.write(false);

  wait();
  while (true) {
    Compute_si.write(1);
    while (!start.read()) wait();
    Compute_si.write(2);
    wait();

    channel_size.write(channel_size_in.read());
    channel_count.write(num_channels_in.read());

    wait();
    channel_size_out.write(channel_size);
    num_channels_out.write(channel_count);

    unsigned int mem_base = input_addr.read();
    data_length = channel_size * channel_count;
    max_ele = 0;
    DWAIT();
    input_port->burst_read(mem_base, data_length, (float *)&channels[0]);
    for (int i = 0; i < data_length; i++) {
      ALOG("Input[" << i << "]: " << channels[i] << endl);
      float val = channels[i];
      if (val < 0) val = -val;
      if (val > max_ele) {
        max_ele = val;
      }
    }
    Compute_si.write(3);
    wait();

    // Start the quantization and compression process
    start_quantize.write(true);
    while (!done_quantize.read()) wait();
    start_quantize.write(false);
    Compute_si.write(4);
    wait();

    // Start the decompression and dequantization process
    start_dequantize.write(true);
    while (!done_dequantize.read()) wait();
    start_dequantize.write(false);
    Compute_si.write(5);
    wait();

    ALOG("===========================" << endl);
    ALOG("Encoded_Diff Size: " << encoded_diff_size << endl);
    ALOG("Diff Size: " << diff_size << endl);
    ALOG("Compression Ratio: " << (float)diff_size / (float)encoded_diff_size
                               << endl);
    ALOG("===========================" << endl);

    compression_ratio_out.write((float)diff_size / (float)encoded_diff_size);

    //  Ignore the output ports for now
    mem_base = output_addr.read();
    output_port->burst_write(mem_base, data_length,
                             (float *)&output_channels[0]);
    //  Ignore the output ports for now

    done.write(true);
    while (start.read()) wait();
    done.write(false);
  }
}
