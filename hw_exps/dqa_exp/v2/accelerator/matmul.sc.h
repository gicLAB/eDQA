void ACCNAME::MatMul() {
  done_matmul.write(false);
  MatMul_si.write(0);
  wait();
  while (true) {
    MatMul_si.write(1);
    Sig_Wait(start_matmul);

    MatMul_si.write(2);
    wait();

    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        output[height * i + j] = 0;
        for (int k = 0; k < depth; ++k) {
          output[height * i + j] +=
              data[k * height + j] * weights[width * i + k];
        }
      }
    }

    // std::cout << "Input matrix:" << std::endl;
    // for (int i = 0; i < height; ++i) {
    //   for (int k = 0; k < depth; ++k) {
    //     std::cout << data[height * i + k] << " ";
    //   }
    //   std::cout << std::endl;
    // }

    // std::cout << "Weight matrix:" << std::endl;
    // for (int k = 0; k < depth; ++k) {
    //   for (int j = 0; j < width; ++j) {
    //     std::cout << weights[k * width + j] << " ";
    //   }
    //   std::cout << std::endl;
    // }

    // std::cout << "Output matrix:" << std::endl;
    // for (int i = 0; i < height; ++i) {
    //   for (int j = 0; j < width; ++j) {
    //     std::cout << output[height * i + j] << " ";
    //   }
    //   std::cout << std::endl;
    // }

    MatMul_si.write(3);
    wait();
    Sig_Done(start_matmul, done_matmul);
  }
}
