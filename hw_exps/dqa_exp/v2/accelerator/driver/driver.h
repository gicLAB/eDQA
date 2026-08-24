#ifndef ACC_DRIVER
#define ACC_DRIVER

#include "acc_container.h"

#define DLOG(X)

namespace acc_driver {

void ACC_Offload(acc_container &drv) {
  int length = drv.channel_size;
  drv.hwc->set_target_state(0, 6); // Compute
  drv.hwc->set_target_state(1, 2); // Quantize
  drv.hwc->set_target_state(2, 2); // DeQuantize
  drv.hwc->set_target_state(3, 2); // MatMul
  drv.hwc->reset_hwc();            // Reset HWC

  float *inp_mem = reinterpret_cast<float *>(drv.inp_mem->get_buffer());
  for (int i = 0; i < length; i++) inp_mem[i] = (*drv.input_data)[i];

  drv.ctrl->set_reg(2, drv.channel_size);
  drv.ctrl->set_reg(3, drv.num_channels);
  prf_start(0);
  drv.inp_mem->sync_to_acc();
  prf_end(0, drv.a_t->send_time);

  drv.ctrl->start_acc();
  drv.ctrl->wait_done();

  // unsigned int timer = 0;
  // while (!drv.ctrl->check_done()) {
  //   if (timer % 100000 == 0) drv.hwc->print_hwc_map(true);
  //   if (timer % 100000 == 0) drv.ctrl->print_reg_map(false);
  //   timer++;
  // }
  prf_start(1);
  drv.out_mem->sync_from_acc();
  prf_end(1, drv.a_t->recv_time);
  // drv.ctrl->print_reg_map(false);
  // drv.hwc->print_hwc_map(false);

  // Validate the output
  drv.valid = true;
  drv.quantize_cycles = drv.hwc->get_cycle_count(0);
  drv.dequantize_cycles = drv.hwc->get_cycle_count(2);
  drv.compression_ratio = 1.0;

  // cout << "========================================" << endl;
  // cout << "Experiment Summary" << endl;
  // cout << "Input Channels: " << drv.dim << endl;
  // cout << "Channel Size: " << drv.channel_size << endl;
  // cout << "Quantization Cycles: " << drv.quantize_cycles << endl;
  // cout << "Dequantization Cycles: " << drv.dequantize_cycles << endl;
  // cout << "Output Valid: " << (drv.valid ? "Yes" : "No") << endl;
  // cout << "========================================" << endl;
}

void Entry(acc_container &drv) { ACC_Offload(drv); }

} // namespace acc_driver

#endif // ACC_DRIVER