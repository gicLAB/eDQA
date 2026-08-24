#ifndef ACC_DRIVER
#define ACC_DRIVER

#include "acc_container.h"

#define DLOG(X)

namespace acc_driver {

void ACC_Offload(acc_container &drv) {
  int length = drv.num_channels * drv.channel_size;

  drv.hwc->set_target_state(0, 3); // Compute
  drv.hwc->set_target_state(1, 2); // Quantize
  drv.hwc->set_target_state(2, 2); // DeQuantize
  drv.hwc->reset_hwc();

  DLOG(cout << "HWC Reset Done" << endl;);

  float *inp_mem = reinterpret_cast<float *>(drv.inp_mem->get_buffer());
  for (int i = 0; i < length; i++) inp_mem[i] = (*drv.input_data)[i];

  drv.ctrl->set_reg(2, drv.channel_size);
  drv.ctrl->set_reg(3, drv.num_channels);
  DLOG(cout << "Set Ctrl Reg" << endl;);

  prf_start(0);
  drv.inp_mem->sync_to_acc();
  prf_end(0, drv.a_t->send_time);
  DLOG(cout << "Input Memory Synced to Accelerator" << endl;);

  DLOG(cout << "Starting Accelerator" << endl;);
  drv.ctrl->start_acc();
  DLOG(cout << "Waiting for Accelerator to complete" << endl;);
  drv.ctrl->wait_done();
  // while (!drv.ctrl->check_done()) {
  //   drv.hwc->print_hwc_map(true);
  //   drv.ctrl->print_reg_map(false);
  // }
  DLOG(cout << "Accelerator completed" << endl;);
  prf_start(1);
  drv.out_mem->sync_from_acc();
  prf_end(1, drv.a_t->recv_time);
  DLOG(cout << "Output Memory Synced from Accelerator" << endl;);
  drv.ctrl->print_reg_map(false);
  drv.hwc->print_hwc_map(false);

  // Validate the output
  drv.valid = true;
  for (int i = 0; i < length; i++) {
    float out_val = reinterpret_cast<float *>(drv.out_mem->get_buffer())[i];
    if (out_val != (*drv.golden_data)[i]) {
      cerr << "Output mismatch at index " << i << ": expected "
           << (*drv.golden_data)[i] << ", got " << out_val << endl;
      drv.valid = false;
    }
  }

  drv.quantize_cycles =
      drv.hwc->get_cycle_count(1); // Store quantization cycles
  drv.dequantize_cycles =
      drv.hwc->get_cycle_count(2); // Store dequantization cycles

  drv.compression_ratio = drv.ctrl->get_reg(6);

  cout << "========================================" << endl;
  cout << "Experiment Summary" << endl;
  cout << "Input Channels: " << drv.num_channels << endl;
  cout << "Channel Size: " << drv.channel_size << endl;
  cout << "Quantization Cycles: " << drv.quantize_cycles << endl;
  cout << "Dequantization Cycles: " << drv.dequantize_cycles << endl;
  // cout << "Compression Ratio: " << drv.compression_ratio << endl;
  cout << "Output Valid: " << (drv.valid ? "Yes" : "No") << endl;
  cout << "========================================" << endl;
}

void Entry(acc_container &drv) { ACC_Offload(drv); }

} // namespace acc_driver

#endif // ACC_DRIVER