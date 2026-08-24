#ifndef SYSTEMC_BINDING
#define SYSTEMC_BINDING

#ifdef SYSC

#include "../acc.sc.h"
#include "secda_tools/axi_support/v5/axi_api_v5.h"
#include "secda_tools/secda_integrator/sysc_types.h"
#include "secda_tools/secda_integrator/systemc_integrate.h"

// This file is specfic to VM SystemC definition
// This contains all the correct port/signal bindings to instantiate the VM
// accelerator
struct sysC_sigs {
  int id;
  Clock_Reset_Define;

  sysC_sigs(int id_) {
    id = id_;
    sc_clock clk_clock("ClkClock", 1, SC_NS);
  }
};

void sysC_binder(ACCNAME *acc, sysC_sigs *scs, a_ctrl *ctrl, h_ctrl *hwc,
                 mm_buf_float *inp_mem, mm_buf_float *out_mem) {

  Clock_Reset_Bind(acc, scs);
  Clock_Reset_Bind(ctrl->ctrl, scs);
  Clock_Reset_Bind(hwc->hwc_resetter, scs);

  CTRL_Bind_CtrlSignals(acc, ctrl);
  CTRL_Bind_RegSignals(input_addr);
  CTRL_Bind_RegSignals(output_addr);
  CTRL_Bind_RegSignals(channel_size_in);
  CTRL_Bind_RegSignals(num_channels_in);
  CTRL_Bind_RegSignals(channel_size_out);
  CTRL_Bind_RegSignals(num_channels_out);
  CTRL_Bind_RegSignals(compression_ratio_out);
  acc->input_port(inp_mem->buffer_chn);
  acc->output_port(out_mem->buffer_chn);

  HWC_Bind_Reset;
  HWC_Bind_Signals(Compute);
  HWC_Bind_Signals(Quantize);
  HWC_Bind_Signals(DeQuantize);
}
#endif // SYSC

#endif // SYSTEMC_BINDING