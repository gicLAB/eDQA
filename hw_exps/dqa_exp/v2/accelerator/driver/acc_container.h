#ifndef ACC_CONTAINER
#define ACC_CONTAINER

#include <cassert>
#include <iomanip>
#include <vector>

#ifdef SYSC
#include "systemc_binding.h"
#else
#endif

#include "../acc_config.sc.h"
#include "secda_tools/axi_support/v5/axi_api_v5.h"
#include "secda_tools/secda_profiler/profiler.h"
#include "secda_tools/secda_utils/multi_threading.h"
#include "secda_tools/secda_utils/utils.h"

#ifdef ACC_NEON
#include "arm_neon.h"
#endif

using namespace std;
using namespace std::chrono;
#define TSCALE nanoseconds
#define TSCAST duration_cast<nanoseconds>

struct acc_times {
  duration_ns fpga_total;
  duration_ns cpu_total;
  duration_ns send_time;
  duration_ns recv_time;
  duration_ns driver;

  void print() {
#ifdef ACC_PROFILE
    cout << "================================================" << endl;
    prf_out(TSCALE, fpga_total);
    prf_out(TSCALE, cpu_total);
    prf_out(TSCALE, send_time);
    prf_out(TSCALE, recv_time);
    prf_out(TSCALE, driver);
    cout << "================================================" << endl;
#endif
    reset();
  }
  void save_prf() {
#ifdef ACC_PROFILE
    std::ofstream file("prf.csv", std::ios::out);
    prf_file_out(TSCALE, fpga_total, file);
    prf_file_out(TSCALE, cpu_total, file);
    prf_file_out(TSCALE, send_time, file);
    prf_file_out(TSCALE, recv_time, file);
    prf_file_out(TSCALE, driver, file);
    file.close();
#endif
  }

  void reset() {
    fpga_total = duration_ns::zero();
    cpu_total = duration_ns::zero();
    send_time = duration_ns::zero();
    recv_time = duration_ns::zero();
    driver = duration_ns::zero();
  }
};

struct offload_details {
  int count = 0;
  bool profile = false;
};

struct acc_container {
// Hardware
#ifdef SYSC
  ACCNAME *acc;
  struct sysC_sigs *scs;
#else
  int *acc;
#endif

  struct a_ctrl *ctrl;
  struct h_ctrl *hwc;

  Profile *profile;

  // Problem Specific Parameters

  // Data
  struct mm_buf_float *inp_mem;
  struct mm_buf_float *out_mem;

  vector<float> *input_data;
  vector<float> *golden_data;

  // Experiment Data
  unsigned quantize_cycles = 0;
  unsigned dequantize_cycles = 0;
  int num_channels = 0;
  int channel_size = 0;
  float compression_ratio = 0.0;
  bool valid = false;
  float quantize_cycles_stdev = 0.0;
  float dequantize_cycles_stdev = 0.0;

  float send_time;
  float recv_time;
  float send_time_stdev;
  float recv_time_stdev;

  // Debugging
  struct offload_details t;
  struct acc_times *a_t;
};

#endif // ACC_CONTAINER