#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef SYSC
#include "secda_tools/secda_integrator/systemc_integrate.h"
#endif

#include "accelerator/driver/driver.h"
#include "secda_tools/secda_profiler/profiler.h"

unsigned int dma_addrs[1] = {dma_addr0};
unsigned int dma_addrs_in[1] = {dma_in0};
unsigned int dma_addrs_out[1] = {dma_out0};
struct acc_times a_t;
static struct Profile profile;

#define DELLOG(X)

#ifdef SYSC
ACCNAME *acc;
struct sysC_sigs *scs;
static struct mm_buf_float inp_mem(0, MM_BL, "inp_mem");
static struct mm_buf_float out_mem(0, MM_BL, "out_mem");
#else
int *acc;
static struct mm_buf_float inp_mem(in_addr, MM_BL, "inp_mem");
static struct mm_buf_float out_mem(out_addr, MM_BL, "out_mem");
#endif

struct a_ctrl *ctrl;
static h_ctrl *hwc;

using namespace std;

void save_experiment_data_to_csv(const acc_container &drv,
                                 const std::string &filename, bool header) {
  bool file_exists = std::ifstream(filename).good();
  std::ofstream outfile(filename, std::ios::app);
  if (!outfile.is_open()) {
    std::cerr << "Error opening file for writing: " << filename << std::endl;
    return;
  }

  // Write CSV header only if file doesn't exist
  if (header) {
    outfile << "num_channels, channel_size, quantize_cycles, "
               "quantize_cycles_stdev, "
               "dequantize_cycles, dequantize_cycles_stdev, "
               "send_time, send_time_stdev, recv_time, recv_time_stdev, "
               "valid, "
               "total_elements, compression_ratio"
            << std::endl;
  }

  // Write experiment data
  outfile << drv.num_channels << ", " << drv.channel_size << ", "
          << drv.quantize_cycles << ", " << drv.quantize_cycles_stdev << ", "
          << drv.dequantize_cycles << ", " << drv.dequantize_cycles_stdev
          << ", " << drv.send_time << ", " << drv.send_time_stdev << ", "
          << drv.recv_time << ", " << drv.recv_time_stdev << ", "
          << (drv.valid ? "true" : "false") << ", "
          << (drv.num_channels * drv.channel_size) << ", "
          << (float)drv.compression_ratio << std::endl;
  outfile.close();
  std::cout << "Experiment data saved to: " << filename << std::endl;
}

void save_run_times_to_csv(const float quantize_vals[],
                           const float dequantize_vals[],
                           const float send_time_vals[],
                           const float recv_time_vals[], unsigned int num_runs,
                           const std::string &filename) {
  bool file_exists = std::ifstream(filename).good();
  std::ofstream outfile(filename, std::ios::app);
  if (!outfile.is_open()) {
    std::cerr << "Error opening file for writing: " << filename << std::endl;
    return;
  }

  if (!file_exists) {
    outfile << "run,quantize_cycles,dequantize_cycles,send_time,recv_time"
            << std::endl;
  }

  for (unsigned int run = 0; run < num_runs; ++run) {
    outfile << run << "," << quantize_vals[run] << "," << dequantize_vals[run]
            << "," << send_time_vals[run] << "," << recv_time_vals[run]
            << std::endl;
  }

  outfile.close();
  std::cout << "Run times saved to: " << filename << std::endl;
}

std::vector<float> load_data(const std::string &filename) {
  std::vector<float> data;
  std::ifstream infile(filename);
  if (!infile.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return data;
  }
  std::string line;
  while (std::getline(infile, line)) {
    std::stringstream ss(line);
    std::string value;
    while (std::getline(ss, value, ',')) {
      if (!value.empty()) {
        data.push_back(std::stof(value));
      }
    }
    break; // Assuming one line of data
  }
  infile.close();
  return data;
}

std::vector<float> load_input_data(int num_channels, int channel_size) {
  std::string filename = "data/" + std::to_string(num_channels) + "_" +
                         std::to_string(channel_size) + ".csv";
  return load_data(filename);
}

std::vector<float> load_golden_data(int num_channels, int channel_size) {
  std::string filename = "data/" + std::to_string(num_channels) + "_" +
                         std::to_string(channel_size) + "_golden_dqa.csv";
  return load_data(filename);
}

int main() {
  // ========================================
  // ========================================
  // Initialize the Accelerator

  DELLOG(std::cout << "===========================" << std::endl;);
#ifdef SYSC
  static ACCNAME _acc("ACCNAME");
  static struct sysC_sigs scs1(1);
  static struct a_ctrl ctrl1;
  static struct h_ctrl hwc1;
  sysC_init();
  hwc1.init_hwc(HWC_Monitor_Count);
  ctrl1.init_sigs(CTRL_Reg_Count);
  sysC_binder(&_acc, &scs1, &ctrl1, &hwc1, &inp_mem, &out_mem);
  acc = &_acc;
  scs = &scs1;
  ctrl = &ctrl1;
  hwc = &hwc1;
  ctrl->set_reg(0, 0); // Set the input address
  ctrl->set_reg(1, 0); // Set the output address
  DELLOG(std::cout << "Initialised the SystemC Modules" << std::endl;);

#else
  acc = getAccBaseAddress<int>(acc_ctrl_address, 65536);
  int *acc_ctrl_base = getAccBaseAddress<int>(acc_ctrl_address, 65536);
  int *hwc_base = getAccBaseAddress<int>(acc_hwc_address, 65536);

  static struct a_ctrl ctrl1(acc_ctrl_base);
  static struct h_ctrl hwc1(hwc_base);
  hwc1.init_hwc(HWC_Monitor_Count);
  ctrl1.init_sigs(CTRL_Reg_Count);

  ctrl = &ctrl1;
  hwc = &hwc1;

  ctrl->set_reg(0, in_addr / 4);  // Set the input address
  ctrl->set_reg(1, out_addr / 4); // Set the output address
  DELLOG(std::cout << "Initialised the DMA" << std::endl;);
#endif
  DELLOG(std::cout << "ACCNAME Accelerator";);
  DELLOG(std::cout << std::endl;);
  DELLOG(std::cout << "===========================" << std::endl;);

  // ========================================
  // ========================================
  // FPGA Driver Initialization
  acc_container drv;

#ifdef SYSC
  drv.scs = scs;
#endif
  drv.profile = &profile;
  drv.acc = acc;
  drv.ctrl = ctrl;
  drv.hwc = hwc;
  drv.a_t = &a_t;
  drv.inp_mem = &inp_mem;
  drv.out_mem = &out_mem;

  // ========================================
  // Run Experiments
  // ========================================

  std::vector<int> channel_sizes = {49, 196, 784, 3136};

  // std::vector<int> channel_sizes = {49};
  // std::vector<int> channel_sizes = {10,10}; // Example sizes

  // std::vector<int> channel_sizes = {784};

  // Save experiment data to CSV
  std::string csv_filename = "dqa_experiment_results.csv";
  std::ofstream(csv_filename, std::ios::trunc).close();

  unsigned int Num_runs = 20;
  for (int i = 0; i < channel_sizes.size(); i++) {
    int num_channels = 1;
    int channel_size = channel_sizes[i];

    // Run each experiment 5 times and save average
    float total_quantize_cycles = 0;
    float total_dequantize_cycles = 0;
    float total_compression_ratio = 0;
    float total_send_time = 0;
    float total_recv_time = 0;
    int valid_count = 0;

    float quantize_cycles_stdev = 0;
    float dequantize_cycles_stdev = 0;

    float quantize_cycles_values[Num_runs];
    float dequantize_cycles_values[Num_runs];

    float send_time_values[Num_runs];
    float recv_time_values[Num_runs];

    for (int run = 0; run < Num_runs; ++run) {
      std::vector<float> input_data =
          load_input_data(num_channels, channel_size);
      std::vector<float> golden_data =
          load_golden_data(num_channels, channel_size);
      drv.input_data = &input_data;
      drv.golden_data = &golden_data;
      drv.num_channels = num_channels;
      drv.channel_size = channel_size;

      prf_start(1);
      acc_driver::Entry(drv);
      prf_end(1, a_t.fpga_total);

      total_quantize_cycles += drv.quantize_cycles;
      total_dequantize_cycles += drv.dequantize_cycles;
      total_compression_ratio += drv.compression_ratio;
      quantize_cycles_values[run] = drv.quantize_cycles;
      dequantize_cycles_values[run] = drv.dequantize_cycles;
      send_time_values[run] = a_t.send_time.count();
      recv_time_values[run] = a_t.recv_time.count();
      total_send_time += a_t.send_time.count();
      total_recv_time += a_t.recv_time.count();
      a_t.reset();
    }

    // Calculate standard deviation
    float quantize_mean = total_quantize_cycles / Num_runs;
    float dequantize_mean = total_dequantize_cycles / Num_runs;
    for (int run = 0; run < Num_runs; ++run) {
      quantize_cycles_stdev += (quantize_cycles_values[run] - quantize_mean) *
                               (quantize_cycles_values[run] - quantize_mean);
      dequantize_cycles_stdev +=
          (dequantize_cycles_values[run] - dequantize_mean) *
          (dequantize_cycles_values[run] - dequantize_mean);
    }
    quantize_cycles_stdev = std::sqrt(quantize_cycles_stdev / Num_runs);
    dequantize_cycles_stdev = std::sqrt(dequantize_cycles_stdev / Num_runs);

    // Store averages and standard deviations in drv before saving
    drv.quantize_cycles = total_quantize_cycles / Num_runs;
    drv.dequantize_cycles = total_dequantize_cycles / Num_runs;
    drv.compression_ratio = total_compression_ratio / Num_runs;
    drv.quantize_cycles_stdev = quantize_cycles_stdev;
    drv.dequantize_cycles_stdev = dequantize_cycles_stdev;
    drv.valid = (valid_count == 5); // Only true if all runs were valid

    drv.send_time = total_send_time / Num_runs;
    drv.recv_time = total_recv_time / Num_runs;

    float send_time_stdev = 0;
    float recv_time_stdev = 0;

    for (int run = 0; run < Num_runs; ++run) {
      send_time_stdev += (send_time_values[run] - drv.send_time) *
                         (send_time_values[run] - drv.send_time);
      recv_time_stdev += (recv_time_values[run] - drv.recv_time) *
                         (recv_time_values[run] - drv.recv_time);
    }
    drv.send_time_stdev = std::sqrt(send_time_stdev / Num_runs);
    drv.recv_time_stdev = std::sqrt(recv_time_stdev / Num_runs);
    DELLOG(cout << "FPGA Done!" << endl;);
    save_experiment_data_to_csv(drv, csv_filename, i == 0);
    string run_times_csv_filename =
        "dqa_run_times_" + std::to_string(channel_size) + ".csv";
    save_run_times_to_csv(quantize_cycles_values, dequantize_cycles_values,
                          send_time_values, recv_time_values, Num_runs,
                          run_times_csv_filename);
  }

  // ========================================

  a_t.print();

  return 0;
}
