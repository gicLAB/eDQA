#ifndef ACC_CONFIG_H
#define ACC_CONFIG_H

#define ACCNAME ACC
#define SUBMODULENAME acc_pe

//==============================================================================
// Hardware Constants
//==============================================================================
// Define any Hardware specific constants for the accelerator
// These constants will be accessible in the driver
// These constants will be used to generate the hardware

//==============================================================================
// Address mapping for the accelerator and DMA
//==============================================================================
#ifdef KRIA
// KRIA
// Pre-Defined Address for Accelerator
#define acc_ctrl_address 0x00A0000000
#define acc_hwc_address 0x00A0010000

#define dma_addr0 0x00A0010000
#define dma_addr1 0x00A0020000
#define dma_addr2 0x00A0030000
#define dma_addr3 0x00A0040000

#define DMA_BL 4194304
#define DMA_RANGE_START 0x0000000037400000
#define DMA_RANGE_END 0x00000000773FFFFF
#define DMA_RANGE_OFFSET 0xC00000         // 1.5MB
#define DMA_RANGE_SIZE 0x0000000040000000 // 1GB
#define DMA_IN_BUF_SIZE 0x20000000        // 32MB
#define DMA_OUT_BUF_SIZE 0x20000000       // 32MB

#define dma_in0 0x38000000
#define dma_in1 0x3A000000
#define dma_in2 0x3C000000
#define dma_in3 0x3E000000

#define dma_out0 0x39000000
#define dma_out1 0x3B000000
#define dma_out2 0x3D000000
#define dma_out3 0x40000000

#else
// Z1
// Pre-Defined Address for Accelerator
#define acc_ctrl_address 0x43C00000
#define acc_hwc_address 0x43C10000

#define dma_addr0 0x40400000
#define dma_addr1 0x40410000
#define dma_addr2 0x40420000
#define dma_addr3 0x40430000

#define dma_in0 0x18000000
#define dma_in1 0x1a000000
#define dma_in2 0x1c000000
#define dma_in3 0x1e000000
#define dma_out0 0x18800000
#define dma_out1 0x1a800000
#define dma_out2 0x1c800000
#define dma_out3 0x1e800000

#define DMA_BL 4194304
#define DMA_RANGE_START 0x18000000
#define DMA_RANGE_END 0x1fffffff
#define DMA_RANGE_SIZE 0x8000000
#endif // KRIA

// AXIMM Constants
#ifdef KRIA
#define MM_BL 0x100000 // 1MB
#define in_addr 0x38000000
#define out_addr 0x39000000
#else
// Z1
#define MM_BL 0x100000 // 1MB
#define in_addr 0x18000000
#define out_addr 0x19000000
#endif

//==============================================================================
// Data types
//==============================================================================
#define ACC_DTYPE sc_int
#define ACC_C_DTYPE int
#define AXI_DWIDTH 32
#define AXI_TYPE sc_uint
#define s_mdma multi_dma<AXI_DWIDTH, 0>
#define mm_buf mm_buffer<unsigned long long>
#define mm_buf_float mm_buffer<float>

#define a_ctrl acc_ctrl<int>
#define h_ctrl hwc_ctrl<int>

//==============================================================================
// ACC Specific Constants
//==============================================================================

// Use  {1: 0.25, 2: 0.5, 3: 0.75, 0: 0.0} as random data values

// ==============
// Buffer sizes
// ==============

#define MAX_CHANNEL_SIZE 3136 // 56 * 56
#define INPUT_SIZE MAX_CHANNEL_SIZE * 1

// ==============
// ACC Specific Constants
// ==============
#define STOPPER -1

// #define BIAS_SEARCH_SPACE 100
#define PERCENTILE_SEARCH_SPACE 100
#define MEAN_SEARCH_SPACE 50

#define BIAS_SEARCH_SPACE 5

// #define PERCENTILE_SEARCH_SPACE 10
// #define MEAN_SEARCH_SPACE 5

// #define MEAN_SEARCH_SPACE 5

#define N_Bits 3
#define M_Bits 0
#define BitWidth (N_Bits + M_Bits)
#define BitWidth_IntRange (1 << (BitWidth - 1))
#define BitWidth_Max (BitWidth_IntRange - 1)
#define BitWidth_Min (-BitWidth_IntRange)

#define M_Range (1 << M_Bits)

#define BitWidth_PowNeg4 (1 << (BitWidth - 4))

#define TREE_DEPTH (M_Range - 1)
#define TREE_DEPTH_BITS 3

#define TreeM_Range (M_Range * 2)

#define MAX_CODEWORD_LENGTH 3
#define MAX_CODEWORD_SIZE 27

#define CODEWORD_LENGTH_BITS 2
const unsigned int RADIX = 16;
const unsigned int BITS_PER_LOOP = 4; // should be log2(RADIX)

#define SYMBOL sc_uint<M_Bits>
#define Digit sc_uint<BITS_PER_LOOP>

#define Codeword sc_uint<MAX_CODEWORD_LENGTH>
#define PackedCodewordAndLength                                                \
  sc_uint<MAX_CODEWORD_LENGTH + CODEWORD_LENGTH_BITS>
#define CodewordLength sc_uint<CODEWORD_LENGTH_BITS>

// ==============
// Signal Counts
// ==============

#define HWC_Monitor_Count 4
#define CTRL_Reg_Count 6

#define s_mdma multi_dma<AXI_DWIDTH, 0>
#define mm_buf mm_buffer<unsigned long long>
#define a_ctrl acc_ctrl<int>

//==============================================================================
// SystemC Specfic SIM/HW Configurations
//==============================================================================
#if defined(SYSC) || defined(__SYNTHESIS__)
#include <systemc.h>

#ifndef __SYNTHESIS__
#include "secda_tools/axi_support/v5/axi_api_v5.h"
#include "secda_tools/secda_integrator/sysc_types.h"
#include "secda_tools/secda_profiler/profiler.h"
#define DWAIT(x) wait(x)
#define DPROF(x) x

#define VERBOSE_ACC
#ifdef VERBOSE_ACC
#define ALOG(x) std::cout << x << std::flush
#else // !VERBOSE_ACC
#define ALOG(x)
#endif

typedef _BDATA<AXI_DWIDTH, AXI_TYPE> ADATA;

#else // __SYNTHESIS__
#include "sysc_types.h"
#define ALOG(x)

struct _NDATA {
  AXI_TYPE<AXI_DWIDTH> data;
  bool tlast;
  inline friend ostream &operator<<(ostream &os, const _NDATA &v) {
    cout << "data&colon; " << v.data << " tlast: " << v.tlast;
    return os;
  }
};

typedef _NDATA ADATA;
#define DPROF(x)
#endif

//==============================================================================
// HW Submodule Construction SIM/HW Structs
//==============================================================================

//==============================================================================

#endif // defined(SYSC) || defined(__SYNTHESIS__)
#endif // ACC_CONFIG_H