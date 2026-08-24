#include <bitset>

#define GET_CODEWORD(cw, length)                                               \
  std::bitset<64>(cw).to_string().substr(64 - length, length)

void ACCNAME::create_codeword(
    CodewordLength symbol_bits[M_Range],
    unsigned int codeword_length_histogram[TREE_DEPTH]) {

  Codeword fcw[MAX_CODEWORD_LENGTH + 1] = {0};

  // Computes the initial codeword value for a symbol with bit length i
  fcw[0] = 0;

  for (int i = 1; i <= MAX_CODEWORD_LENGTH; i++) {
#pragma HLS PIPELINE II = 1
    fcw[i] = (fcw[i - 1] + codeword_length_histogram[i - 1]) << 1;
  }

  for (int i = 0; i < M_Range; ++i) {
#pragma HLS PIPELINE II = 5
    CodewordLength length = symbol_bits[i];
    // if symbol has 0 bits, it doesn't need to be encoded
    if (length != 0) {
      Codeword cw = fcw[length];

      // Reverse the bits in the codeword
      for (int j = 0; j < MAX_CODEWORD_LENGTH; j++) {
#pragma HLS UNROLL
        cw[j] = fcw[length][MAX_CODEWORD_LENGTH - 1 - j];
      }
      cw >>= (MAX_CODEWORD_LENGTH - (length));
      encoding_length[i] = length;
      encoding_cwords[i] = cw;
      fcw[length]++;

    } else {
      encoding_length[i] = 0;
    }
  }
}

void ACCNAME::encode(int channel_size) {

  // cdiff is the data that we want to encode
  // encoding is the codeword for each symbol
  // encoded_diff is the encoded data
  int j = 0;
  ALOG("=========================" << endl);
  ALOG("Encoded Data" << endl);
  ALOG("=========================" << endl);

  for (int i = 0; i < channel_size; i++) {
    int dex = diff[i];
    int codeword_length = encoding_length[dex];
    Codeword codeword = encoding_cwords[dex];
    for (int k = 0; k < codeword_length; k++) {
#pragma HLS PIPELINE II = 1
      encoded_diff[j++] = codeword[k];
    }
    encoded_diff_size += codeword_length;
    diff_size += M_Bits;
    ALOG(dex << "|" << std::flush);
  }
  ALOG(endl);
  for (int i = 0; i < MAX_CHANNEL_SIZE * 3; i++) {
    ALOG(encoded_diff[i] << std::flush);
  }
  ALOG(endl);
  ALOG("=========================" << endl);
}