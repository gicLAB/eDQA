void ACCNAME::huffman_reconstruct() {
  int node_count = 1; // root at 0
  for (int i = 0; i < TreeM_Range; ++i) {
    de_node_symbol[i] = -1;
    de_left[i] = 0;
    de_right[i] = 0;
    de_parent[i] = 0;
  }

  ALOG("=========================" << endl);
  ALOG("Encoding" << endl);
  ALOG("=========================" << endl);
  for (int i = 0; i < M_Range; ++i) {
#pragma HLS PIPELINE II = 1
    int length = encoding_length[i];
    if (length == 0) continue;
    Codeword code = encoding_cwords[i];
    ALOG("Symbol: " << i << " | Length: " << length << " | Code: ");
    int curr = 0;
    for (int j = 0; j < length; ++j) {
      bool bit = (code >> (j)) & 1;
      ALOG(bit << flush);
      if (bit) {
        if (de_right[curr] == 0) {
          de_right[curr] = node_count;
          de_parent[node_count] = curr;
          node_count++;
        }
        curr = de_right[curr];
      } else {
        if (de_left[curr] == 0) {
          de_left[curr] = node_count;
          de_parent[node_count] = curr;
          node_count++;
        }
        curr = de_left[curr];
      }
    }
    de_node_symbol[curr] = i;
    ALOG(endl);
  }

  ALOG("=========================" << endl);

#ifndef __SYNTHESIS__
#include <fstream>
#include <sstream>
  {
    std::ofstream dot_file("huffman_tree.dot");
    dot_file << "digraph HuffmanTree {\n";
    dot_file << "  edge [color=red];\n";
    for (int i = 0; i < node_count; ++i) {
      if (de_left[i] != 0) {
        dot_file << "  " << i << " -> " << de_left[i] << " [label=\"0\"];\n";
      }
      if (de_right[i] != 0) {
        dot_file << "  " << i << " -> " << de_right[i] << " [label=\"1\"];\n";
      }
    }
    for (int i = 0; i < node_count; ++i) {
      if (de_node_symbol[i] != -1) {
        dot_file << "  " << i << " [shape=box,label=\"" << de_node_symbol[i]
                 << "\",style=filled,fillcolor=green];\n";
      } else {
        dot_file << "  " << i << " [label=\"\"];\n";
      }
    }
    dot_file << "}\n";
    dot_file.close();
  }

#endif // __SYNTHESIS__
}

void ACCNAME::huffman_decode() {
  int rc_idx = 0;
  int curr = 0;
  for (int i = 0; i < MAX_CHANNEL_SIZE * 3; ++i) {
#pragma HLS PIPELINE II = 1

    bool bit = encoded_diff[i];
    int previous = curr;
    curr = bit ? de_right[curr] : de_left[curr];
    if (curr == 0) {
      break;
    } // error in decoding
    if (de_node_symbol[curr] != -1) {
      decoded_diff[rc_idx++] = de_node_symbol[curr];
      curr = 0;
      if (rc_idx >= channel_size) break;
    }
  }
  ALOG("=========================" << endl);
  ALOG("Decoded Data" << endl);
  ALOG("=========================" << endl);
  for (int i = 0; i < rc_idx; ++i) {
    ALOG(decoded_diff[i] << "|" << flush);
  }
  ALOG(endl);
  ALOG("=========================" << endl);
}

void ACCNAME::dequant_DQA() {
#pragma HLS PIPELINE II = 1
  // Decompression: decode encoded_diff back to decoded_diff using encoding
  // table Reconstruct Huffman tree using parent, left, and right arrays
  huffman_reconstruct();
  wait();

  // Decode encoded_diff bit by bit
  huffman_decode();
  wait();
  float dl = max_ele / BitWidth_PowNeg4;
  for (int i = 0; i < channel_size; ++i) {
#pragma HLS PIPELINE II = 1
    output_channels[i] = dl * (decoded_diff[i] + quantized_channels[i]);
  }
  wait();
}

void ACCNAME::DeQuantize() {
  DeQuantize_si.write(0);
  done_dequantize.write(false);
  wait();
  while (true) {
    DeQuantize_si.write(1);
    while (!start_dequantize.read()) wait();
    DeQuantize_si.write(2);
    wait();
    dequant_DQA();
    wait();
    DeQuantize_si.write(3);
    done_dequantize.write(true);
    while (start_dequantize.read()) wait();
    done_dequantize.write(false);
    DeQuantize_si.write(4);
    wait();
  }
}