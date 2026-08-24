
void ACCNAME::canonize_tree(SymFreq sorted[M_Range], int num_symbols,
                            unsigned int codeword_length_histogram[TREE_DEPTH],
                            CodewordLength symbol_bits[M_Range]) {

  DPROF(assert(num_symbols <= M_Range));
  for (int i = 0; i < M_Range; i++) {
    symbol_bits[i] = 0;
  }

  int length = TREE_DEPTH;
  int count = 0;

  for (int k = 0; k < num_symbols; k++) {
    if (count == 0) {
      do {
#pragma HLS LOOP_TRIPCOUNT min = 1 avg = 1 max = 2
        length--;
        count = codeword_length_histogram[length];
      } while (count == 0);
    }
    symbol_bits[sorted[k].value] = length;
    count--;
  }
}

void ACCNAME::truncate_tree(
    unsigned int input_length_histogram[TREE_DEPTH],
    unsigned int output_length_histogram_0[TREE_DEPTH],
    unsigned int output_length_histogram_1[TREE_DEPTH]) {

  for (int i = 0; i < TREE_DEPTH; i++) {
    output_length_histogram_0[i] = input_length_histogram[i];
  }

  SYMBOL j = MAX_CODEWORD_LENGTH;
  for (int i = TREE_DEPTH - 1; i > MAX_CODEWORD_LENGTH; i--) {

    while (output_length_histogram_0[i] != 0) {
#pragma HLS LOOP_TRIPCOUNT min = 3 max = 3 avg = 3
      if (j == MAX_CODEWORD_LENGTH) {
        do {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1 avg = 1
          j--;
        } while (output_length_histogram_0[j] == 0);
      }

      // Move leaf with depth i to depth j+1.
      output_length_histogram_0[j] -= 1;
      output_length_histogram_0[j + 1] += 2;
      output_length_histogram_0[i - 1] += 1;
      output_length_histogram_0[i] -= 2;
      j++;
    }
  }

  // Copy the output to meet dataflow requirements and check the validity
  unsigned int limit = 1;
  for (int i = 0; i < TREE_DEPTH; i++) {
    output_length_histogram_1[i] = output_length_histogram_0[i];
    DPROF(assert(output_length_histogram_0[i] >= 0));
    DPROF(assert(output_length_histogram_0[i] <= limit));
    limit *= 2;
  }
}

void ACCNAME::compute_bit_length(SYMBOL parent[M_Range - 1],
                                 SymNode left[M_Range - 1],
                                 SymNode right[M_Range - 1], int num_symbols,
                                 unsigned int length_histogram[TREE_DEPTH]) {

  DPROF(assert(num_symbols > 0));
  DPROF(assert(num_symbols <= M_Range));

  for (int i = 0; i < TREE_DEPTH; i++) {
#pragma HLS pipeline II = 1
    internal_length_histogram[i] = 0;
    length_histogram[i] = 0;
  }
  for (int i = 0; i < TreeM_Range - 1; i++) {
#pragma HLS pipeline II = 1
    child_depth[i] = 0;
  }

  child_depth[num_symbols - 2] = 1;
  for (int i = num_symbols - 3; i >= 0; i--) {
#pragma HLS pipeline II = 3
    sc_uint<TREE_DEPTH_BITS> length = child_depth[parent[i]] + 1;

    child_depth[i] = length;
    if (!left[i].internal || !right[i].internal) {
      int children = 0;
      if (!left[i].internal) children++;
      if (!right[i].internal) children++;

      unsigned int count = internal_length_histogram[length];
      count += children;
      internal_length_histogram[length] = count;
      length_histogram[length] = count;
    }
  }
}

void ACCNAME::create_tree(SymFreq in[M_Range], int num_symbols,
                          SYMBOL parent[M_Range - 1], SymNode left[M_Range - 1],
                          SymNode right[M_Range - 1]) {
  unsigned int tree_count = 0;
  unsigned int in_count = 0;
  DPROF(assert(num_symbols > 0));
  DPROF(assert(num_symbols <= M_Range));

  for (int i = 0; i < TreeM_Range; i++) {
#pragma HLS PIPELINE II = 1
    tree_frequency[i] = 0;
  }

  for (int i = 0; i < (num_symbols - 1); i++) {
#pragma HLS PIPELINE II = 5

    unsigned int node_freq = 0;
    DPROF(assert(in_count < num_symbols || tree_count < i));
    unsigned int intermediate_freq = tree_frequency[tree_count];
    SymFreq s = in[in_count];
    if ((in_count < num_symbols && s.frequency <= intermediate_freq) ||
        tree_count == i) {
      left[i].value = s.value;
      left[i].internal = false;
      node_freq = s.frequency;
      in_count++;
    } else {
      left[i].value = -1;
      left[i].internal = true;
      node_freq = tree_frequency[tree_count];
      parent[tree_count] = i;
      tree_count++;
    }

    DPROF(assert(in_count < num_symbols || tree_count < i));

    intermediate_freq = tree_frequency[tree_count];
    s = in[in_count];
    if ((in_count < num_symbols && s.frequency <= intermediate_freq) ||
        tree_count == i) {
      right[i].value = s.value;
      right[i].internal = false;
      tree_frequency[i] = node_freq + s.frequency;
      in_count++;
    } else {
      right[i].value = -1;
      right[i].internal = true;
      tree_frequency[i] = node_freq + intermediate_freq;
      parent[tree_count] = i;
      tree_count++;
    }
    DPROF(assert(i == 0 || tree_frequency[i] >= tree_frequency[i - 1]));
  }
  parent[tree_count] = 0;
}

void ACCNAME::count_diff_frequency() {
  for (unsigned int i = 0; i < channel_size; ++i) {
    freq[i] = 0; // Initialize frequency array
  }

  for (unsigned int i = 0; i < channel_size; ++i) {
    freq[diff[i]]++; // Count frequency of each diff
  }
}

void ACCNAME::filter(int *num_symbols) {
#pragma HLS INLINE off
  int j = 0;
  for (int i = 0; i < M_Range; i++) {
#pragma HLS pipeline II = 1
    if (freq[i] != 0) {
      filtered[j].frequency = freq[i];
      filtered[j].value = i;
      j++;
    }
  }
  *num_symbols = j;
}

void ACCNAME::sort(int num_symbols) {
  SymFreq previous_sorting[M_Range];
  SymFreq sorting[M_Range];
  SYMBOL digit_histogram[RADIX];
  SYMBOL digit_location[RADIX];
#pragma HLS ARRAY_PARTITION variable = digit_location complete dim = 1
#pragma HLS ARRAY_PARTITION variable = digit_histogram complete dim = 1

  Digit current_digit[M_Range];

  for (int j = 0; j < num_symbols; j++) {
#pragma HLS PIPELINE II = 1
    sorting[j] = filtered[j];
  }

  for (int shift = 0; shift < 32; shift += BITS_PER_LOOP) {

    for (int i = 0; i < RADIX; i++) {
#pragma HLS pipeline II = 1
      digit_histogram[i] = 0;
    }

    for (int j = 0; j < num_symbols; j++) {
#pragma HLS PIPELINE II = 1
      Digit digit = (sorting[j].frequency >> shift) & (RADIX - 1);
      current_digit[j] = digit;
      digit_histogram[digit]++;
      previous_sorting[j] = sorting[j];
    }

    digit_location[0] = 0;

    for (int i = 1; i < RADIX; i++)
#pragma HLS PIPELINE II = 1
      digit_location[i] = digit_location[i - 1] + digit_histogram[i - 1];

    for (int j = 0; j < num_symbols; j++) {
#pragma HLS PIPELINE II = 1
      Digit digit = current_digit[j];
      sorting[digit_location[digit]] = previous_sorting[j];
      sorted[digit_location[digit]] = previous_sorting[j];
      digit_location[digit]++;
    }
  }
}

void ACCNAME::huffman_encode() {

  int num_symbols;

  // Quantize_si.write(5);
  // wait();

  count_diff_frequency(); // Count frequency of each diff

  // Quantize_si.write(6);
  // wait();
  filter(&num_symbols); // Filter out symbols with zero frequency

  // Quantize_si.write(7);
  // wait();
  sort(num_symbols);

  ALOG("=========================" << endl);
  ALOG("Sorted symbols by frequency" << endl);
  ALOG("=========================" << endl);

  for (int i = 0; i < num_symbols; i++) {
    ALOG("Value: " << sorted[i].value << " |  freq: " << sorted[i].frequency
                   << "\n");
  }
  ALOG("=========================" << endl);

  // Could make these SYMBOL to save space

  for (int i = 0; i < TREE_DEPTH; i++) {
#pragma HLS PIPELINE II = 1
    length_histogram[i] = 0;
    truncated_length_histogram1[i] = 0;
    truncated_length_histogram2[i] = 0;
  }
  for (int i = 0; i < M_Range; i++) {
#pragma HLS PIPELINE II = 1
    symbol_bits[i] = 0;
  }

  // Quantize_si.write(8);
  // wait();

  int previous_frequency = -1;
  for (int i = 0; i < num_symbols; i++) {
    sorted_copy1[i].value = sorted[i].value;
    sorted_copy1[i].frequency = sorted[i].frequency;
    sorted_copy2[i].value = sorted[i].value;
    sorted_copy2[i].frequency = sorted[i].frequency;
    DPROF(assert(previous_frequency <= (int)sorted[i].frequency));
    previous_frequency = sorted[i].frequency;
  }

  // Quantize_si.write(9);
  // wait();

  create_tree(sorted_copy1, num_symbols, parent, left, right);

  // Quantize_si.write(10);
  // wait();
  compute_bit_length(parent, left, right, num_symbols, length_histogram);

  // Quantize_si.write(11);
  // wait();

#ifndef __SYNTHESIS__
  // Check the result of computing the tree histogram
  int codewords_in_tree = 0;
  for (int i = 0; i < TREE_DEPTH; i++) {
#pragma HLS PIPELINE II = 1
    // if (length_histogram[i] > 0)
    //   ALOG("Value: " << length_histogram[i] << " codewords with length " << i
    //                  << "\n");
    codewords_in_tree += length_histogram[i];
  }
  // prevent infinite loop
  DPROF(assert(codewords_in_tree == num_symbols));
#endif

  // Quantize_si.write(12);
  // wait();
  truncate_tree(length_histogram, truncated_length_histogram1,
                truncated_length_histogram2);

  // Quantize_si.write(13);
  // wait();
  canonize_tree(sorted_copy2, num_symbols, truncated_length_histogram1,
                symbol_bits);
  // Quantize_si.write(14);
  // wait();
  create_codeword(symbol_bits, truncated_length_histogram2);

  // Quantize_si.write(15);
  // wait();

  encode(channel_size);

  // Quantize_si.write(16);
  // wait();
}
