#include <Arduino.h>

#define M 4  // Number of quantization levels (2^numBitsPerSample)

void mAryQuantization(int sampleArray[], int numBitsPerSample, int bitString[], int length) {
  int minVal = sampleArray[0];
  int maxVal = sampleArray[0];

  // Find min and max values
  for (int i = 1; i < length; i++) {
    if (sampleArray[i] < minVal) {
      minVal = sampleArray[i];
    }
    if (sampleArray[i] > maxVal) {
      maxVal = sampleArray[i];
    }
  }

  // Quantization
  for (int i = 0; i < length; i++) {
    if (sampleArray[i] == minVal) {
      bitString[i] = 1;
    } else if (sampleArray[i] == maxVal) {
      bitString[i] = M - 1;
    } else {
      for (int j = 0; j < M; j++) {
        if (minVal + j * (maxVal - minVal) / (M - 1) <= sampleArray[i] &&
            sampleArray[i] <= minVal + (j + 1) * (maxVal - minVal) / (M - 1)) {
          bitString[i] = j;
          break;
        }
      }
    }
  }
}

void myHomotopy(int A[][6], int y[], int x[], int m) {
  int iter_times = 2000;
  int act_set[m];

  for (int iter_idx = 1; iter_idx <= iter_times; iter_idx++) {
    // Compute residual correlations
    int c[m];
    for (int i = 0; i < m; i++) {
      c[i] = 0;
      for (int j = 0; j < 6; j++) {
        c[i] += A[j][i] * (y[j] - (A[j][0] * x[0] + A[j][1] * x[1] + A[j][2] * x[2] +
                                   A[j][3] * x[3] + A[j][4] * x[4] + A[j][5] * x[5]));
      }
    }

    // Compute active set
    int lambda_max_idx = 0;
    int lambda_max = abs(c[0]);
    for (int i = 1; i < m; i++) {
      if (abs(c[i]) > lambda_max) {
        lambda_max = abs(c[i]);
        lambda_max_idx = i;
      }
    }

    int count = 0;
    for (int i = 0; i < m; i++) {
      if (abs(abs(c[i]) - lambda_max) < 1e-5) {
        act_set[count] = i;
        count++;
      }
    }

    int state[m];
    for (int i = 0; i < m; i++) {
      state[i] = 0;
    }
    for (int i = 0; i < count; i++) {
      state[act_set[i]] = 1;
    }

    // Compute direction
    int R[count][count];
    int d[count];
    for (int i = 0; i < count; i++) {
      for (int j = 0; j < count; j++) {
        R[i][j] = 0;
        for (int k = 0; k < 6; k++) {
          R[i][j] += A[k][act_set[i]] * A[k][act_set[j]];
        }
      }
    }

    for (int i = 0; i < count; i++) {
      d[i] = (c[act_set[i]] >= 0) ? 1 : -1;
    }

    // Compute step
    int gamma = 1000;
    for (int i = 0; i < m - 1; i++) {
      if (state[i]) {
        int my_id = 0;
        for (int j = 0; j < count; j++) {
          if (act_set[j] == i) {
            my_id = j;
            break;
          }
        }
        int tmp = max(0, -x[i] / d[my_id]);
        if (tmp > 0) {
          gamma = min(tmp, gamma);
        }
      } else {
        int my_id = 0;
        int av = 0;
        for (int j = 0; j < 6; j++) {
          av += A[j][i] * d[my_id];
        }

        int tmp1 = max(0, (lambda_max - c[i]) / (1 - av));
        int tmp2 = max(0, (lambda_max + c[i]) / (1 + av));
        int tmp = min(tmp1, tmp2);
        if (tmp > 0) {
          gamma = min(tmp, gamma);
        }
      }
    }

    for (int i = 0; i < count; i++) {
      x[act_set[i]] = x[act_set[i]] + gamma * d[i];
    }

    int norm = 0;
    for (int i = 0; i < 6; i++) {
      norm += (y[i] - (A[i][0] * x[0] + A[i][1] * x[1] + A[i][2] * x[2] +
                       A[i][3] * x[3] + A[i][4] * x[4] + A[i][5] * x[5])) *
              (y[i] - (A[i][0] * x[0] + A[i][1] * x[1] + A[i][2] * x[2] +
                       A[i][3] * x[3] + A[i][4] * x[4] + A[i][5] * x[5]));
    }

    if (norm < 1e-6) {
      break;
    }
  }
}

void setup() {
  Serial.begin(9600);

 
}

void genkey(){
  int alice_rssi_values[] = {-39, -33, /*...*/};
  int bob_rssi_values[] = {-25, -26, /*...*/};

  int startIndex = 0;
  int endIndex = 64;

  int rssi_alice[endIndex - startIndex];
  int rssi_bob[endIndex - startIndex];

  for (int i = 0; i < endIndex - startIndex; i++) {
    rssi_alice[i] = alice_rssi_values[startIndex + i];
    rssi_bob[i] = bob_rssi_values[startIndex + i];
  }

  // Define filter parameters
  int order = 5;
  int framelen = 11;

  int Secret_key1[(endIndex - startIndex) * 2];
  int Secret_key2[(endIndex - startIndex) * 2];

  mAryQuantization(rssi_alice, 2, Secret_key1, endIndex - startIndex);
  mAryQuantization(rssi_bob, 2, Secret_key2, endIndex - startIndex);

  int bits_1[(endIndex - startIndex) * 2];
  int bits_2[(endIndex - startIndex) * 2];

  int len_bits = min(sizeof(bits_1) / sizeof(bits_1[0]), sizeof(bits_2) / sizeof(bits_2[0]));
  int bits_a[len_bits];
  int bits_b[len_bits];

  for (int i = 0; i < len_bits; i++) {
    bits_1[i] = Secret_key1[i];
    bits_2[i] = Secret_key2[i];
  }

  int A[6][6];
  int y1[6];
  int y2[6];
  int y[6];

  // Initialize A, y1, and y2 with appropriate values
  // ...

  for (int i = 0; i < 6; i++) {
    y[i] = y1[i] - y2[i];
  }

  int mismatch[len_bits];
  
  myHomotopy(A, y, mismatch, len_bits);

  int bits_recover[len_bits];
  for (int i = 0; i < len_bits; i++) {
    bits_recover[i] = bits_a[i] ^ mismatch[i];
  }

  Serial.println("Before error correction");
  // Print Secret_key1 and Secret_key2
  // ...

  Serial.println("After perform error correction for Alice");
  // Print bits_recover
  // ...
}
int counter = 0;
void loop() {
  // Do nothing in the loop
  Serial.println("Keygen");
  Serial.println(counter);
  genkey();
  counter++;

  delay(5000);
}