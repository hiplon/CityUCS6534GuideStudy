/*
  LoRa Duplex communication

  Sends a message every half second, and polls continually
  for new incoming messages. Implements a one-byte addressing scheme,
  with 0xFF as the broadcast address.

  Uses readString() from Stream class to read payload. The Stream class'
  timeout may affect other functuons, like the radio's callback. For an

  created 28 April 2017
  by Tom Igoe
*/
#include <SPI.h>              // include libraries
#include <LoRa.h>
#include <ChaCha.h>

const int csPin = 7;          // LoRa radio chip select
const int resetPin = 6;       // LoRa radio reset
const int irqPin = 1;         // change for your board; must be a hardware interrupt pin
static const uint8_t key[32] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10};
static const uint8_t iv[8] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01};
template <typename T>
int sign(T val) {
    return (T(0) < val) - (val < T(0));
}
String outgoing;              // outgoing message

byte msgCount = 0;            // count of outgoing messages
int seqCount = 0;            // count of outgoing messages
byte localAddress = 0xAA;     // address of this device
byte destination = 0xBB;      // destination to send to
long lastSendTime = 0;        // last send time
int interval = 2000;          // interval between sends

const int iter_times = 2000;
const int numBitsPerSample = 2;
const double alpha = 1.0;
const int M = pow(2, numBitsPerSample);

// Function to calculate dot product of two arrays
double dotProduct(double a[][6], double b[][6], int size) {
    double result = 0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < 6; j++) {
            result += a[i][j] * b[i][j];
        }
    }
    return result;
}

void myHomotopy(double A[][6], double y[], double result[], int &iter_idx) {
    int n = sizeof(A) / sizeof(A[0]);
    int m = sizeof(A[0]) / sizeof(A[0][0]);

    double x[m];
    memset(x, 0, sizeof(x));

    int act_set[m];
    memset(act_set, 0, sizeof(act_set));

    for (int iter_idx = 1; iter_idx <= iter_times; iter_idx++) {
        // Compute residual correlations
        double c[m];
        for (int i = 0; i < m; i++) {
            c[i] = 0;
            for (int j = 0; j < n; j++) {
                c[i] += A[j][i] * (y[j] - dotProduct(A[j], x, m));
            }
        }

        // Compute active set
        int lambda_max_idx = 0;
        double lambda_max = abs(c[0]);
        for (int i = 1; i < m; i++) {
            if (abs(c[i]) > lambda_max) {
                lambda_max = abs(c[i]);
                lambda_max_idx = i;
            }
        }

        memset(act_set, 0, sizeof(act_set));
        for (int i = 0; i < m; i++) {
            if (abs(abs(c[i]) - lambda_max) < 1e-5) {
                act_set[i] = 1;
            }
        }

        double state[m];
        memset(state, 0, sizeof(state));
        for (int i = 0; i < m; i++) {
            state[i] = 0;
            if (act_set[i] == 1) {
                state[i] = 1;
            }
        }

        // Compute direction
        double R[m][m];
        memset(R, 0, sizeof(R));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                for (int k = 0; k < n; k++) {
                    R[i][j] += A[k][i] * A[k][j];
                }
            }
        }

        double d[m];
        memset(d, 0, sizeof(d));
        for (int i = 0; i < m; i++) {
            if (state[i] == 1) {
                int my_id = findIndex(act_set, i, m);
                d[i] = sign(c[act_set[my_id]]);
            } else {
                double av = dotProduct(A[i], dotProduct(A, act_set, m), m);
                double tmp1 = max(0, (lambda_max - c[i]) / (1 - av));
                double tmp2 = max(0, (lambda_max + c[i]) / (1 + av));
                d[i] = min(tmp1, tmp2);
            }
        }

        // Compute step
        double gamma = 1000;
        for (int i = 0; i < m - 1; i++) {
            if (state[i] == 1) {
                int my_id = findIndex(act_set, i, m);
                double tmp = max(0, -x[i] / d[my_id]);
                gamma = min(tmp, gamma);
            } else {
                double av = dotProduct(A[i], dotProduct(A, act_set, m), m);
                double tmp1 = max(0, (lambda_max - c[i]) / (1 - av));
                double tmp2 = max(0, (lambda_max + c[i]) / (1 + av));
                double tmp = min(tmp1, tmp2);
                if (tmp > 0) {
                    gamma = min(tmp, gamma);
                }
            }
        }

        // Update x
        for (int i = 0; i < m; i++) {
            if (state[i] == 1) {
                int my_id = findIndex(act_set, i, m);
                x[i] = x[i] + gamma * d[my_id];
            }
        }

        // Check for convergence
        double norm_y = 0;
        for (int i = 0; i < n; i++) {
            double dot_product_result = dotProduct(A[i], x, m);
            norm_y += pow(y[i] - dot_product_result, 2);
        }

        if (sqrt(norm_y) < 1e-6) {
            break;
        }
    }

    // Set the result and iteration index
    for (int i = 0; i < m; i++) {
        result[i] = x[i];
    }
}

// Function to find the index of a value in an array
int findIndex(int arr[], int value, int size) {
    for (int i = 0; i < size; i++) {
        if (arr[i] == value) {
            return i;
        }
    }
    return -1;  // Value not found
}


void mAryQuantization(double sampleArray[], int sampleArrayLength, double result[], int validIndices[]) {
    double minVal = sampleArray[0];
    double maxVal = sampleArray[0];

    // Find min and max values
    for (int i = 1; i < sampleArrayLength; i++) {
        minVal = min(minVal, sampleArray[i]);
        maxVal = max(maxVal, sampleArray[i]);
    }

    double sampleSize = maxVal - minVal;
    double ratioStep = (1 - alpha) / M;
    double ratioGband = alpha / (M - 1);

    double stepSize = ratioStep * sampleSize;
    double gbandSize = ratioGband * sampleSize;

    double offset = minVal;
    int jj = 0;

    for (int i = 0; i < M; i++) {
        double levelBase = offset;
        double levelTop = levelBase + stepSize;
        offset = offset + stepSize + gbandSize;

        for (int j = 0; j < sampleArrayLength; j++) {
            if (sampleArray[j] == minVal) {
                result[jj] = 1;  // Decimal assignment starts from 0
                validIndices[jj] = j;
                jj += 1;
                break;
            } else if (sampleArray[j] == maxVal) {
                result[jj] = M - 1;  // Decimal assignment starts from 0
                validIndices[jj] = j;
                jj += 1;
                break;
            } else if (levelBase <= sampleArray[j] && sampleArray[j] <= levelTop) {
                result[jj] = i;  // Decimal assignment starts from 0
                validIndices[jj] = j;
                jj += 1;
            }
        }
    }
}


void setup() {
  Serial.begin(9600);                   // initialize serial
  while (!Serial);

  Serial.println("LoRaKEY Duplex Sender");

  // override the default CS, reset, and IRQ pins (optional)
  //LoRa.setPins(csPin, resetPin, irqPin);// set CS, reset, IRQ pin

  if (!LoRa.begin(868E6)) {             // initialize ratio at 915 MHz
    Serial.println("LoRa init failed. Check your connections.");
    while (true);                       // if failed, do nothing
  }

  Serial.println("LoRa init succeeded.");
}

void loop() {
  if (millis() - lastSendTime > interval) {
    String message = "SYN " + String(seqCount);   // send a message
    seqCount++;

    double alice_rssi_values[32];
    double bob_rssi_values[32];

    // Extract "Alice's RSSI" and "Bob's RSSI"
    // (Note: You need to implement this part based on your data format)

    // Initialize other variables and perform calculations
    // (Note: Adapt the code based on your specific requirements and available Arduino libraries)

    // Output results
    Serial.println("Results:");
    lastSendTime = millis();            // timestamp the message
    interval = random(2000) + 1000;    // 2-3 seconds
  }
  // parse for a packet, and call onReceive with the result:
  onReceive(LoRa.parsePacket());
}

void sendMessage(String outgoing) {
  LoRa.beginPacket();                   // start packet
  LoRa.write(destination);              // add destination address
  LoRa.write(localAddress);             // add sender address
  LoRa.write(msgCount);                 // add message ID
  LoRa.write(outgoing.length());        // add payload length
  Serial.println(outgoing.length());
  LoRa.print(outgoing);                 // add payload
  LoRa.endPacket();                     // finish packet and send it
  msgCount++;                           // increment message ID
}

void onReceive(int packetSize) {
  if (packetSize == 0) return;          // if there's no packet, return

  // read packet header bytes:
  int recipient = LoRa.read();          // recipient address
  byte sender = LoRa.read();            // sender address
  byte incomingMsgId = LoRa.read();     // incoming msg ID
  byte incomingLength = LoRa.read();    // incoming msg length

  String incoming = "";

  while (LoRa.available()) {
    incoming += (char)LoRa.read();
  }

  if (incomingLength != incoming.length()) {   // check length for error
    Serial.println("error: message length does not match length");
    return;                             // skip rest of function
  }

  // if the recipient isn't this device or broadcast,
  if (recipient != localAddress && recipient != 0xFF) {
    Serial.println("This message is not for me.");
    return;                             // skip rest of function
  }

  // if message is for this device, or broadcast, print details:
  Serial.println("Received from: 0x" + String(sender, HEX));
  Serial.println("Sent to: 0x" + String(recipient, HEX));
  Serial.println("Message ID: " + String(incomingMsgId));
  Serial.println("Message length: " + String(incomingLength));
  Serial.println("Alice's RSSI: " + incoming);
  Serial.println("Bob's RSSI: " + String(LoRa.packetRssi()));
  Serial.println("Snr: " + String(LoRa.packetSnr()));
  Serial.println();


}

