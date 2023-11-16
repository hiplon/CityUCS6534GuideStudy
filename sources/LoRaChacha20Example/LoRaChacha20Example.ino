/*
  LoRa Simple Gateway/Node Exemple
*/

#include <SPI.h>              // include libraries
#include <LoRa.h>
#include <ChaCha.h>

const long frequency = 868E6;  // LoRa Frequency

const int csPin = 10;          // LoRa radio chip select
const int resetPin = 9;        // LoRa radio reset
const int irqPin = 2;          // change for your board; must be a hardware interrupt pin

static const uint8_t key[32] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10};
static const uint8_t iv[8] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01};


void setup() {
  Serial.begin(9600);                   // initialize serial
  while (!Serial);

  LoRa.setPins(csPin, resetPin, irqPin);

  if (!LoRa.begin(frequency)) {
    Serial.println("LoRa init failed. Check your connections.");
    while (true);                       // if failed, do nothing
  }

  Serial.println("LoRa init succeeded.");
  Serial.println();
  Serial.println("LoRa Simple Node");
  Serial.println("Only receive messages from gateways");
  Serial.println("Tx: invertIQ disable");
  Serial.println("Rx: invertIQ enable");
  Serial.println();

  LoRa.onReceive(onReceive);
  LoRa.onTxDone(onTxDone);
  LoRa_rxMode();
}

void loop() {
  if (runEvery(4000)) { // repeat every 1000 millis
    ChaCha chacha1(20);
    chacha1.setKey(key,sizeof(key));
    chacha1.setIV(iv,sizeof(iv));
    
    //uint8_t ciphertext[32];
    //chacha1.encrypt()
    String encrypt_message = "";
    //const byte Plaintext[] = {0x01, 0x23, 0x45};
    const byte Plaintext[] = "PM2.5: 30|PM10: 35|SO2: 8|NO2: 25";
    //String message = "Hello";
    //Serial.println(message);
   // uint8_t messageArray[message.length()];
    //message.toCharArray(messageArray, message.length());
    int PlaintextSize = (sizeof Plaintext);
    int cryptSize = PlaintextSize;
    
    byte outputtext[PlaintextSize];

    memset(outputtext, 0, PlaintextSize);

    byte cryptmsg[cryptSize];
    //String input = (char*)messageArray;
    String input = String((char*)Plaintext);
    
    Serial.println("Before Encrypt");
    //Serial.println(sizeof(input));
    Serial.println(input);
    Serial.println("Start Encrypt");
    chacha1.encrypt(cryptmsg,Plaintext,PlaintextSize);
    
    Serial.println("Encrypt Success");
    String cryptmsgstr = String((char*)cryptmsg);


    ChaCha chacha2(20);
    chacha2.setKey(key,sizeof(key));
    chacha2.setIV(iv,sizeof(iv));

    Serial.println("After Encrypt");
    Serial.println(cryptmsgstr);
    //Serial.println(sizeof(cryptmsg));
    chacha2.decrypt(outputtext,cryptmsg,cryptSize);
    Serial.println("Decrypt Success");
    //message += "I'm a Node! ";
    //message += millis();
    
    //LoRa_sendMessage(message); // send a message
    //String output = String((char*)outputtext);
    String output = (char*)outputtext;
    Serial.println("After Decrypt");
    //Serial.println(sizeof(outputtext));
    //Serial.println(sizeof(output));
    Serial.println(output);
  }
}

void LoRa_rxMode(){
  LoRa.enableInvertIQ();                // active invert I and Q signals
  LoRa.receive();                       // set receive mode
}

void LoRa_txMode(){
  LoRa.idle();                          // set standby mode
  LoRa.disableInvertIQ();               // normal mode
}

void LoRa_sendMessage(String message) {
  LoRa_txMode();                        // set tx mode
  LoRa.beginPacket();                   // start packet
  LoRa.print(message);                  // add payload
  LoRa.endPacket(true);                 // finish packet and send it
}

void onReceive(int packetSize) {
  String message = "";

  while (LoRa.available()) {
    message += (char)LoRa.read();
  }

  Serial.print("Node Receive: ");
  Serial.println(message);
}

void onTxDone() {
  Serial.println("TxDone");
  LoRa_rxMode();
}

boolean runEvery(unsigned long interval)
{
  static unsigned long previousMillis = 0;
  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis >= interval)
  {
    previousMillis = currentMillis;
    return true;
  }
  return false;
}

