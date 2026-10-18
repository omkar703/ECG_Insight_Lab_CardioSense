/*
 * ============================================================================
 * CardioSense / ECG Insight Lab — Hardware Prototype Firmware
 * Target: AD8232 Heart Rate Monitor Breakout + Arduino Uno R3 (ATmega328P)
 * Version: 1.0.0
 * Protocol: Line-oriented CSV over UART (115200 baud)
 *
 * CAUTION — ELECTRICAL SAFETY NOTICE:
 * THIS IS PROTOTYPE RESEARCH FIRMWARE FOR BENCH-TOP TESTING WITH
 * SIGNAL GENERATORS OR ISOLATED HARDWARE SIMULATORS ONLY.
 * NEVER CONNECT ELECTRODES TO HUMAN SUBJECTS WITHOUT VERIFIED MEDICAL-GRADE
 * GALVANIC ISOLATION (IEC 60601-1 COMPLIANCE) AND EXPERT CLINICAL REVIEW.
 * ARDUINO USB CONNECTIONS ARE NOT GALVANICALLY ISOLATED FROM MAINS GROUND.
 * ============================================================================
 */

// ----------------------------------------------------------------------------
// Pin Mapping (AD8232 Breakout to Arduino Uno R3)
// ----------------------------------------------------------------------------
const int PIN_ECG_OUTPUT = A0;   // Analog input: AD8232 OUTPUT pin
const int PIN_LO_PLUS    = 10;   // Digital input: Lead-Off Detect + (LO+)
const int PIN_LO_MINUS   = 11;   // Digital input: Lead-Off Detect - (LO-)
// Note: AD8232 3.3V -> Arduino 3.3V; AD8232 GND -> Arduino GND.
// SDN (Shutdown) pin is left unconnected (pulled high internally on breakout).

// ----------------------------------------------------------------------------
// Sampling & Communication Parameters
// ----------------------------------------------------------------------------
const unsigned long BAUD_RATE = 115200UL;

// Target nominal sample rate: 360 Hz (compatibility with MIT-BIH benchmark)
// Nominal interval in microseconds: 1,000,000 / 360 = 2777.777... us -> 2778 us
const unsigned long SAMPLE_INTERVAL_MICROS = 2778UL;
const unsigned long NOMINAL_SAMPLING_RATE_HZ = 360UL;

// ----------------------------------------------------------------------------
// State Variables (No dynamic String memory allocation)
// ----------------------------------------------------------------------------
unsigned long lastSampleMicros = 0UL;
uint32_t sequenceNumber = 0;
uint16_t overrunCount = 0;

void setup() {
  // Configure pin modes
  pinMode(PIN_LO_PLUS, INPUT);
  pinMode(PIN_LO_MINUS, INPUT);
  pinMode(PIN_ECG_OUTPUT, INPUT);

  // Initialize serial communication
  Serial.begin(BAUD_RATE);
  while (!Serial && millis() < 3000) {
    // Wait for native USB if needed (e.g., Leonardo), timeout at 3s on Uno
  }

  // Output protocol identification & metadata header lines
  // Lines starting with '#' are comment/header lines safely skipped by host parser.
  Serial.println(F("# CardioSense AD8232 Arduino Uno v1.0"));
  Serial.print(F("# baud="));
  Serial.print(BAUD_RATE);
  Serial.print(F(", nominal_fs="));
  Serial.print(NOMINAL_SAMPLING_RATE_HZ);
  Serial.print(F(", interval_us="));
  Serial.println(SAMPLE_INTERVAL_MICROS);
  Serial.println(F("# fields=seq,micros,adc,lo_plus,lo_minus"));
  Serial.println(F("# NOTICE: RAW 10-BIT ADC COUNTS (0-1023). NOT CALIBRATED ECG MV. PROTOTYPE ONLY."));

  lastSampleMicros = micros();
}

void loop() {
  unsigned long currentMicros = micros();

  // Check micros-based schedule
  if ((currentMicros - lastSampleMicros) >= SAMPLE_INTERVAL_MICROS) {
    // Detect severe scheduling overrun (> 2x sample interval)
    if ((currentMicros - lastSampleMicros) > (2 * SAMPLE_INTERVAL_MICROS)) {
      overrunCount++;
    }

    lastSampleMicros += SAMPLE_INTERVAL_MICROS;
    // Catch up if micros() wrapped or drifted severely
    if ((currentMicros - lastSampleMicros) > (4 * SAMPLE_INTERVAL_MICROS)) {
      lastSampleMicros = currentMicros;
    }

    // Read sensors
    int rawAdc = analogRead(PIN_ECG_OUTPUT);       // 10-bit ADC: 0 to 1023
    int loPlus = digitalRead(PIN_LO_PLUS);         // 0 or 1
    int loMinus = digitalRead(PIN_LO_MINUS);       // 0 or 1

    // Bounds clamp validation
    if (rawAdc < 0) rawAdc = 0;
    if (rawAdc > 1023) rawAdc = 1023;

    // Line-oriented CSV transmission: seq,micros,adc,lo_plus,lo_minus
    // Using direct numeric prints to guarantee zero heap fragmentation
    Serial.print(sequenceNumber);
    Serial.print(',');
    Serial.print(currentMicros);
    Serial.print(',');
    Serial.print(rawAdc);
    Serial.print(',');
    Serial.print(loPlus);
    Serial.print(',');
    Serial.println(loMinus);

    sequenceNumber++;
  }
}
