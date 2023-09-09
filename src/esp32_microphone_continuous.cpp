/* Edge Impulse Arduino examples
 * Copyright (c) 2022 EdgeImpulse Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// If your target is limited in memory remove this macro to save 10K RAM
#define EIDSP_QUANTIZE_FILTERBANK   0

/*
 ** NOTE: If you run into TFLite arena allocation issue.
 **
 ** This may be due to may dynamic memory fragmentation.
 ** Try defining "-DEI_CLASSIFIER_ALLOCATION_STATIC" in boards.local.txt (create
 ** if it doesn't exist) and copy this file to
 ** `<ARDUINO_CORE_INSTALL_PATH>/arduino/hardware/<mbed_core>/<core_version>/`.
 **
 ** See
 ** (https://support.arduino.cc/hc/en-us/articles/360012076960-Where-are-the-installed-cores-located-)
 ** to find where Arduino installs cores on your machine.
 **
 ** If the problem persists then there's not enough memory for this model and application.
 */

/* Includes ---------------------------------------------------------------- */
#include <voiceAssistant_inferencing.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "timer_u32.h"

#include "driver/i2s.h"

/** Audio buffers, pointers and selectors */
typedef struct {
    int8_t *buffers[2];
    unsigned char buf_select;
    unsigned char buf_ready;
    unsigned int buf_count;
    unsigned int n_samples;
} inference_t;

static inference_t inference;
static const uint32_t sample_buffer_size = 1024;
static signed short sampleBuffer[sample_buffer_size];
uint32_t bufferPointer = 0;
static bool debug_nn = false ; // Set this to true to see e.g. features generated from the raw signal
static int print_results = -(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW);
static bool record_status = true;

#define AUDIO_BUFFER_MAX 16000
bool transmitNow = false;
// hw_timer_t * timer = NULL;
// portMUX_TYPE timerMux = portMUX_INITIALIZER_UNLOCKED; 
// uint8_t audioBuffer[AUDIO_BUFFER_MAX];
// uint8_t transmitBuffer[AUDIO_BUFFER_MAX];


static void audio_inference_callback(uint32_t n_bytes);
static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr);
static bool microphone_inference_record(void);
static bool microphone_inference_start(uint32_t n_samples);
/**
 * @brief      Arduino setup function
 */
void setup()
{
    // put your setup code here, to run once:
    Serial.begin(115200);
    // comment out the below line to cancel the wait for USB connection (needed for native USB)
    while (!Serial);
    Serial.println("Edge Impulse Inferencing Demo");
    adc1_config_width(ADC_WIDTH_12Bit); // configure the analogue to digital converter
    adc1_config_channel_atten(ADC1_CHANNEL_0, ADC_ATTEN_0db); // connects the ADC 1 with channel 0 (GPIO 36)
    // summary of inferencing settings (from model_metadata.h)
    ei_printf("Inferencing settings:\n");
    ei_printf("\tInterval: ");
    ei_printf_float((float)EI_CLASSIFIER_INTERVAL_MS);
    ei_printf(" ms.\n");
    ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    ei_printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
    ei_printf("\tNo. of classes: %d\n", sizeof(ei_classifier_inferencing_categories) / sizeof(ei_classifier_inferencing_categories[0]));

    run_classifier_init();
    ei_printf("\nStarting continious inference in 2 seconds...\n");
    ei_sleep(2000);

    if (microphone_inference_start(EI_CLASSIFIER_SLICE_SIZE) == false) {
        ei_printf("ERR: Could not allocate audio buffer (size %d), this could be due to the window length of your model\r\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT);
        return;
    }

    ei_printf("Recording...\n");
}

/**
 * @brief      Arduino main function. Runs the inferencing loop.
 */
void loop()
{
    // uint64_t t0, dt;
    // t0 = timer_u32();
    bool m = microphone_inference_record();
    if (!m) {
        ei_printf("ERR: Failed to record audio...\n");
        return;
    }
    // dt = timer_u32() - t0;
    // ei_printf("%f\n", timer_delta_s(dt));
    // Serial.write((uint8_t*)inference.buffers[inference.buf_select ^ 1], inference.n_samples);

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_SLICE_SIZE;
    signal.get_data = &microphone_audio_signal_get_data;
    ei_impulse_result_t result = {0};

    EI_IMPULSE_ERROR r = run_classifier_continuous(&signal, &result, debug_nn);
    if (r != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", r);
        return;
    }
    if (++print_results >= (EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)) {
        // print the predictions
        // ei_printf("Predictions ");
        // ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)",
            // result.timing.dsp, result.timing.classification, result.timing.anomaly);
        // ei_printf(": \n");
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            // ei_printf("    %s: ", result.classification[ix].label);
            // ei_printf_float(result.classification[ix].value);
            // ei_printf("\n");
            if (result.classification[ix].label == "keyWord") {
                if (result.classification[ix].value > 0.9) {
                    ei_printf("\nkeyWord with %f\n", result.classification[ix].value);
                } else {
                    ei_printf(".");
                }
            }
        }
#if EI_CLASSIFIER_HAS_ANOMALY == 1
        ei_printf("    anomaly score: ");
        ei_printf_float(result.anomaly);
        ei_printf("\n");
#endif

        print_results = 0;
    }
}

static void audio_inference_callback(uint32_t n_bytes)
{
    // signed short min = 32767;
    // signed short max = -32768;
    // for (int i = 0; i < n_bytes; i++) {
    //     if (sampleBuffer[i] < min) {
    //         min = sampleBuffer[i];
    //     }
    //     if (sampleBuffer[i] > max) {
    //         max = sampleBuffer[i];
    //     }
    // }
    // ei_printf("%d %d\n", min, max);
    for(int i = 0; i < n_bytes; i++) {
        inference.buffers[inference.buf_select][inference.buf_count++] = (uint8_t) sampleBuffer[i]; //map(sampleBuffer[i], min , max, 0, 255);//map(sampleBuffer[i], min, max, -98, 97);
        if(inference.buf_count >= inference.n_samples) {
            // ei_printf("%d\n", sampleBuffer[i]);
            inference.buf_select ^= 1;
            inference.buf_count = 0;
            inference.buf_ready = 1;
        }
    }
}

// void IRAM_ATTR onTimer() {
//   portENTER_CRITICAL_ISR(&timerMux); // says that we want to run critical code and don't want to be interrupted
//   int adcVal = adc1_get_raw(ADC1_CHANNEL_6); // reads the ADC
//   uint8_t value = map(adcVal, 0 , 4096, 0, 255);  // converts the value to 0..255 (8bit)
//   audioBuffer[bufferPointer] = value; // stores the value
//   bufferPointer++;
 
//   if (bufferPointer == AUDIO_BUFFER_MAX) { // when the buffer is full
//     bufferPointer = 0;
//     memcpy(transmitBuffer, audioBuffer, AUDIO_BUFFER_MAX); // copy buffer into a second buffer
//     transmitNow = true; // sets the value true so we know that we can transmit now
//   }
//   portEXIT_CRITICAL_ISR(&timerMux); // says that we have run our critical code
// }
int adcVal;
signed short value;


void IRAM_ATTR onTimer() {
    int adcVal = adc1_get_raw(ADC1_CHANNEL_6); // reads the ADC #TODO
    int8_t value = map(adcVal, 0 , 4096, -128, 127);  // converts the value to 0..255 (8bit)
    inference.buffers[inference.buf_select][inference.buf_count++] = value;
    if(inference.buf_count >= inference.n_samples) {
        // ei_printf("%d\n", sampleBuffer[i]);
        inference.buf_select ^= 1;
        inference.buf_count = 0;
        inference.buf_ready = 1;
    }
//   audioBuffer[bufferPointer] = value; // stores the value
//   bufferPointer++;
 
}
hw_timer_t *timer = NULL;
static void capture_samples(void* arg) { 
    // portENTER_CRITICAL_ISR(&timerMux); 
    const int32_t i2s_bytes_to_read = (uint32_t)arg;
    size_t bytes_read = i2s_bytes_to_read;

    timer = timerBegin(0, 80, true); // 80 Prescaler
    timerAttachInterrupt(timer, &onTimer, true); // binds the handling function to our timer 
    timerAlarmWrite(timer, 125, true);
    timerAlarmEnable(timer);
    // while (record_status) {
    //    delay(100);
    // }
    vTaskDelete(NULL);

}


static void capture_samples1(void* arg) {
    
    // timer = timerBegin(0, 80, true); // 80 Prescaler
    // timerAttachInterrupt(timer, &onTimer, true); // binds the handling function to our timer 
    // timerAlarmWrite(timer, 125, true);
    // timerAlarmEnable(timer);

    const int32_t i2s_bytes_to_read = (uint32_t)arg;
    size_t bytes_read = i2s_bytes_to_read;
    
    uint64_t last_read = 0, current_read;
    uint32_t beatTime = (uint32_t) (1000000/8000);
    // unsigned long time_diff;

    while (record_status) {
        // if (transmitNow) {
        //     transmitNow = false;

        //     audio_inference_callback(AUDIO_BUFFER_MAX);
        // }
        last_read = timer_u32();
        for (int i = 0; i < i2s_bytes_to_read; i++) {
            adcVal = adc1_get_raw(ADC1_CHANNEL_6);
            // value = (int8_t) map(adcVal, 0 , 4096, -128, 127);
            value = (signed short) map(adcVal, 0 , 4096, 0, 255);
            // ei_printf("%d ", value);
            sampleBuffer[i] = value;
            // sampleBuffer[i] = ;
            // current_read = timer_u32();
            // ei_printf("%d %d\n", current_read, last_read);
            // time_diff = micros() - last_read;
            // while (timer_delta_us(timer_u32() - last_read) < beatTime);
            // if (time_diff < beatTime) {
            // // //     // ei_printf("%d\n", (beatTime - time_diff));
            // // // //     // vTaskDelay((beatTime - time_diff)/1000/portTICK_PERIOD_MS);

            //     delayMicroseconds((uint32_t)(beatTime - micros() + last_read));
            // }
            last_read = timer_u32();
        }
        if (record_status) {
            // break;
            audio_inference_callback(i2s_bytes_to_read);
        }
        else {
            break;
        }
    }
    vTaskDelete(NULL);
}

/**
 * @brief      Init inferencing struct and setup/start PDM
 *
 * @param[in]  n_samples  The n samples
 *
 * @return     { description_of_the_return_value }
 */
static bool microphone_inference_start(uint32_t n_samples)
{
    inference.buffers[0] = (int8_t*)malloc(n_samples * sizeof(int8_t));

    if (inference.buffers[0] == NULL) {
        return false;
    }

    inference.buffers[1] = (int8_t *)malloc(n_samples * sizeof(int8_t));

    if (inference.buffers[1] == NULL) {
        ei_free(inference.buffers[0]);
        return false;
    }

    inference.buf_select = 0;
    inference.buf_count = 0;
    inference.n_samples = n_samples;
    inference.buf_ready = 0;

    // if (i2s_init(EI_CLASSIFIER_FREQUENCY)) {
    //     ei_printf("Failed to start I2S!");
    // }

    ei_sleep(100);

    record_status = true;
    // const int32_t i2s_bytes_to_read = (uint32_t)arg;
    // size_t bytes_read = i2s_bytes_to_read;

    // timer = timerBegin(0, 80, true); // 80 Prescaler
    // timerAttachInterrupt(timer, &onTimer, true); // binds the handling function to our timer 
    // timerAlarmWrite(timer, 125, true);
    // timerAlarmEnable(timer);
    xTaskCreate(capture_samples, "CaptureSamples", 1024*8, (void*)sample_buffer_size, 0, NULL);

    return true;
}

/**
 * @brief      Wait on new data
 *
 * @return     True when finished
 */
static bool microphone_inference_record(void)
{
    bool ret = true;

    if (inference.buf_ready == 1) {
        ei_printf(
            "Error sample buffer overrun. Decrease the number of slices per model window "
            "(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)\n");
        ret = false;
    }

    while (inference.buf_ready == 0) {
        delay(1);
    }

    inference.buf_ready = 0;
    return true;
}

/**
 * Get raw audio signal data
 */
static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr)
{
    //  for (int ix = 0; ix < length; ix++) {
    //         ei_printf("%d\n", inference.buffers[inference.buf_select ^ 1][ix]<<8);
    //     }
    numpy::int8_to_float(&inference.buffers[inference.buf_select ^ 1][offset], out_ptr, length);

    return 0;
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif