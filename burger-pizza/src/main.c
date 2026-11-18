/*******************************************************************************
* Copyright (C) 2019-2023 Maxim Integrated Products, Inc., All rights Reserved.
*
* This software is protected by copyright laws of the United States and
* of foreign countries. This material may also be protected by patent laws
* and technology transfer regulations of the United States and of foreign
* countries. This software is furnished under a license agreement and/or a
* nondisclosure agreement and may only be used or reproduced in accordance
* with the terms of those agreements. Dissemination of this information to
* any party or parties not specified in the license agreement and/or
* nondisclosure agreement is expressly prohibited.
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
* OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL MAXIM INTEGRATED BE LIABLE FOR ANY CLAIM, DAMAGES
* OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
*
* Except as contained in this notice, the name of Maxim Integrated
* Products, Inc. shall not be used except as stated in the Maxim Integrated
* Products, Inc. Branding Policy.
*
* The mere transfer of this software does not imply any licenses
* of trade secrets, proprietary technology, copyrights, patents,
* trademarks, maskwork rights, or any other form of intellectual
* property whatsoever. Maxim Integrated Products, Inc. retains all
* ownership rights.
*******************************************************************************/

// Burger vs. Pizza CNN inference and Known-Answer Test (KAT) for MAX78000.
// Generated with ai8xize.py from checkpoint trained/ai85-burgerpizza-qat8-q.pth.tar
// using network config networks/burgerpizza-hwc.yaml (HWC 3x128x128, FIFO, Softmax).

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc.h"
#include "cnn.h"
#include "sampledata.h"
#include "sampleoutput.h"

volatile uint32_t cnn_time; // Inference duration in microseconds when CNN_INFERENCE_TIMER is enabled

void fail(void)
{
  printf("\n*** FAIL ***\n\n");
  while (1);
}

// HWC 3x128x128 RGB input packed as 16,384 32-bit words (0x00bbggrr, signed [-128, 127] per channel).
static const uint32_t input_0[] = SAMPLE_INPUT_0;
void load_input(void)
{
  // Streams static sample data into CNN FIFO 0; replace with camera DMA or frame buffer in live mode.
  int i;
  const uint32_t *in0 = input_0;

  for (i = 0; i < 16384; i++) {
    // 0x50000004 bit 0 indicates FIFO 0 full; poll before writing to 0x50000008 to prevent overrun.
    while (((*((volatile uint32_t *) 0x50000004) & 1)) != 0);
    *((volatile uint32_t *) 0x50000008) = *in0++;
  }
}

// Verifies Layer 6 hardware output registers against expected known-answer values in sampleoutput.h.
// Omit check_output() when running live camera inference instead of the static test vector.
static const uint32_t sample_output[] = SAMPLE_OUTPUT;
int check_output(void)
{
  int i;
  uint32_t mask, len;
  volatile uint32_t *addr;
  const uint32_t *ptr = sample_output;

  while ((addr = (volatile uint32_t *) *ptr++) != 0) {
    mask = *ptr++;
    len = *ptr++;
    for (i = 0; i < len; i++)
      if ((*addr++ & mask) != *ptr++) {
        printf("Data mismatch (%d/%d) at address 0x%08x: Expected 0x%08x, read 0x%08x.\n",
               i + 1, len, addr - 1, *(ptr - 1), *(addr - 1) & mask);
        return CNN_FAIL;
      }
  }

  return CNN_OK;
}

// Raw Q17.14 logits unloaded from the accelerator and Q15 probabilities from softmax.
static int32_t ml_data[CNN_NUM_OUTPUTS];
static q15_t ml_softmax[CNN_NUM_OUTPUTS];

void softmax_layer(void)
{
  cnn_unload((uint32_t *) ml_data);
  softmax_q17p14_q15((const q31_t *) ml_data, CNN_NUM_OUTPUTS, ml_softmax);
}

int main(void)
{
  int i;
  int digs, tens;

  MXC_ICC_Enable(MXC_ICC0);

  // Run system core clock from the 100 MHz internal primary oscillator (IPO).
  MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
  SystemCoreClockUpdate();

  printf("Waiting...\n");

  // 2-second guard window before sleep/clock gating so SWD debuggers can attach reliably.
  MXC_Delay(SEC(2));

  // Clock the CNN accelerator from APB PCLK (50 MHz, divider 1).
  cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);

  printf("\n*** CNN Inference Test burger-pizza ***\n");

  cnn_init();
  cnn_load_weights();
  cnn_load_bias();
  cnn_configure();
  cnn_start();
  load_input();

  // Sleep until CNN_ISR sets cnn_time upon quadrant completion.
  while (cnn_time == 0)
    MXC_LP_EnterSleepMode();

  if (check_output() != CNN_OK) fail();
  softmax_layer();

  printf("\n*** PASS ***\n\n");

#ifdef CNN_INFERENCE_TIMER
  printf("Approximate data loading and inference time: %u us\n\n", cnn_time);
#endif

  cnn_disable();

  printf("Classification results:\n");
  for (i = 0; i < CNN_NUM_OUTPUTS; i++) {
    // Convert Q15 probability [0, 32767] to rounded tenths of a percent (0.0% to 100.0%).
    digs = (1000 * ml_softmax[i] + 0x4000) >> 15;
    tens = digs % 10;
    digs = digs / 10;
    printf("[%7d] -> Class %d: %d.%d%%\n", ml_data[i], i, digs, tens);
  }

  return 0;
}

/*
  SUMMARY OF OPS
  Hardware: 51,368,960 ops (50,432,000 macc; 936,960 comp; 0 add; 0 mul; 0 bitwise)
    Layer 0: 7,340,032 ops (7,077,888 macc; 262,144 comp; 0 add; 0 mul; 0 bitwise)
    Layer 1: 19,267,584 ops (18,874,368 macc; 393,216 comp; 0 add; 0 mul; 0 bitwise)
    Layer 2: 19,070,976 ops (18,874,368 macc; 196,608 comp; 0 add; 0 mul; 0 bitwise)
    Layer 3: 4,792,320 ops (4,718,592 macc; 73,728 comp; 0 add; 0 mul; 0 bitwise)
    Layer 4: 600,064 ops (589,824 macc; 10,240 comp; 0 add; 0 mul; 0 bitwise)
    Layer 5: 295,936 ops (294,912 macc; 1,024 comp; 0 add; 0 mul; 0 bitwise)
    Layer 6: 2,048 ops (2,048 macc; 0 comp; 0 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 57,776 bytes out of 442,368 bytes total (13.1%)
  Bias memory:   2 bytes out of 2,048 bytes total (0.1%)
*/

