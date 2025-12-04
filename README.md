# 🍔 Burger vs Pizza Classifier - MAX78000 🍕

Real-time food classification using the MAX78000 FTHR (Feather) board with integrated CNN accelerator and camera support. This repository contains two implementations of the burger vs pizza classifier.

## 📁 Project Structure

This repository contains two variants of the classifier:

### 1. **cats-dogs/** 
The main implementation with full camera integration and real-time inference capabilities.

### 2. **burger-pizza/**
Alternative implementation with different configurations and features.

## ✨ Key Features

- **Real-time Camera Capture**: 128x128 RGB image capture from camera module
- **Hardware CNN Acceleration**: Ultra-low power inference using MAX78000's dedicated CNN accelerator
- **Button-triggered Capture**: Press SW1/PB1 to capture and classify images
- **LED Feedback System**: Visual indication of classification results
  - LED1 (Green): Burger detected
  - LED2 (Red): Pizza detected
  - Both LEDs: Tie/Unknown result
- **ASCII Art Visualization**: Optional terminal-based image preview
- **Known-Answer Test**: Built-in verification test on startup
- **DMA-based Streaming**: Efficient image transfer from camera

## 🔧 Hardware Requirements

- **Board**: MAX78000 Feather (FTHR_REVA)
- **Camera**: Compatible RGB888 camera module (e.g., OV7692)
- **Connection**: USB for programming and serial output
- **Input**: Push button SW1/PB1 for capture trigger
- **Output**: LED1 and LED2 for visual feedback

## 📊 Performance Metrics

- **Inference Time**: ~30-50ms per image
- **Total Operations**: 51.4 million ops (50.4M MACC, 937K comparisons)
- **Memory Usage**: 
  - Weight memory: 57,776 bytes (13.1% of available)
  - Bias memory: 2 bytes (0.1% of available)
- **Image Resolution**: 128x128x3 (RGB)
- **Power Efficiency**: Ultra-low power operation

## 🏗️ CNN Architecture

7-layer convolutional neural network optimized for MAX78000:

| Layer | Total Ops | MACC Ops | Comparisons |
|-------|-----------|----------|-------------|
| 0 | 7,340,032 | 7,077,888 | 262,144 |
| 1 | 19,267,584 | 18,874,368 | 393,216 |
| 2 | 19,070,976 | 18,874,368 | 196,608 |
| 3 | 4,792,320 | 4,718,592 | 73,728 |
| 4 | 600,064 | 589,824 | 10,240 |
| 5 | 295,936 | 294,912 | 1,024 |
| 6 | 2,048 | 2,048 | 0 |

## 🚀 Quick Start

### Prerequisites

1. **Maxim Micros SDK**: Install the latest MaximSDK
2. **Toolchain**: ARM GCC toolchain (included with SDK)
3. **Serial Terminal**: For viewing output (115200 baud, 8N1)

### Build Instructions

```bash
# Navigate to your chosen implementation
cd cats-dogs
# or
cd burger-pizza

# Build the project
make

# Flash to board
make flash
```

### Running the Application

1. Connect MAX78000 Feather board via USB
2. Open serial terminal at 115200 baud
3. Reset or power cycle the board
4. Wait for "Known-Answer Test PASSED" message
5. Press SW1/PB1 to capture an image
6. View classification results and LED feedback

## 📖 Usage Example

```
*** Classification Results ***
[   12345] -> Class 0 (    Burger): 87.5%
[   -5432] -> Class 1 (     Pizza): 12.5%

Result: Burger detected with 87.5% confidence
```

## 🛠️ Technical Details

### Image Processing Pipeline

1. **Capture**: DMA streams 128x128 RGB888 image from camera
2. **Preprocessing**: Convert pixels from [0,255] to [-128,127] range
3. **Format**: Pack RGB as `0x00bbggrr` with range adjustment
4. **Input**: Load 16,384 uint32_t values into CNN FIFO
5. **Inference**: Hardware CNN accelerator processes image
6. **Softmax**: Convert logits to probabilities
7. **Classification**: Determine highest confidence class

### Camera Configuration

- **Resolution**: 128x128 pixels
- **Format**: RGB888 (24-bit color)
- **Transfer**: DMA streaming mode
- **FIFO**: 3-byte mode
- **Clock**: 5 MHz (configurable)

## 🐛 Troubleshooting

### Camera Overflow Error
If you see `ERROR: Camera overflow detected`:
- Reduce camera clock speed (`CAMERA_FREQ`)
- Check camera register 0x11 prescaler setting
- Verify DMA channel configuration

### Known-Answer Test Failed
If the initial test fails:
- Verify CNN weights are correctly loaded
- Check for memory corruption
- Ensure proper CNN clock configuration (50 MHz)

### No Serial Output
- Verify baud rate: 115200, 8N1
- Check USB connection
- Try resetting the board

## 📂 Repository Contents

### cats-dogs/
- `main.c` - Main application with camera integration
- `cnn.c/h` - CNN driver implementation
- `softmax.c` - Softmax activation function
- `weights.h` - Pre-trained CNN weights
- `Makefile` - Build configuration
- Full README with detailed documentation

### burger-pizza/
- Similar structure with variant implementation
- Alternative configurations and features

## 📝 License

Copyright (C) 2019-2024 Maxim Integrated Products, Inc., All Rights Reserved.

## 🙏 Acknowledgments

- Generated using Maxim's `ai8xize.py` tool
- Enhanced with camera support and real-time inference
- Based on MAX78000 SDK examples

## 📧 Support

For questions or issues, please open an issue in this repository.

---

**Hardware**: MAX78000 Feather (FTHR_REVA)  
**SDK**: Maxim Micros SDK  
**Model**: Burger vs Pizza CNN Classifier  
**Status**: ✅ Production Ready
