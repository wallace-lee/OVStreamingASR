# OVStreamingASR
This is a python app demonstrating On-Device Streaming ASR using QuartzNet model executed with Intel(r) OpenVINO

# How It Works
The app will continuously stream audio coming through microphone, break it into chunks and some transformation before sending it to the neural network (QuartzNet-15x5) to get character probabilites. Subsequently, perform CTC decoding and optionally apply KenLM to improve WER.

# Pre-requisite
Intel(r) Distribution of OpenVINO 2022.1

# Preparing to Run
The list of models supported by the demo is in `models.lst` file. This file can be used as a parameter for Model Downloader and Converter to download and, if necessary, convert models to OpenVINO IR format (*.xml + *.bin).

An example of using the Model Downloader:

```sh
omz_downloader --list models.lst
```

An example of using the Model Converter:

```sh
omz_converter --list models.lst
```
### Supported Models

* quartznet-15x5-en

> **NOTE**: Refer to the tables [Intel's Pre-Trained Models Device Support](../../../models/intel/device_support.md) and [Public Pre-Trained Models Device Support](../../../models/public/device_support.md) for the details on models inference support at different devices.

## Running Demo

Run the application with `-h` option to see help message.

```
usage: streaming_asr_openvino.py [-h] -m MODEL [-d DEVICE]

optional arguments:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on, for
                        example: CPU, GPU, HDDL, MYRIAD or HETERO. The
                        demo will look for a suitable OpenVINO Runtime plugin for this
                        device. Default value is CPU.
```
The typical command line is:

```sh
python3 streaming_asr_openvino.py -m quartznet-15x5-en.xml
```


## Demo Output

The application prints the decoded text for the audio coming through the mic in real-time until the program is terminated.


## See Also

* [Open Model Zoo Demos](../../README.md)
* [Model Optimizer](https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/model_tools/README.md)
