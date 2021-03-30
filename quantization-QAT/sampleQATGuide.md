# Sample QAT Guide

This note provides a recipe to run QAT using ResNet-50 example (TF->ONNX-TensorRT).

## Table Of Contents

* [Setup](#setup)
    * [Requirements](#requirements)
* [Step by Step Guide](#step-by-step-guide)
* [Reference](#reference)

## Setup

- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- [TensorFlow 20.06-tf1-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)
- GPU-based architecture:
  - [NVIDIA Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
  - [NVIDIA Turing](https://www.nvidia.com/en-us/geforce/turing/)
  - [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)
- TensorRT-7.1 NGC Container (`nvcr.io/nvidia/tensorrt:20.08-py3`)
- ONNX-Graphsurgeon 0.2.1 in TensorRT-7.1 source code repo



1. Download docker images and source code

```
docker pull nvcr.io/nvidia/tensorflow:20.01-tf1-py3
docker pull nvcr.io/nvidia/tensorrt:20.08-py3
```

```
mkdir -p workspace && cd workspace 
git clone https://github.com/NVIDIA/DeepLearningExamples
mv DeepLearningExamples/TensorFlow/Classification/ConvNets ./ConvNets
git clone https://github.com/NVIDIA/sampleQAT.git

```

2. Download and preprocess the ImageNet dataset.

e.g. ImageNet 1k

* [Download the images](http://image-net.org/download-images)
* Extract the training and validation data:


* Preprocess dataset to TFRecord form using [script](https://github.com/tensorflow/models/blob/archive/research/inception/inception/data/build_imagenet_data.py). Additional metadata from [autors repository](https://github.com/tensorflow/models/tree/archive/research/inception/inception/data) might be required.




## Step by Step Guide


### Step 1: Quantization Aware Training


start docker container, check dataset folder:

```bash
docker run -it --gpus all --network=host -v /raid:/raid nvcr.io/nvidia/tensorflow:20.01-tf1-py3 -v /path/to/dataset:/data /bin/bash
python3 -m pip install -r sampleQAT/requirements.txt
```

```
-rwxr-xr-x 1 1001 1001 145118773 Mar 26 07:11 train-01000-of-01024*
...
-rwxr-xr-x 1 1001 1001 144624004 Mar 26 07:11 train-01023-of-01024*
-rwxr-xr-x 1 1001 1001  51864495 Mar 26 07:12 validation-00100-of-00128*
...
-rwxr-xr-x 1 1001 1001  53116738 Mar 26 07:12 validation-00107-of-00128*
...
```


1. Download pretrain weights or run training from scratch 
    
    ```bash
    export DATA_DIR=/path/to/your/dataset 
    CUDA_VISIBLE_DEVICES=6 python3 main.py --arch=resnet50 --mode=train_and_evaluate --iter_unit=epoch --num_iter=5     --batch_size=64 --warmup_steps=100 --use_cosine --label_smoothing 0.1     --lr_init=0.256 --lr_warmup_epochs=8 --momentum=0.875 --weight_decay=3.0517578125e-05     --use_tf_amp --use_static_loss_scaling --loss_scale 128 --data_dir=${DATA_DIR}
    ```

2. Finetune a RN50 model with quantization nodes and save the final checkpoint.
    
    ```bash
    CUDA_VISIBLE_DEVICES=6 sh resnet50v1.5/training/GPU1_RN50_QAT.sh result_train/ /raid/data/ImageNet/2012/tfrecord-mini result_dir
    ```

3. Post process the above RN50 QAT checkpoint by reshaping the weights of final FC layer into a 1x1 conv layer.
   
    ```bash
    ll result_dir/

    -rw-r--r--  1 root root       271 Mar 26 08:46 checkpoint
    -rw-r--r--  1 root root         8 Mar 26 08:32 model.ckpt-6000.data-00000-of-00002
    -rw-r--r--  1 root root 204687240 Mar 26 08:32 model.ckpt-6000.data-00001-of-00002
    -rw-r--r--  1 root root     40619 Mar 26 08:32 model.ckpt-6000.index
    -rw-r--r--  1 root root   7811299 Mar 26 08:33 model.ckpt-6000.meta
    ...
    -rw-r--r--  1 root root         8 Mar 26 08:46 model.ckpt-9380.data-00000-of-00002
    -rw-r--r--  1 root root 204687240 Mar 26 08:46 model.ckpt-9380.data-00001-of-00002
    -rw-r--r--  1 root root     40619 Mar 26 08:46 model.ckpt-9380.index
    -rw-r--r--  1 root root   7811299 Mar 26 08:46 model.ckpt-9380.meta
    ```
    
    ```bash
    CUDA_VISIBLE_DEVICES="" python3 postprocess_ckpt.py --input result_dir --output postprocess_result_dir
    ```

Note: 

- `postprocess_ckpt.py` is a utility to convert the final classification FC layer into a 1x1 convolution layer using the same weights. This is required to ensure TensorRT can parse QAT models successfully.

	- Arguments:

		* `--input` : Input folder to the trained (QAT) checkpoint of RN50.
		* `--output` : Output folder of the new checkpoint file which has the FC layer weights reshaped into 1x1 conv layer weights.


### Step 2: Export frozen graph of RN50 QAT 

To export frozen graphs (which can be used for inference with <a href="https://developer.nvidia.com/tensorrt">TensorRT</a>), use:

```bash

export PATH_TO_CHECKPOINT=./postprocess_result_dir/new.ckpt
export OUTPUT_FILE_NAME=./postprocess_result_dir/resnet50_qat_frozen_graph.pb

CUDA_VISIBLE_DEVICES=6 python3 export_frozen_graph.py --checkpoint $PATH_TO_CHECKPOINT --quantize --use_final_conv --use_qdq --symmetric --input_format NCHW --compute_format NCHW --output_file=$OUTPUT_FILE_NAME

```

### Step 3: Constant folding

run the following command to perform constant folding on TF graph

```bash

export INPUT_PB_NAME=./postprocess_result_dir/resnet50_qat_frozen_graph.pb
export OUTPUT_PB_NAME=./postprocess_result_dir/resnet50_qat_step3.pb

CUDA_VISIBLE_DEVICES=6 python3 fold_constants.py --input $INPUT_PB_NAME --output $OUTPUT_PB_NAME

```

Arguments:
* `--output_node` : Output node name of the RN50 graph (Default: `resnet50_v1.5/output/softmax_1`)



### Step 4: TF2ONNX conversion

For RN50 QAT, `tf.quantization.quantize_and_dequantize` operation (QDQ) is converted into `QuantizeLinear` and `DequantizeLinear` operations.
Support for converting QDQ operations has been added in `1.6.1` version of TF2ONNX.

Command to convert RN50 QAT TF graph to ONNX

```bash

export INPUT_PB_NAME=./postprocess_result_dir/resnet50_qat_step3.pb
export OUTPUT_FILE_NAME=./postprocess_result_dir/resnet50_qat_step4.onnx
CUDA_VISIBLE_DEVICES=6 python3 -m tf2onnx.convert --input $INPUT_PB_NAME --output $OUTPUT_FILE_NAME --inputs input:0 --outputs resnet50/output/softmax_1:0 --opset 11
```

Arguments:
* `--inputs` : Name of input tensors
* `--outputs` : Name of output tensors
* `--opset` : ONNX opset version



### Step 5: Post processing ONNX

Run the following command to postprocess the ONNX graph using ONNX-Graphsurgeon API. This step removes the `transpose` nodes after `Dequantize` nodes. 

```bash
wget https://github.com/NVIDIA/TensorRT/archive/refs/heads/release/7.1.zip && unzip 7.1.zip  && mv TensorRT-release-7.1/tools/onnx-graphsurgeon ./  && rm -rf 7.1.zip TensorRT-release-7.1

cd onnx-graphsurgeon && make install && cd .. && pip3 list|grep onnx-graphsurgeon
```

```bash

export INPUT_FILE_NAME=./postprocess_result_dir/resnet50_qat_step4.onnx
export OUTPUT_FILE_NAME=./postprocess_result_dir/resnet50_qat_step5.onnx
CUDA_VISIBLE_DEVICES=6 python3 postprocess_onnx.py --input $INPUT_FILE_NAME --output $OUTPUT_FILE_NAME
```


### Step 6: Build TensorRT engine from ONNX graph

Make sure run following steps with TensorRT 7.1:

- exit the current docker container (`nvcr.io/nvidia/tensorflow:20.01-tf1-py3`)
- start TensorRT docker container: `docker run -it --gpus all --network=host -v /raid:/raid nvcr.io/nvidia/tensorrt:20.08-py3 /bin/bash`


```bash
cd /path/to/your/workspace
export INPUT_FILE_NAME=./postprocess_result_dir/resnet50_qat_step5.onnx
python3 build_engine.py --onnx $INPUT_FILE_NAME
```

Arguments:
* `--onnx` : Path to RN50 QAT onnx graph 
* `--engine` : Output file name of TensorRT engine.
* `--verbose` : Flag to enable verbose logging


### Step 7: TensorRT Inference

Command to run inference on a sample image

```bash
python3 infer.py --engine <input_trt_engine>
```

Arguments:
* `--engine` : Path to input RN50 TensorRT engine. 
* `--labels` : Path to imagenet 1k labels text file provided.
* `--image` : Path to the sample image
* `--verbose` : Flag to enable verbose logging


## Reference


This guide is prepared based on following work:

- [Resnet-50 Deep Learning Example](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/Classification/ConvNets/resnet50v1.5/README.md) and its [QAT section](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5#quantization-aware-training)

- [sampleQAT](https://github.com/NVIDIA/sampleQAT)



The following resources provide a deeper understanding about Quantization aware training, TF2ONNX and importing a model into TensorRT using Python:

**Quantization Aware Training**
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/pdf/1712.05877.pdf)
- [Quantization Aware Training guide](https://www.tensorflow.org/model_optimization/guide/quantization/training)


**Parsers**
- [TF2ONNX Converter](https://github.com/onnx/tensorflow-onnx)
- [ONNX Parser](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/parsers/Onnx/pyOnnx.html)

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The Python API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)
- [Importing A Model Using A Parser In Python](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#import_model_python)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)