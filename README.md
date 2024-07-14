# Intel_LLM

This repository contains the files of the project titled "Running GenAI on Intel AI Laptops and Simple LLM Inference on CPU and fine-tuning of LLM Models using Intel® OpenVINO™".

## Project Description

This project leverages the TinyLlama model and optimizes it using Intel® OpenVINO™ to create a responsive chatbot. The chatbot is deployed using Gradio for an easy-to-use web interface. The project includes scripts to convert the model to OpenVINO format and compress it for better performance.

## How to Run

### Step 1: Clone the Repository

First, clone the repository to your local machine:

```sh
git clone git@github.com:adilzubair/Bitmasters_Intel_LLM.git
cd Bitmasters_Intel_LLM
```

### Step 2: Install Dependencies

To install the necessary dependencies, run:

```sh
python setup.py
```

### Step 3: Convert and Compress the Model

Before running the chatbot, you need to convert the TinyLlama model to the OpenVINO format and optionally compress it for better performance.

To convert and compress the model, run:

```sh
python convert_model.py
```

### Step 4: Ensure openvino_model Directory Exists

Make sure the openvino_model directory is created and contains the converted model files. The convert_model.py script will handle this for you.

### Step 5: Running the Chatbot

After converting the model, you can run the chatbot using:

```sh
python chatbot.py
```

## Usage

The chatbot interface is powered by Gradio. You can adjust the advanced settings such as temperature, top-p, top-k, and repetition penalty to control the behavior of the model's responses.
Advanced Options

#### 1.Temperature: Controls the randomness of the model's output. Higher values result in more random responses.
#### 2.Top-p: Controls the cumulative probability of token selection.
#### 3.Top-k: Limits the number of token choices to the top k tokens.
#### 4.Repetition Penalty: Penalizes repeated tokens to promote more diverse outputs.
