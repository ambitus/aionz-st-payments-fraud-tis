# Linux on Z AI Payments Fraud Detection Solution Template

This is a [Linux on Z AI solution template](https://ambitus.github.io/aionz-solution-templates/) for **Authorized Push Payment (APP) fraud detection**. This solution template provides an example of how to deploy AI using an IBM Linux on Z environment, leveraging open source frameworks, Triton Inference Server, and modern deep learning techniques.

Within this solution template, various phases of the AI lifecycle are included. Work through the following steps to deploy your own payments fraud detection solution on IBM Z.

## Overview

This solution template focuses on detecting **Authorized Push Payment (APP) fraud** using LSTM (Long Short-Term Memory) neural networks. APP fraud occurs when fraudsters manipulate victims into authorizing payments to accounts controlled by criminals, making it a critical challenge for financial institutions.

### Key Features

- **LSTM-based Deep Learning Model**: Utilizes recurrent neural networks to capture temporal patterns in transaction sequences
- **Comprehensive Feature Engineering**: Incorporates transaction details, account characteristics, bank metrics, and behavioral patterns
- **Multi-Dataset Integration**: Combines data from banks, accounts (people and companies), and bank transfers
- **ONNX Model Export**: Unified pipeline with reprocessing and trained model exported in ONNX format for optimized inference
- **IBM Telum AI Acceleration**: Leverages IBM Z Integrated Accelerator for AI with ONNX Runtime
- **Triton Inference Server**: Production-ready model serving with high-performance inference and Snap ML preprocessing

### IBM Supported Components
- **AI Toolkit for IBM Z and LinuxONE** A family of popular open-source AI frameworks with IBM Elite Support and adapted for IBM Z and LinuxONE hardware.
- **IBM Synthetic Data Sets**  A family of artificially generated, enterprise-grade datasets that enhance predictive artificial intelligence (AI) model training and large language models (LLMs).

## Use Cases

This solution template is designed for:

- **Real-time Fraud Detection**: Score transactions as they occur
- **Batch Processing**: Analyze historical transactions for fraud patterns
- **Risk Assessment**: Evaluate account and transaction risk levels
- **Fraud Investigation**: Support fraud analysts with AI-powered insights

## Performance Benefits

- **Hardware Acceleration**: Faster inference using IBM Telum AI accelerator
- **Low Latency**: Optimized for real-time fraud detection requirements
- **High Throughput**: Efficient batch processing for large transaction volumes
- **Energy Efficiency**: Reduced power consumption with dedicated AI hardware


## Solution Architecture
This solution consists of two main components:

### Part 1: Model Training
**Location**: `zST-model-training-jupyter/APP_Fraud_LSTM_model_training.ipynb`

### Part 2: Model Serving
**Location**: `zST-model-training-jupyter/APP_Fraud_LSTM_model_serving.ipynb`

```
┌─────────────────────────────────────────────────────────────────────┐
│                          PART 1: MODEL TRAINING                     │
│              (APP_Fraud_LSTM_model_training.ipynb)                  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     Data Sources                            │
│  (Banks, Accounts, Transactions)                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Preparation & Feature Engineering         │
│  - Merge datasets                                           │
│  - Clean and transform features                             │
│  - Handle missing values                                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            Preprocessing Pipeline (Keras)                   │
│  - Scaling (MinMax)                                         │
│  - Encoding (OneHot, Ordinal)                               │
│  - Imputation                                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    LSTM Model Training  (Keras)             │
│  - Sequential pattern learning                              │
│  - Fraud detection optimization                             │
│  - Model evaluation and validation                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   Model & Preprocessor Export               │
│  - ONNX model: fraud_detection_model.onnx                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │
┌─────────────────────────────────────────────────────────────────────┐
│                          PART 2: MODEL SERVING                      │
│              (APP_Fraud_LSTM_model_serving.ipynb)                   │
└─────────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              NVIDIA Triton Inference Server                 │
│  - Model repository configuration                           │
│  - Server initialization                                    │
│  - Health checks and monitoring                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────────┐
│              Inference Pipeline in Triton                  │
│                                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Step 1: ONNX Runtime with IBM Telum EP             │   │
│  │  - Load fraud_detection_model.onnx                  │   │
│  │  - Initialize Telum Execution Provider              │   │
│  │  - Hardware-accelerated LSTM inference              │   │
│  │  ┌───────────────────────────────────────────────┐  │   │
│  │  │  IBM Z Integrated Accelerator for AI          │  │   │
│  │  │  (Telum Processor)                            │  │   │
│  │  │  - On-chip AI acceleration                    │  │   │
│  │  │  - Low-latency inference                      │  │   │
│  │  │  - Energy-efficient processing                │  │   │
│  │  └───────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────┬───────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Real-time Fraud Detection                  │
│  - Preprocessing and LSTM inference with Telum acceleration │
│  - Fraud probability prediction                             │
│  - Low-latency responses (<1ms)                             │
│  - High-throughput batch processing                         │
└─────────────────────────────────────────────────────────────┘
```


## Solution Components

### 1. Model Training Notebook

**Location**: `zST-model-training-jupyter/APP_Fraud_LSTM_model_training.ipynb`

This Jupyter notebook provides the end-to-end training pipeline for the APP fraud detection model:

#### Data Preparation
- Loads datasets from IBM Synthetic Datasets:
  - `us_small_banks.csv` - Bank information and metrics
  - `us_small_liquid_accts_people.csv` - Personal account details
  - `us_small_liquid_accts_companies.csv` - Company account details
  - `us_small_bank_xfers-chrono.csv` - Transaction records

#### Feature Engineering
The model uses rich feature sets including:
- **Transaction Features**: Amount, currency, format, type, day of week
- **Fraud Labels**: `Is_APP_Fraud` target variable
- **Sender Account Features**: Country, currency, entity type, overdraft limits, branch/bank metrics
- **Recipient Account Features**: Country, entity type
- **Temporal Features**: Cyclical encoding of time-based patterns
- **Boolean Indicators**: Sufficient funds, overdraft status, hold status

#### Preprocessing Pipeline with Keras
The preprocessing pipeline is built using **Keras** and exported to **ONNX format** for production deployment:
- **Temporal/Cyclical Features**: MinMax scaling for time-based features
- **Boolean Features**: Pass-through for binary indicators
- **Amount Features**: Imputation (constant 0) + MinMax scaling
- **Count Features**: Median imputation + MinMax scaling
- **Low Cardinality Categorical**: Most frequent imputation + One-Hot encoding
- **High Cardinality Categorical**: Most frequent imputation + Ordinal encoding + MinMax scaling

#### Model Architecture
The model is built using **Keras** and exported to **ONNX format** for production deployment.
- **LSTM-based Neural Network**: Captures sequential patterns in transaction data
- **Training Configuration**: Optimized for fraud detection with class imbalance handling
- **Evaluation Metrics**: Precision, recall, F1-score, and AUC-ROC

#### Model Export
The trained model and preprocessing pipeline are exported for production deployment:
- **ONNX Model** (`.onnx`): Primary deployment format for LSTM model with Telum AI acceleration
- **Keras Format** (`.keras`): Native TensorFlow/Keras format for reference
- **Model Weights** (`.weights.h5`): Separate weight files

### 2a. Model Serving Notebook

**Location**: `zST-model-serving-jupyter/APP_Fraud_LSTM_model_serving.ipynb`

This Jupyter notebook demonstrates how to serve the trained ONNX model using **ONNX Runtime** and the **IBM Telum AI Execution Provider**.

### 2b. Model Serving Container

**Location**: `zST-model-serving-triton/Dockerfile`
This Triton container demonstrates how to serve the trained ONNX model using **NVIDIA Triton**, **ONNX Runtime** and the **IBM Telum AI Execution Provider**.

#### Key Components

- **Triton Inference Server**: Industry-standard inference serving solution supporting multiple frameworks
- **ONNX Runtime**: High-performance inference engine for ONNX models
- **IBM Telum AI Execution Provider**: Hardware acceleration leveraging IBM Z Integrated Accelerator for AI
  - Repository: [IBM/onnxruntime-ep-telum](https://github.com/IBM/onnxruntime-ep-telum)
  - Provides optimized inference on IBM z16 and newer systems
  - Accelerates neural network operations using on-chip AI accelerator

#### Inference Pipeline

1. **Load ONNX Preprocessor**: Import the adapted preprocessing pipeline from `fraud_detection_model.onnx`
2. **Load ONNX Model**: Import the trained `fraud_detection_model.onnx` model
3. **Configure Triton Server**: Set up model repository with unified preprocessing and inference model
4. **Initialize ONNX Runtime with Telum EP**: Enable IBM Z AI acceleration
5. **Real-time Inference**:
   - Preprocess incoming transactions preprocessing pipeline
   - Score preprocessed features with LSTM model
   - Return fraud predictions with low-latency
6. **Batch Processing**: Handle high-throughput fraud detection workloads

### 3. Datasets Directory

**Location**: `zST-model-training-jupyter/datasets/`

Obtain a trial copy of the IBM Synthetic Data Sets and place it in this directory. The notebook expects the following CSV files:
- `us_small_banks.csv`
- `us_small_liquid_accts_people.csv`
- `us_small_liquid_accts_companies.csv`
- `us_small_bank_xfers-chrono.csv`

## Getting Started

### Prerequisites

#### For Accelerating on IBM Z and IBM LinuxONE

#### For Model Training
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Required Python packages:
  ```
  pandas
  numpy
  tensorflow
  keras
  joblib
  tf2onnx
  onnx
  ```

#### For Model Serving and Acceleration
- Obtain a set of open-source AI packages in [AI Toolkit for IBM Z and LinuxONE.](https://ibm.github.io/ai-on-z-101/aitoolkitloz/)
- IBM Z16 or newer system with Telum processor (for AI acceleration)
- NVIDIA Triton Inference Server with Snap ML support
- ONNX Runtime with IBM Telum Execution Provider
  - Installation: [IBM/onnxruntime-ep-telum](https://github.com/IBM/onnxruntime-ep-telum)
- Python 3.8 or higher
- tritonclient Python package

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-org/aionz-st-payments-fraud-tis.git
   cd aionz-st-payments-fraud-tis
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your datasets in the `zST-model-training-jupyter/datasets/` directory

### Running the Training Pipeline

1. Navigate to the Jupyter notebook directory:
   ```bash
   cd zST-model-training-jupyter
   ```

2. Launch Jupyter and open the training notebook:
   ```bash
   jupyter notebook APP_Fraud_LSTM_model_training.ipynb
   ```

3. Execute the notebook cells sequentially to:
   - Load and prepare datasets
   - Engineer features
   - Build preprocessing pipeline with Keras
   - Train the LSTM model
   - Evaluate model performance
   - Export adapted preprocessing and trained model in unified ONNX format

### Training Output Artifacts

After successful training, the following artifacts will be generated:

- `saved_model/fraud_detection_unified.onnx` - **ONNX format model (primary deployment artifact)**
- `saved_model/model.keras` - Complete Keras model (reference)
- `saved_model/wts.weights.h5` - Model weights (reference)

## Model Serving in Jupyter Notebooks

**Location**: `zST-model-serving-jupyter/APP_Fraud_LSTM_model_serving.py`

This notebook/script demonstrates direct ONNX Runtime inference against the unified ONNX model — no server required.

### Prerequisites

- Python 3.9+
- ONNX Runtime:
  ```bash
  pip install onnxruntime numpy
  ```
- On IBM z16 / LinuxONE, install the Telum Execution Provider instead:
  ```bash
  # Follow build & install instructions at:
  # https://github.com/IBM/onnxruntime-ep-telum
  ```
- Training output artifact: `saved_model/fraud_detection_unified.onnx`
  (produced by the training notebook — run Part 1 first)

### Steps

#### 1 · Ensure the trained model is available

After running the training notebook, confirm the ONNX file exists:
```
saved_model/fraud_detection_unified.onnx
```

#### 2 · Install dependencies

```bash
pip install onnxruntime numpy
```

#### 3 · Open the serving notebook

```bash
cd zST-model-serving-jupyter
jupyter notebook APP_Fraud_LSTM_model_serving.py
# or run as a plain Python script:
python3 APP_Fraud_LSTM_model_serving.py
```

#### 4 · Load the model and select Execution Provider

The script auto-detects the best available provider:

```python
import onnxruntime as ort

ONNX_MODEL_PATH = './saved_model/fraud_detection_unified.onnx'
TELUM_EP = 'TelumExecutionProvider'
CPU_EP   = 'CPUExecutionProvider'

available = ort.get_available_providers()
providers = [TELUM_EP, CPU_EP] if TELUM_EP in available else [CPU_EP]

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    ONNX_MODEL_PATH,
    sess_options=sess_options,
    providers=providers,
)
print(f'Active EP: {session.get_providers()[0]}')
```

> On IBM z16 / LinuxONE with the Telum EP installed the AIU is used automatically.
> On all other platforms the standard CPU EP is used.

#### 5 · Inspect model inputs and outputs

The unified ONNX model accepts **24 named inputs**, one tensor per feature, each shaped `[batch, 7, 1]`:

| Tensor group | dtype | Shape |
|---|---|---|
| 15 numeric inputs (`month` … `from_bank_num_total_locations`) | `FP32` | `[batch, 7, 1]` |
| 9 categorical inputs (`from_entity_type` … `payment_currency`) | `BYTES` (string) | `[batch, 7, 1]` |
| **`output_0`** (fraud probability) | `FP32` | `[batch, 7, 1]` |

The fraud probability for the **most-recent transaction** is `output_0[:, -1, 0]`.

#### 6 · Run inference on sample transactions

```python
# Build the feed dict (one ndarray per input, shape [N, 7, 1])
feed = sequences_to_feed(sequences)   # helper defined in the notebook

# Run
raw = session.run(['output_0'], feed)[0]   # [N, 7, 1]
fraud_probs = raw[:, -1, 0]               # last-timestep score per sequence

THRESHOLD = 0.5
for i, prob in enumerate(fraud_probs):
    label = 'FRAUD' if prob >= THRESHOLD else 'GENUINE'
    print(f'Sequence {i+1}: score={prob:.4f}  →  {label}')
```

#### 7 · Live single-transaction scoring

Maintain a sliding window of the last 6 transactions, append the new transaction, then call the model:

```python
history_window  = last_6_transactions_for_account   # list of 6 rows
new_transaction = [month, day_of_month, ..., payment_currency]
live_sequence   = [history_window + [new_transaction]]  # shape [1, 7, 24]

live_prob, _ = run_inference(live_sequence)
print(f'Fraud score: {live_prob[0]:.4f}')
```

---

## Model Serving with Triton container

**Location**: `zST-model-serving-triton/`

The trained ONNX model is served via **NVIDIA Triton Inference Server** running as a Docker container. Triton exposes HTTP/REST, gRPC, and Prometheus endpoints.

### Prerequisites

| Requirement | Notes |
|---|---|
| Docker 20.10+ | container runtime |
| Python 3.9+ | host-side test client only |
| `tritonclient[http]`, `numpy` | `pip install tritonclient[http] numpy` |
| `fraud_detection_unified.onnx` | produced by the training notebook |

### Steps

#### 1 · Train the model (if not already done)

```bash
cd zST-model-training-jupyter
jupyter nbconvert --to notebook --execute APP_Fraud_LSTM_model_training.ipynb
cd ..
```

This produces `saved_model/fraud_detection_unified.onnx`.

#### 2 · Set up the Triton model repository

```bash
cd zST-model-serving-triton
python3 serve.py
# optional overrides:
#   --model-path /path/to/fraud_detection_unified.onnx
#   --repo-dir   /path/to/model_repository
```

`serve.py` copies the ONNX file and writes `config.pbtxt`. Only needs to run once (or after a model update).

Expected output layout:
```
model_repository/
└── fraud_detection/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

#### 3 · Build the Docker image

```bash
# IBM z16 / LinuxONE (default — uses IBM Z accelerated Triton image):
docker build -t triton-fraud-detection .

# x86_64 development machine:
docker build \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/tritonserver:24.01-py3 \
  -t triton-fraud-detection .
```

#### 4 · Start the container

```bash
docker run -d \
  --name triton-server \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v $(pwd)/model_repository:/models:ro \
  triton-fraud-detection
```

| Port | Protocol | Purpose |
|------|----------|---------|
| 8000 | HTTP/REST | Inference, health, metadata |
| 8001 | gRPC | Inference (gRPC protocol) |
| 8002 | HTTP | Prometheus metrics (`/metrics`) |

#### 5 · Verify server health

```bash
# Server live
curl http://localhost:8000/v2/health/live

# Server ready
curl http://localhost:8000/v2/health/ready

# Model metadata
curl http://localhost:8000/v2/models/fraud_detection

# Prometheus metrics
curl http://localhost:8002/metrics
```

#### 6 · Run the test client

```bash
pip install tritonclient[http] numpy
python3 test_triton_client.py [--url localhost:8000]
```

The test suite covers:

| Test | Description |
|------|-------------|
| 1 · Server health | `is_server_live`, `is_server_ready`, `is_model_ready` |
| 2 · Genuine sequences | G-1 (retail), G-2 (business) — expects score < 0.5 |
| 3 · Fraud sequences | F-1 (impersonation scam), F-2 (invoice fraud) — expects score ≥ 0.5 |
| 4 · Timestep evolution | Per-timestep probabilities for all 4 sequences |
| 5 · Summary table | Accuracy across 4 labelled sequences |
| 6 · Live scoring | Suspicious late-night wire appended to genuine history |
| 7 · Throughput benchmark | 1 000-sequence timed batch |

#### 7 · One-shot build + launch (alternative)

```bash
cd zST-model-serving-triton
bash build_and_deploy.sh
```

The script runs steps 1–6 automatically and prints the endpoints when the server is ready.

#### 8 · (Optional) Enable IBM Telum AIU acceleration

Add the following `optimization` block to `model_repository/fraud_detection/config.pbtxt` before starting the container:

```protobuf
optimization {
  execution_accelerators {
    cpu_execution_accelerator [
      {
        name: "telum"
      }
    ]
  }
}
```

This routes compatible ONNX operators to the on-chip **Telum AI Unit (AIU)** on IBM z16 / LinuxONE Emperor 4 via the [IBM ONNX Runtime Telum EP](https://github.com/IBM/onnxruntime-ep-telum).

#### 9 · Stop / remove the container

```bash
docker stop triton-server && docker rm triton-server
```

### Deployment Architecture

| Component | Description |
|---|---|
| `fraud_detection_unified.onnx` | Unified ONNX model — preprocessing + LSTM + sigmoid output head |
| `config.pbtxt` | Triton model configuration (inputs, outputs, dynamic batching) |
| Triton Inference Server | Production inference platform (HTTP, gRPC, Prometheus) |
| ONNX Runtime backend | High-performance ONNX execution within Triton |
| IBM Telum EP *(optional)* | Hardware AI acceleration on IBM z16 / LinuxONE |


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms specified in the LICENSE file.

## Authors & Contributors

    - Pui Yen (MVI Tech)
    - Alex Osadchyy (IBM)
    - Erik Altman 
    - Saurabh Srivastava

## Related Resources

- [IBM Z 101](https://www.ibm.com/z/learn/101)
- [Linux on Z AI Solution Templates](https://ambitus.github.io/aionz-solution-templates/)
- [IBM Snap ML](https://www.zurich.ibm.com/snapml/) - High-performance machine learning library for IBM Z
- [IBM ONNX Runtime Telum Execution Provider](https://github.com/IBM/onnxruntime-ep-telum)
- [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
- [ONNX Runtime](https://onnxruntime.ai/)
- [IBM Z AI Solutions](https://www.ibm.com/z/artificial-intelligence)
- [IBM Z Integrated Accelerator for AI](https://www.ibm.com/z/artificial-intelligence)