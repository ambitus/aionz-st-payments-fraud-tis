# Linux on Z AI Payments Fraud Detection Solution Template

This is a [Linux on Z AI solution template](https://ambitus.github.io/aionz-solution-templates/) for **Authorized Push Payment (APP) fraud detection**. This solution template provides an example of how to deploy AI using an IBM Linux on Z environment, leveraging open source frameworks, Triton Inference Server, and modern deep learning techniques.

Within this solution template, various phases of the AI lifecycle are included. Work through the following steps to deploy your own payments fraud detection solution on IBM Z.

## Overview

This solution template focuses on detecting **Authorized Push Payment (APP) fraud** using LSTM (Long Short-Term Memory) neural networks. APP fraud occurs when fraudsters manipulate victims into authorizing payments to accounts controlled by criminals, making it a critical challenge for financial institutions.

### Key Features

- **LSTM-based Deep Learning Model**: Recurrent neural network capturing temporal patterns across 7-transaction sliding windows
- **Unified ONNX Export**: Preprocessing (cyclical encoding, log transforms, normalisations, vocabulary lookups) and the two-layer LSTM are fused into a single self-contained ONNX file — no separate preprocessing step at serving time
- **Comprehensive Feature Engineering**: 15 numeric and 9 categorical features per transaction covering amounts, account characteristics, bank metrics, and temporal patterns
- **IBM Telum AI Acceleration**: On-chip AI acceleration via the IBM Z Integrated Accelerator for AI with ONNX Runtime
- **Triton Inference Server**: Production-ready model serving with HTTP/REST, gRPC, and Prometheus endpoints

### IBM Supported Components
- **AI Toolkit for IBM Z and LinuxONE** — A family of popular open-source AI frameworks with IBM Elite Support adapted for IBM Z and LinuxONE hardware.
- **IBM Synthetic Data Sets** —  are pre-built, artificially generated enterprise data collections designed to train and improve predictive artificial intelligence (AI) models without exposing real-world sensitive information.

## Use Cases

- **Real-time Fraud Detection**: Score transactions as they occur with sub-millisecond latency
- **Batch Processing**: Analyse historical transactions for fraud patterns
- **Risk Assessment**: Evaluate account and transaction risk levels
- **Fraud Investigation**: Support fraud analysts with AI-powered insights

## Performance Benefits

- **Hardware Acceleration**: Faster inference using the IBM Telum AI accelerator
- **Low Latency**: Optimised for real-time fraud detection requirements
- **High Throughput**: Efficient batch processing for large transaction volumes
- **Energy Efficiency**: Reduced power consumption with dedicated AI hardware


## Solution Architecture

This solution consists of three components:

| Part | Notebook / Directory | Description |
|------|----------------------|-------------|
| **1 — Model Training** | `zST-model-training-jupyter/APP_Fraud_LSTM_model_training.ipynb` | Data preparation, feature engineering, LSTM training, ONNX export |
| **2 — Serving (Jupyter)** | `zST-model-serving-jupyter/APP_Fraud_LSTM_model_serving.ipynb` | Direct ONNX Runtime inference — no server required |
| **3 — Serving (Triton)** | `zST-model-serving-triton/` | Production serving via Triton Inference Server in Docker |

```
┌──────────────────────────────────────────────────────────────────────┐
│           PART 1: MODEL TRAINING                                     │
│           zST-model-training-jupyter/                                │
│           APP_Fraud_LSTM_model_training.ipynb                        │
└──────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  IBM Synthetic Datasets                                     │
│  us_small_banks.csv · us_small_liquid_accts_people.csv      │
│  us_small_liquid_accts_companies.csv                        │
│  us_small_bank_xfers-chrono.csv                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Data Preparation & Feature Engineering                     │
│  - Join banks / accounts / transactions                     │
│  - Cyclical temporal encoding (month, day, hour)            │
│  - Boolean cast, NaN fill, categorical normalisation        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Keras Preprocessing Model                                  │
│  - CyclicalEncoding + Normalization (temporal features)     │
│  - LogTransform + Normalization (amounts, counts)           │
│  - OnnxVocabOneHot (low-cardinality categoricals)           │
│  - OnnxVocabOrdinal + Normalization (high-cardinality)      │
│  - Boolean pass-through                                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  LSTM Model Training  (Keras / TensorFlow)                  │
│  - 2 × LSTM(200) stacked layers                             │
│  - Focal loss · EarlyStopping on train F1                   │
│  - Chronological 50/30/20 train/val/test split              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Unified ONNX Export                                        │
│  saved_model/fraud_detection_unified.onnx                   │
│  (preprocessing + LSTM + sigmoid — single file)             │
└──────┬──────────────────────────────────────┬───────────────┘
       │                                      │
       ▼                                      ▼
┌──────────────────────────────┐  ┌──────────────────────────────────┐
│  PART 2: SERVING (JUPYTER)   │  │  PART 3: SERVING (TRITON)        │
│  zST-model-serving-jupyter/  │  │  zST-model-serving-triton/       │
│  APP_Fraud_LSTM_model_       │  │  Dockerfile · serve.py ·         │
│    serving.ipynb             │  │    test_triton_client.py         │
│                              │  │                                  │
│  ONNX Runtime (direct)       │  │  NVIDIA Triton Inference Server  │
│  TelumExecutionProvider      │  │  ONNX Runtime backend            │
│    or CPUExecutionProvider   │  │  TelumExecutionProvider (opt.)   │
│  (auto-detected)             │  │                                  │
│  Peak fraud score across     │  │  HTTP · gRPC · Prometheus        │
│  7 timesteps                 │  │  Dynamic batching                │
└──────────────────────────────┘  └──────────────────────────────────┘
```


## Solution Components

### 1. Model Training Notebook

**Location**: `zST-model-training-jupyter/APP_Fraud_LSTM_model_training.ipynb`

End-to-end training pipeline for the APP fraud detection model.

#### Data Preparation
Loads and joins four IBM Synthetic Dataset CSV files:
- `us_small_banks.csv` — bank information and transaction counts
- `us_small_liquid_accts_people.csv` — personal account details
- `us_small_liquid_accts_companies.csv` — company account details
- `us_small_bank_xfers-chrono.csv` — chronologically ordered transaction records

The join enriches each transaction with sender-account features (country, currency, entity type, overdraft limit, branch/bank metrics) and recipient-account features (country, entity type), while filtering out cash transactions.

#### Feature Engineering
Twenty-four features are prepared per transaction:

| Group | Features | Preprocessing |
|---|---|---|
| Temporal | Month, Day_Of_Month, Day_Of_Week, Hour, Minute | `CyclicalEncoding` → `Normalization` |
| Boolean | Is_Weekday, Sufficient_Funds, Overdraft_Okay, Is_Hold | Cast to int, pass-through |
| Amounts | Amount_Paid, From_End_Balance, From_Account_Max_Overdraft | `LogTransform` → `Normalization` |
| Counts | From_Branch_Account_Count, From_Bank_Num_Transactions, From_Bank_Num_Total_Locations | `Normalization` |
| Low-cardinality categorical (5) | From/To_Entity_Type, From_Account_Type, Payment_Format, Transaction_Type | `OnnxVocabOneHot` |
| High-cardinality categorical (4) | From/To_Account_Country, From/To_Account_Currency | `OnnxVocabOrdinal` → `Normalization` |

All preprocessing layers (`CyclicalEncoding`, `LogTransform`, `TimeOfDayEncoding`, `OnnxVocabOneHot`, `OnnxVocabOrdinal`) are defined in `fraud_model_layers.py` and are ONNX-compatible — they use only opset-13 primitives so they can be traced and exported.

#### Model Architecture
- **Input**: 7-transaction sliding window; one `[batch, 7, 1]` tensor per feature
- **Preprocessing**: Keras functional model applied per timestep (TimeDistributed-equivalent)
- **LSTM**: Two stacked `LSTM(200, return_sequences=True)` layers
- **Output**: `Dense(1, sigmoid)` — fraud probability at every timestep; `[batch, 7, 1]`
- **Loss**: Focal loss (`gamma=2`, `alpha=0.25`) to handle class imbalance
- **Early stopping**: Monitors training F1 (not `val_F1` — temporal distribution shift means val fraud scores differ from training patterns)

#### Model Export
The training notebook exports a single unified ONNX file that bundles preprocessing and LSTM together:

```
saved_model/
├── fraud_detection_unified.onnx   ← primary deployment artifact (preprocessing + LSTM)
├── fraud_detection_model.keras    ← Keras format (reference)
└── model_weights.weights.h5       ← weights only (reference)
```

`fraud_detection_unified.onnx` takes 24 raw feature tensors as input and returns per-timestep fraud probabilities. No separate preprocessing step is needed at serving time.

---

### 2. Model Serving Notebook

**Location**: `zST-model-serving-jupyter/APP_Fraud_LSTM_model_serving.ipynb`

Demonstrates direct ONNX Runtime inference — no server or Docker required. The notebook loads `fraud_detection_unified.onnx`, auto-detects the best available Execution Provider, and runs inference against embedded sample sequences from the IBM Synthetic Payments dataset.

#### What it covers

| Section | Description |
|---|---|
| 1 · Dependencies | `pip install onnxruntime numpy` — no TensorFlow at serving time |
| 2 · Load model & EP | Auto-detects `TelumExecutionProvider` (IBM z16+) or falls back to `CPUExecutionProvider` |
| 3 · Inspect inputs/outputs | Lists all 24 input tensors and the `Identity:0` output |
| 4 · Sample data | 2 genuine + 4 real APP fraud sequences from the CSV dataset |
| 5 · Inference helper | `sequences_to_feed()` builds the `[N, 7, 1]` feed dict; `run_inference()` returns peak fraud score |
| 6 · Inference results | Per-sequence fraud scores and classification at threshold 0.5 |
| 7 · Timestep evolution | Per-timestep fraud probability tables for all 6 sequences |
| 8 · Summary table | Accuracy across all 6 labelled sequences |
| 9 · Live scoring | Sliding-window single-transaction example (appends suspicious Wire to genuine history) |
| 10 · Benchmark | 1 000-sequence throughput benchmark |

#### Key design notes

- **Output tensor name**: `Identity:0` (the ONNX graph output as produced by `tf2onnx`)
- **Scoring**: Peak probability across all 7 timesteps — `raw[:, :, 0].max(axis=1)` — rather than the last timestep, because some fraud sequences end with a small cleanup payment after the peak (e.g. F-4: $14 after a $167 k wire)
- **Vocabulary**: All string inputs must match the trained vocabulary exactly — `'United States'` not `'US'`, `'ACH'`/`'Wire'`/`'Debit Non-Prepaid'` not `'ACH Credit'`

---

### 3. Model Serving with Triton Container

**Location**: `zST-model-serving-triton/`

Serves `fraud_detection_unified.onnx` via NVIDIA Triton Inference Server running as a Docker container.

#### Files

| File | Purpose |
|---|---|
| `Dockerfile` | Builds the Triton container image (IBM Z or x86_64) |
| `serve.py` | One-time setup: copies the ONNX file and writes `config.pbtxt` |
| `test_triton_client.py` | HTTP test client (mirrors the serving notebook's sample data and tests) |
| `build_and_deploy.sh` | One-shot build + launch script |
| `model_repository/fraud_detection/config.pbtxt` | Triton model config (24 inputs, `Identity:0` output, dynamic batching) |
| `model_repository/fraud_detection/1/model.onnx` | The deployed ONNX model |

#### Model inputs / outputs

| Tensor | dtype | Shape |
|---|---|---|
| 15 numeric inputs (`month` … `from_bank_num_total_locations`) | `FP32` | `[batch, 7, 1]` |
| 9 categorical inputs (`from_entity_type` … `payment_currency`) | `BYTES` | `[batch, 7, 1]` |
| **`Identity:0`** (fraud probability) | `FP32` | `[batch, 7, 1]` |

#### Test suite

| Test | Description |
|---|---|
| 1 · Server health | `is_server_live`, `is_server_ready`, `is_model_ready` |
| 2 · Genuine sequences | G-1 (checking), G-2 (money market) — expects score < 0.5 |
| 3 · Fraud sequences | F-1…F-4 (real APP fraud from CSV, 4 fraudsters) — expects score ≥ 0.5 |
| 4 · Timestep evolution | Per-timestep probabilities for all 6 sequences |
| 5 · Summary table | Accuracy across all 6 labelled sequences |
| 6 · Live scoring | Suspicious late-night Wire appended to genuine G-1 history |
| 7 · Throughput benchmark | 1 000-sequence timed batch |

---

## Getting Started

### Prerequisites

#### For Model Training
- Python 3.9+
- Jupyter Notebook or JupyterLab
- Required Python packages:
  ```
  pandas numpy tensorflow keras tf2onnx onnx
  ```
- IBM Synthetic Datasets CSV files (see Datasets section below)

#### For Model Serving (Jupyter notebook)
- Python 3.9+
  ```bash
  pip install onnxruntime numpy
  ```
- On IBM z16 / LinuxONE 4+, install the Telum Execution Provider:
  ```bash
  # Follow build & install instructions at:
  # https://github.com/IBM/onnxruntime-ep-telum
  ```

#### For Model Serving (Triton container)
- Docker 20.10+
- Python 3.9+ (host-side test client)
  ```bash
  pip install tritonclient[http] numpy
  ```
- On IBM z16+ / LinuxONE 4+ the `Dockerfile` builds the [IBM ONNX Runtime Telum EP](https://github.com/IBM/onnxruntime-ep-telum) from source during `docker build` — no manual installation needed

> **AI Toolkit for IBM Z and LinuxONE** — For a curated set of supported AI packages see the [AI Toolkit](https://ibm.github.io/ai-on-z-101/aitoolkitloz/).

### Datasets

Place the IBM Synthetic Dataset CSV files in `zST-model-training-jupyter/datasets/`:

```
zST-model-training-jupyter/datasets/
├── us_small_banks.csv
├── us_small_liquid_accts_people.csv
├── us_small_liquid_accts_companies.csv
└── us_small_bank_xfers-chrono.csv
```

---

## Part 1 — Running the Training Pipeline

```bash
cd zST-model-training-jupyter
jupyter notebook APP_Fraud_LSTM_model_training.ipynb
```

Run all cells sequentially. The notebook will:
1. Load and join the four datasets
2. Engineer features (temporal, boolean, amounts, counts, categoricals)
3. Build and adapt the Keras preprocessing model
4. Train the two-layer LSTM with focal loss and EarlyStopping
5. Export the unified ONNX model

### Training output artifacts

```
saved_model/
├── fraud_detection_unified.onnx   ← primary serving artifact
├── fraud_detection_model.keras    ← Keras reference
└── model_weights.weights.h5       ← weights reference

checkpoints/app_fraud_lstm_keras_preprocessing/
├── best.weights.h5                ← best train-F1 epoch (restored by EarlyStopping)
└── iter-NN.weights.h5             ← per-epoch checkpoints
```

---

## Part 2 — Model Serving in Jupyter

**Location**: `zST-model-serving-jupyter/APP_Fraud_LSTM_model_serving.ipynb`

### Prerequisites

```bash
pip install onnxruntime numpy
# IBM z16 / LinuxONE 4+: pip install onnxruntime-telum (or follow repo instructions)
```

Confirm the training artifact exists:
```
saved_model/fraud_detection_unified.onnx
```

### Steps

#### 1 · Open the notebook

```bash
cd zST-model-serving-jupyter
jupyter notebook APP_Fraud_LSTM_model_serving.ipynb
```

#### 2 · Load the model and select Execution Provider

The notebook auto-detects the best available provider:

```python
import onnxruntime as ort

ONNX_MODEL_PATH = './saved_model/fraud_detection_unified.onnx'
TELUM_EP        = 'TelumExecutionProvider'
CPU_EP          = 'CPUExecutionProvider'

available = ort.get_available_providers()
providers = [TELUM_EP, CPU_EP] if TELUM_EP in available else [CPU_EP]

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options=sess_options, providers=providers)
print(f'Active EP: {session.get_providers()[0]}')
```

> On IBM z16 / LinuxONE 4+ with the Telum EP installed, the AIU is used automatically.
> On all other platforms the standard CPU EP is used.

#### 3 · Inspect model inputs and outputs

```python
for inp in session.get_inputs():
    print(f'  {inp.name:<35}  type={inp.type:<16}  shape={inp.shape}')

for out in session.get_outputs():
    print(f'  {out.name:<35}  type={out.type:<16}  shape={out.shape}')
```

The unified ONNX model accepts **24 named inputs**, one tensor per feature, each shaped `[batch, 7, 1]`, and returns a single output `Identity:0` shaped `[batch, 7, 1]`.

#### 4 · Run inference on sample transactions

```python
# Build the feed dict: one ndarray per input, shape [N, 7, 1]
feed = sequences_to_feed(sequences)     # helper defined in the notebook

# Run
output_names = [out.name for out in session.get_outputs()]
raw = session.run(output_names, feed)[0]   # [N, 7, 1]

# Peak fraud probability across all 7 timesteps
fraud_probs = raw[:, :, 0].max(axis=1)

THRESHOLD = 0.5
for i, prob in enumerate(fraud_probs):
    label = '🚨 FRAUD' if prob >= THRESHOLD else '✅ GENUINE'
    print(f'Sequence {i+1}: score={prob:.4f}  →  {label}')
```

#### 5 · Live single-transaction scoring

Maintain a sliding window of the last 6 transactions, append the new transaction, then call the model:

```python
history_window  = last_6_transactions_for_account    # list of 6 rows × 24 features
new_transaction = [month, day_of_month, ..., payment_currency]
live_sequence   = [history_window + [new_transaction]]   # shape [1, 7, 24]

live_prob, _ = run_inference(live_sequence)
print(f'Fraud score: {live_prob[0]:.4f}')
```

---

## Part 3 — Model Serving with Triton Container

**Location**: `zST-model-serving-triton/`

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

Expected layout:
```
model_repository/
└── fraud_detection/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

#### 3 · Build the Docker image

```bash
# IBM z16+ / LinuxONE 4+ (default — uses IBM Z accelerated Triton image):
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
|---|---|---|
| 8000 | HTTP/REST | Inference (`POST /v2/models/fraud_detection/infer`), health, metadata |
| 8001 | gRPC | Inference (gRPC protocol) |
| 8002 | HTTP | Prometheus metrics (`/metrics`) |

#### 5 · Verify server health

```bash
curl http://localhost:8000/v2/health/live
curl http://localhost:8000/v2/health/ready
curl http://localhost:8000/v2/models/fraud_detection
curl http://localhost:8002/metrics
```

#### 6 · Run the test client

```bash
pip install tritonclient[http] numpy
python3 test_triton_client.py [--url localhost:8000]
```

The test suite covers:

| Test | Description |
|---|---|
| 1 · Server health | `is_server_live`, `is_server_ready`, `is_model_ready` |
| 2 · Genuine sequences | G-1 (checking), G-2 (money market) — expects score < 0.5 |
| 3 · Fraud sequences | F-1…F-4 (real APP fraud from CSV, 4 fraudsters) — expects score ≥ 0.5 |
| 4 · Timestep evolution | Per-timestep probabilities for all 6 sequences |
| 5 · Summary table | Accuracy across all 6 labelled sequences |
| 6 · Live scoring | Suspicious late-night Wire appended to genuine G-1 history |
| 7 · Throughput benchmark | 1 000-sequence timed batch |

#### 7 · One-shot build + launch (alternative)

```bash
cd zST-model-serving-triton
bash build_and_deploy.sh
```

The script runs steps 1–6 automatically and prints the endpoints when the server is ready.

#### 8 · IBM Telum AIU acceleration (s390x)

On IBM z16+ / LinuxONE 4+, the `Dockerfile` automatically builds the [IBM ONNX Runtime Telum EP](https://github.com/IBM/onnxruntime-ep-telum) from source and stages `libtelum_plugin_ep.so` at `/opt/telum_ep/`. The `tritonserver` start command registers the library via `--onnxruntime-execution-provider-library` when the file is present.

To also instruct Triton's ONNX Runtime backend to prefer the Telum EP for individual models, add the following `optimization` block to `model_repository/fraud_detection/config.pbtxt`:

```protobuf
optimization {
  execution_accelerators {
    cpu_execution_accelerator [
      {
        name: "TelumPluginExecutionProvider"
      }
    ]
  }
}
```

On x86_64 the build step is skipped automatically (`TARGETARCH` ≠ `s390x`) and the library is absent, so the flag is omitted and the standard CPU EP is used.

#### 9 · Stop / remove the container

```bash
docker stop triton-server && docker rm triton-server
```

### Deployment Architecture

| Component | Description |
|---|---|
| `fraud_detection_unified.onnx` | Unified ONNX model — preprocessing + LSTM + sigmoid output head |
| `config.pbtxt` | Triton model configuration (24 inputs, `Identity:0` output, dynamic batching up to 256) |
| `serve.py` | Sets up the model repository and generates `config.pbtxt` |
| `test_triton_client.py` | HTTP test client using real CSV fraud sequences |
| Triton Inference Server | Production inference platform (HTTP, gRPC, Prometheus) |
| ONNX Runtime backend | High-performance ONNX execution within Triton |
| `libtelum_plugin_ep.so` | Built from source on s390x; staged at `TELUM_EP_LIBRARY_PATH=/opt/telum_ep/` |
| IBM Telum EP | Registered via `--onnxruntime-execution-provider-library` on s390x; skipped on x86_64 |

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms specified in the LICENSE file.

## Authors & Contributors

    - Pui Yen (MVI Tech)
    - Alex Osadchyy (IBM)
    - Erik Altman (IBM)
    - Saurabh Srivastava (IBM)

## Related Resources

- [IBM Z 101](https://www.ibm.com/z/learn/101)
- [Linux on Z AI Solution Templates](https://ambitus.github.io/aionz-solution-templates/)
- [AI Toolkit for IBM Z and LinuxONE](https://ibm.github.io/ai-on-z-101/aitoolkitloz/)
- [IBM ONNX Runtime Telum Execution Provider](https://github.com/IBM/onnxruntime-ep-telum)
- [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
- [ONNX Runtime](https://onnxruntime.ai/)
- [IBM Z AI Solutions](https://www.ibm.com/z/artificial-intelligence)
- [IBM Z Integrated Accelerator for AI](https://www.ibm.com/z/artificial-intelligence)
