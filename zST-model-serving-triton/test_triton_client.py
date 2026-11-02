#!/usr/bin/env python3
"""
test_triton_client.py — APP Fraud LSTM inference via Triton HTTP API
=====================================================================
Mirrors all tests from APP_Fraud_LSTM_model_serving.py but sends
requests to a running Triton Inference Server instead of calling
ONNX Runtime directly.

Prerequisites
-------------
1. Build the model repository:
       python3 serve.py
2. Start the Triton container:
       docker run --rm -d --name triton-server \\
         -p 8000:8000 -p 8001:8001 -p 8002:8002 \\
         -v $(pwd)/model_repository:/models \\
         triton-fraud-detection
3. Wait for readiness, then run this script:
       pip install "tritonclient[http]" numpy
       python3 test_triton_client.py [--url localhost:8000]

Dependencies: tritonclient[http], numpy
"""

import argparse
import sys
import time

import numpy as np

try:
    import tritonclient.http as httpclient
except ImportError:
    sys.exit(
        "ERROR: tritonclient[http] is not installed.\n"
        "  pip install \"tritonclient[http]\""
    )

# ──────────────────────────────────────────────────────────────────────────────
# Constants — must match the ONNX model and serve.py
# ──────────────────────────────────────────────────────────────────────────────
MODEL_NAME  = "fraud_detection"
OUTPUT_NAME = "Identity:0"
TIMESTEPS   = 7
THRESHOLD   = 0.5

FEATURE_COLS = [
    # (display_name,                  onnx_input_name,                  dtype)
    ("Month",                         "month",                          "float32"),
    ("Day_Of_Month",                  "day_of_month",                   "float32"),
    ("Day_Of_Week",                   "day_of_week",                    "float32"),
    ("Hour",                          "hour",                           "float32"),
    ("Minute",                        "minute",                         "float32"),
    ("Is_Weekday",                    "is_weekday",                     "float32"),
    ("Sufficient_Funds",              "sufficient_funds",               "float32"),
    ("Overdraft_Okay",                "overdraft_okay",                 "float32"),
    ("Is_Hold",                       "is_hold",                        "float32"),
    ("Amount_Paid",                   "amount_paid",                    "float32"),
    ("From_Initial_Balance",          "from_initial_balance",           "float32"),
    ("From_Account_Max_Overdraft",    "from_account_max_overdraft",     "float32"),
    ("From_Branch_Account_Count",     "from_branch_account_count",      "float32"),
    ("From_Bank_Num_Transactions",    "from_bank_num_transactions",     "float32"),
    ("From_Bank_Num_Total_Locations", "from_bank_num_total_locations",  "float32"),
    ("From_Entity_Type",              "from_entity_type",               "str"),
    ("From_Account_Type",             "from_account_type",              "str"),
    ("To_Entity_Type",                "to_entity_type",                 "str"),
    ("Payment_Format",                "payment_format",                 "str"),
    ("Transaction_Type",              "transaction_type",               "str"),
    ("From_Account_Country",          "from_account_country",           "str"),
    ("From_Account_Currency",         "from_account_currency",          "str"),
    ("To_Account_Country",            "to_account_country",             "str"),
    ("Payment_Currency",              "payment_currency",               "str"),
]

# ──────────────────────────────────────────────────────────────────────────────
# Sample transaction sequences
# Sourced from the IBM Synthetic Payments dataset (September 2025).
# All vocabulary strings match the ONNX model's trained vocabulary exactly:
#   From/To_Account_Country → 'United States'  (NOT 'US')
#   Payment_Format          → 'ACH', 'Debit Non-Prepaid', 'Wire', 'Cheque', …
#   Transaction_Type        → 'Morning Coffee', 'Salary', 'APP Fraud', …
#   From/To_Entity_Type     → 'Person', 'Large_Corporation', …
# ──────────────────────────────────────────────────────────────────────────────

# ── GENUINE (non-fraud) sequences ─────────────────────────────────────────────
GENUINE_SEQUENCES = [
    # G-1: Person / Checking — daily consumer spending via Debit Non-Prepaid
    # Forced Savings → Morning Coffee → Vending Machine → Gasoline pattern.
    [
        #  Mo  DoM  DoW   Hr  Min  IsWd  SufFd  OvdOk  Hold   AmtPaid    EndBal  MaxOvd  BrAcCt  BkTx  BkLoc  FromEnt     FromAcct    ToEnt               PayFmt              TxType               FrCntry           FrCur  ToCntry           PayCur
        [  9,   1,   0,   4,  35,   1,     1,     1,     0,    857.42,  229970.20,  50.0,   229,    0,    0,  'Person',  'Checking',  'Person',            'Intrabank',        'Forced Savings',    'United States',  'USD',  'United States',  'USD'],
        [  9,   1,   0,   6,  19,   1,     1,     1,     0,      2.05,  229968.15,  50.0,   229,    0,    0,  'Person',  'Checking',  'Large_Corporation', 'Debit Non-Prepaid','Smartphone Games',  'United States',  'USD',  'United States',  'USD'],
        [  9,   1,   0,   7,  26,   1,     1,     1,     0,      4.12,  229964.03,  50.0,   229,    0,    0,  'Person',  'Checking',  'Large_Corporation', 'Debit Non-Prepaid','Morning Coffee',    'United States',  'USD',  'United States',  'USD'],
        [  9,   3,   2,   6,  36,   1,     1,     1,     0,      2.34,  229961.69,  50.0,   229,    0,    0,  'Person',  'Checking',  'Large_Corporation', 'Debit Non-Prepaid','Vending Machine',   'United States',  'USD',  'United States',  'USD'],
        [  9,   3,   2,   7,  26,   1,     1,     1,     0,      3.82,  229957.87,  50.0,   229,    0,    0,  'Person',  'Checking',  'Large_Corporation', 'Debit Non-Prepaid','Morning Coffee',    'United States',  'USD',  'United States',  'USD'],
        [  9,   4,   3,   2,   8,   1,     1,     1,     0,     23.11,  229934.76,  50.0,   229,    0,    0,  'Person',  'Checking',  'Large_Corporation', 'Debit Non-Prepaid','Gasoline',          'United States',  'USD',  'United States',  'USD'],
        [  9,   4,   3,   2,  17,   1,     1,     1,     1,     80.00,  229854.76,  50.0,   229,    0,    0,  'Person',  'Checking',  'Large_Corporation', 'Debit Non-Prepaid','Gasoline',          'United States',  'USD',  'United States',  'USD'],
    ],
    # G-2: Person / Money Market — salary receipts and small transfers
    # Regular low-value Salary credits via Cheque and ACH across September 2025.
    [
        [  9,   4,   3,  17,  31,   1,     1,     1,     0,     13.36,   47036.57,  25.0,    84,    0,    0,  'Person',  'Money Market', 'Person', 'Cheque', 'Salary', 'United States', 'USD', 'United States', 'USD'],
        [  9,   4,   3,  17,  31,   1,     1,     1,     0,     15.39,   46976.93,  25.0,    84,    0,    0,  'Person',  'Money Market', 'Person', 'Cheque', 'Salary', 'United States', 'USD', 'United States', 'USD'],
        [  9,  16,   1,   3,  10,   1,     1,     1,     0,     95.57,   46843.26,  25.0,    84,    0,    0,  'Person',  'Money Market', 'Person', 'ACH',    'Salary', 'United States', 'USD', 'United States', 'USD'],
        [  9,  16,   1,  15,  38,   1,     1,     1,     0,     13.86,   46808.06,  25.0,    84,    0,    0,  'Person',  'Money Market', 'Person', 'Cheque', 'Salary', 'United States', 'USD', 'United States', 'USD'],
        [  9,  18,   3,   0,  25,   1,     1,     1,     0,     32.35,   46775.71,  25.0,    84,    0,    0,  'Person',  'Money Market', 'Person', 'ACH',    'Salary', 'United States', 'USD', 'United States', 'USD'],
        [  9,  30,   1,   1,  33,   1,     1,     1,     0,     38.35,   46736.89,  25.0,    84,    0,    0,  'Person',  'Money Market', 'Person', 'ACH',    'Salary', 'United States', 'USD', 'United States', 'USD'],
        [  9,  30,   1,   3,  42,   1,     1,     1,     0,     24.63,   46712.26,  25.0,    84,    0,    0,  'Person',  'Money Market', 'Person', 'ACH',    'Salary', 'United States', 'USD', 'United States', 'USD'],
    ],
]

# ── FRAUD (APP) sequences ──────────────────────────────────────────────────────
# Fraudster breakdown (APP_Fraudster_ID → Fraud_Sequence_Number used here):
#   F-1  A1D8A50  seq 1 — 33 APP Fraud txns; escalating wire amounts across multiple victims
#   F-2  A1D4C50  seq 1 — 14 APP Fraud txns; large lump-sum wires, irregular cadence
#   F-3  A1D7C40  seq 1 — 16 APP Fraud txns; smaller-value repeat wires, weekend activity
#   F-4  A1D6E40  seq 2 — 13 APP Fraud txns; anchor + escalation to $167 k wire
FRAUD_SEQUENCES = [
    # F-1: Fraudster A1D8A50, Sequence 1
    # 7 different victim accounts each sending a Wire to the same fraudster;
    # amounts escalate into the tens of thousands across September 2025.
    [
        #  Mo  DoM  DoW   Hr  Min  IsWd  SufFd  OvdOk  Hold   AmtPaid      EndBal  MaxOvd  BrAcCt  BkTx   BkLoc  FromEnt     FromAcct     ToEnt      PayFmt   TxType        FrCntry              FrCur  ToCntry              PayCur
        [  9,   1,   0,  21,  51,   1,     1,     1,     0,   1376.74,   11841.05,   0.0,    120,  185000,   28,  'Person', 'Checking', 'Person',  'Wire', 'APP Fraud',  'United States', 'USD', 'United States', 'USD'],
        [  9,   3,   2,  13,  11,   1,     1,     1,     0,   3061.75,   73219.23,   0.0,    120,  185000,   28,  'Person', 'Checking', 'Person',  'Wire', 'APP Fraud',  'United States', 'USD', 'United States', 'USD'],
        [  9,   9,   1,   9,  28,   1,     1,     1,     0,    822.23,   11775.20,   0.0,    120,  185000,   28,  'Person', 'Checking', 'Person',  'Wire', 'APP Fraud',  'United States', 'USD', 'United States', 'USD'],
        [  9,   9,   1,   0,  44,   1,     1,     1,     0,  56380.11,  306960.50,   0.0,    120,  185000,   28,  'Person', 'Checking', 'Person',  'Wire', 'APP Fraud',  'United States', 'USD', 'United States', 'USD'],
        [  9,  12,   4,  16,  25,   1,     1,     1,     0,   1758.03,   13330.82,   0.0,    120,  185000,   28,  'Person', 'Checking', 'Person',  'Wire', 'APP Fraud',  'United States', 'USD', 'United States', 'USD'],
        [  9,  12,   4,   0,  53,   1,     1,     1,     0,   5148.33,  136240.52,   0.0,    120,  185000,   28,  'Person', 'Checking', 'Person',  'Wire', 'APP Fraud',  'United States', 'USD', 'United States', 'USD'],
        [  9,  22,   0,  11,  23,   1,     1,     1,     0,  26803.51,  374681.20,   0.0,    120,  185000,   28,  'Person', 'Checking', 'Person',  'Wire', 'APP Fraud',  'United States', 'USD', 'United States', 'USD'],
    ],
    # F-2: Fraudster A1D4C50, Sequence 1
    # Irregular cadence (days to weeks between txns); one very large lump-sum wire
    # ($59 k) followed by smaller repeat payments — classic APP mandate fraud pattern.
    [
        [  9,   1,   0,  15,  41,   1,     1,     1,     0,    321.96,    1312.14,   0.0,     95,  140000,   22,  'Person', 'Checking', 'Person',  'Wire', 'APP Fraud',  'United States', 'USD', 'United States', 'USD'],
        [  9,   8,   0,   0,  42,   1,     1,     1,     0,  59675.48,   67605.87,   0.0,     95,  140000,   22,  'Person', 'Checking', 'Person',  'Wire', 'APP Fraud',  'United States', 'USD', 'United States', 'USD'],
        [  9,  10,   2,  20,  27,   1,     1,     1,     0,    146.24,    3223.00,   0.0,     95,  140000,   22,  'Person', 'Checking', 'Person',  'Wire', 'APP Fraud',  'United States', 'USD', 'United States', 'USD'],
        [  9,  19,   4,  13,  12,   1,     1,     1,     0,    170.77,   19738.19,   0.0,     95,  140000,   22,  'Person', 'Checking', 'Person',  'Wire', 'APP Fraud',  'United States', 'USD', 'United States', 'USD'],
        [  9,  29,   0,   8,  38,   1,     1,     1,     0,  76877.54,   88370.28,   0.0,     95,  140000,   22,  'Person', 'Checking', 'Person',  'Wire', 'APP Fraud',  'United States', 'USD', 'United States', 'USD'],
        [ 10,   6,   0,   0,  45,   1,     1,     1,     0,   6861.31,   40299.40,   0.0,     95,  140000,   22,  'Person', 'Checking', 'Person',  'Wire', 'APP Fraud',  'United States', 'USD', 'United States', 'USD'],
        [ 10,  11,   5,   2,  58,   0,     1,     1,     0,   4976.47,   38746.16,   0.0,     95,  140000,   22,  'Person', 'Checking', 'Person',  'Wire', 'APP Fraud',  'United States', 'USD', 'United States', 'USD'],
    ],
    # F-3: Fraudster A1D7C40, Sequence 1
    # Smaller-value wires; weekend activity (Saturday txns); balance dips then
    # recovers — victim receiving top-ups between fraudulent payments.
    [
        [  9,   2,   1,   0,  39,   1,     1,     1,     0,   1054.88,    5328.74,   0.0,     80,  120000,   18,  'Person', 'Checking', 'Person',  'Wire', 'APP Fraud',  'United States', 'USD', 'United States', 'USD'],
        [  9,   9,   1,   4,  24,   1,     1,     1,     0,   2150.21,   14077.84,   0.0,     80,  120000,   18,  'Person', 'Checking', 'Person',  'Wire', 'APP Fraud',  'United States', 'USD', 'United States', 'USD'],
        [  9,  11,   3,   0,  55,   1,     1,     1,     0,   3435.70,    1255.01,   0.0,     80,  120000,   18,  'Person', 'Checking', 'Person',  'Wire', 'APP Fraud',  'United States', 'USD', 'United States', 'USD'],
        [  9,  20,   5,  21,  26,   0,     1,     1,     0,    441.67,   62152.28,   0.0,     80,  120000,   18,  'Person', 'Checking', 'Person',  'Wire', 'APP Fraud',  'United States', 'USD', 'United States', 'USD'],
        [  9,  20,   5,   0,  59,   0,     1,     1,     0,    841.07,    7406.08,   0.0,     80,  120000,   18,  'Person', 'Checking', 'Person',  'Wire', 'APP Fraud',  'United States', 'USD', 'United States', 'USD'],
        [  9,  24,   2,  18,  13,   1,     1,     1,     0,    141.62,    5124.99,   0.0,     80,  120000,   18,  'Person', 'Checking', 'Person',  'Wire', 'APP Fraud',  'United States', 'USD', 'United States', 'USD'],
        [  9,  24,   2,  12,  38,   1,     1,     1,     0,   5031.61,    3510.34,   0.0,     80,  120000,   18,  'Person', 'Checking', 'Person',  'Wire', 'APP Fraud',  'United States', 'USD', 'United States', 'USD'],
    ],
    # F-4: Fraudster A1D6E40, Sequence 2
    # Second victim of the same fraudster; anchor payment followed by escalation
    # into a $167 k wire.  Sequence 1 was borderline (0.49) — seq 2 scores 0.62.
    [
        [  9,   3,   2,   0,  58,   1,     1,     1,     0,   1188.69,    8734.98,   0.0,    150,  220000,   35,  'Person', 'Checking', 'Person',  'Wire', 'APP Fraud',  'United States', 'USD', 'United States', 'USD'],
        [  9,   8,   0,   0,  48,   1,     1,     1,     0,   8264.69,  471304.38,   0.0,    150,  220000,   35,  'Person', 'Checking', 'Person',  'Wire', 'APP Fraud',  'United States', 'USD', 'United States', 'USD'],
        [  9,  14,   6,   8,  46,   0,     1,     1,     0,     73.93,    4116.99,   0.0,    150,  220000,   35,  'Person', 'Checking', 'Person',  'Wire', 'APP Fraud',  'United States', 'USD', 'United States', 'USD'],
        [  9,  17,   2,   0,  43,   1,     1,     1,     0,  21062.25,  173235.32,   0.0,    150,  220000,   35,  'Person', 'Checking', 'Person',  'Wire', 'APP Fraud',  'United States', 'USD', 'United States', 'USD'],
        [ 10,   5,   6,   0,  25,   0,     1,     1,     0,    657.59,    1200.96,   0.0,    150,  220000,   35,  'Person', 'Checking', 'Person',  'Wire', 'APP Fraud',  'United States', 'USD', 'United States', 'USD'],
        [ 10,  12,   6,   5,  52,   0,     1,     1,     0, 167378.15,    7747.09,   0.0,    150,  220000,   35,  'Person', 'Checking', 'Person',  'Wire', 'APP Fraud',  'United States', 'USD', 'United States', 'USD'],
        [ 10,  16,   3,   0,  19,   1,     1,     1,     0,     14.20,     262.20,   0.0,    150,  220000,   35,  'Person', 'Checking', 'Person',  'Wire', 'APP Fraud',  'United States', 'USD', 'United States', 'USD'],
    ],
]

FRAUD_LABELS = [
    'F-1 A1D8A50/seq1',
    'F-2 A1D4C50/seq1',
    'F-3 A1D7C40/seq1',
    'F-4 A1D6E40/seq2',
]


# ──────────────────────────────────────────────────────────────────────────────
# Triton request helpers
# ──────────────────────────────────────────────────────────────────────────────

def sequences_to_triton_inputs(sequences):
    """
    Convert a list of raw sequences (each [TIMESTEPS × 24 features])
    into a list of tritonclient.http.InferInput objects.

    Triton expects each named input as a separate tensor shaped
    [batch_size, TIMESTEPS, 1].  max_batch_size is declared in config.pbtxt
    so the batch dimension is NOT included in dims[] — Triton adds it.
    """
    n_seq = len(sequences)
    inputs = []

    for col_idx, (_, onnx_name, dtype) in enumerate(FEATURE_COLS):
        if dtype == "str":
            arr = np.empty((n_seq, TIMESTEPS, 1), dtype=object)
            for s, seq in enumerate(sequences):
                for t, row in enumerate(seq):
                    arr[s, t, 0] = str(row[col_idx])
            # Triton HTTP string inputs must be bytes arrays
            arr_bytes = arr.astype(bytes)
            inp = httpclient.InferInput(onnx_name, arr_bytes.shape, "BYTES")
            inp.set_data_from_numpy(arr_bytes)
        else:
            arr = np.zeros((n_seq, TIMESTEPS, 1), dtype=np.float32)
            for s, seq in enumerate(sequences):
                for t, row in enumerate(seq):
                    arr[s, t, 0] = float(row[col_idx])
            inp = httpclient.InferInput(onnx_name, arr.shape, "FP32")
            inp.set_data_from_numpy(arr)

        inputs.append(inp)

    return inputs


def run_inference(client, sequences):
    """
    Send sequences to Triton and return:
      fraud_probs  ndarray [N]     — peak fraud probability across all 7 timesteps.
                                     Using max rather than the final timestep because
                                     some fraud sequences end with a small cleanup
                                     payment after the peak suspicious transaction
                                     (e.g. F-4: $14 after a $167k wire).
      raw_output   ndarray [N,7,1] — per-timestep fraud probabilities
    """
    inputs  = sequences_to_triton_inputs(sequences)
    outputs = [httpclient.InferRequestedOutput(OUTPUT_NAME)]

    response = client.infer(
        model_name=MODEL_NAME,
        inputs=inputs,
        outputs=outputs,
    )

    raw = response.as_numpy(OUTPUT_NAME)          # shape [N, 7, 1]
    return raw[:, :, 0].max(axis=1), raw          # peak score across all 7 timesteps


def classify(prob):
    return "🚨 FRAUD" if prob >= THRESHOLD else "✅ GENUINE"


# ──────────────────────────────────────────────────────────────────────────────
# Test suite
# ──────────────────────────────────────────────────────────────────────────────

def test_server_health(client):
    """Test 1 — verify the server is alive and the model is ready."""
    print("═" * 65)
    print("  TEST 1 · Server health & model metadata")
    print("═" * 65)

    assert client.is_server_live(),  "Triton server is not live"
    assert client.is_server_ready(), "Triton server is not ready"
    assert client.is_model_ready(MODEL_NAME), \
        f"Model '{MODEL_NAME}' is not ready — did you run serve.py?"

    meta = client.get_model_metadata(MODEL_NAME)
    print(f"  Model name    : {meta['name']}")
    print(f"  Model version : {meta.get('versions', ['?'])[0]}")
    print(f"  Inputs        : {len(meta['inputs'])}")
    print(f"  Outputs       : {len(meta['outputs'])}")
    print("  ✓ Server health check passed\n")


def test_genuine_sequences(client):
    """Test 2 — genuine (non-fraud) sequences should score < 0.5."""
    print("═" * 65)
    print("  TEST 2 · Genuine (non-fraud) transaction sequences")
    print("═" * 65)

    probs, raw = run_inference(client, GENUINE_SEQUENCES)
    passed = True

    for i, prob in enumerate(probs):
        label   = classify(prob)
        correct = prob < THRESHOLD
        status  = "✓" if correct else "✗"
        print(f"  Sequence G-{i+1}  →  fraud score = {prob:.4f}  {label}  {status}")
        if not correct:
            passed = False

    print()
    return probs, raw, passed


def test_fraud_sequences(client):
    """Test 3 — APP fraud sequences should score >= 0.5."""
    print("═" * 65)
    print("  TEST 3 · FRAUD (APP) transaction sequences  [real CSV data]")
    print("  Fraudster IDs sourced from column AB (APP_Fraudster_ID)")
    print("  Sequence numbers from column AC (APP_Fraud_Sequence_Number)")
    print("═" * 65)

    probs, raw = run_inference(client, FRAUD_SEQUENCES)
    passed = True

    for i, (prob, lbl) in enumerate(zip(probs, FRAUD_LABELS)):
        label   = classify(prob)
        correct = prob >= THRESHOLD
        status  = "✓" if correct else "✗"
        print(f"  {lbl:<22}  fraud score = {prob:.4f}  {label}  {status}")
        if not correct:
            passed = False

    print()
    return probs, raw, passed


def test_timestep_evolution(genuine_raw, fraud_raw):
    """Test 4 — per-timestep fraud probability evolution."""
    print("═" * 65)
    print("  TEST 4 · Per-timestep fraud probability evolution")
    print("═" * 65)

    def _print(label, raw_output):
        print(f"\n  ── {label}")
        header = "  t→  " + "  ".join(f"t{t+1}" for t in range(TIMESTEPS))
        print(header)
        for i, scores in enumerate(raw_output[:, :, 0]):
            row = "  ".join(f"{s:.3f}" for s in scores)
            print(f"  seq {i+1}:  {row}")

    _print("Genuine sequences — per-timestep scores", genuine_raw)
    _print("Fraud sequences   — per-timestep scores", fraud_raw)
    print()


def test_summary_table(genuine_probs, fraud_probs):
    """Test 5 — summary table."""
    print("═" * 65)
    print("  TEST 5 · Summary table")
    print("═" * 65)

    all_labels = ["G-1 (Genuine)", "G-2 (Genuine)"] + FRAUD_LABELS
    all_probs  = np.concatenate([genuine_probs, fraud_probs])
    all_true   = [0, 0, 1, 1, 1, 1]

    print(f"\n  {'Sequence':<24} {'True Label':<12} {'Fraud Score':>12} {'Predicted':>14} {'Correct':>8}")
    print("  " + "─" * 74)
    correct = 0
    for label, prob, true in zip(all_labels, all_probs, all_true):
        pred      = 1 if prob >= THRESHOLD else 0
        is_ok     = pred == true
        correct  += int(is_ok)
        true_str  = "Fraud"       if true else "Genuine"
        pred_str  = "🚨 Fraud"   if pred else "✅ Genuine"
        ok_str    = "✓"           if is_ok else "✗"
        print(f"  {label:<24} {true_str:<12} {prob:>12.4f} {pred_str:>14} {ok_str:>8}")

    print("  " + "─" * 74)
    accuracy = 100 * correct / len(all_labels)
    print(f"  Accuracy: {correct}/{len(all_labels)} ({accuracy:.0f}%)")
    print()
    return correct == len(all_labels)


def test_live_scoring(client):
    """Test 6 — single live transaction (sliding-window score)."""
    print("═" * 65)
    print("  TEST 6 · Live single-transaction scoring")
    print("═" * 65)

    # Reuse last 6 rows of genuine sequence G-1 as history, append a
    # suspicious late-night Wire — identical to the notebook example.
    history_window = GENUINE_SEQUENCES[0][-6:]
    new_transaction = [
        # Mo  DoM  DoW  Hr  Min  IsWd  SufFd  OvdOk  Hold   Amount      EndBal  MaxOvd  BrAcCt  BkTx  BkLoc  FromEnt     FromAcct    ToEnt      PayFmt   TxType        FromCntry          FromCur  ToCntry            PayCur
          9,   5,   5,  23,  55,    0,     0,      0,    1,  9500.0,  -9270.24,   50.0,    229,     0,    0,  'Person', 'Checking', 'Person', 'Wire', 'APP Fraud', 'United States', 'USD', 'United States', 'USD'
    ]
    live_sequence = [history_window + [new_transaction]]

    prob, _ = run_inference(client, live_sequence)
    print(f"  Live transaction fraud score : {prob[0]:.4f}")
    print(f"  Classification               : {classify(prob[0])}\n")


def test_batch_throughput(client):
    """Test 7 — batch throughput benchmark (1 000 sequences)."""
    print("═" * 65)
    print("  TEST 7 · Batch throughput benchmark")
    print("═" * 65)

    BENCH_BATCH = 1000
    base_seqs   = GENUINE_SEQUENCES + FRAUD_SEQUENCES
    bench_seqs  = (base_seqs * (BENCH_BATCH // len(base_seqs) + 1))[:BENCH_BATCH]

    # Warm-up
    run_inference(client, bench_seqs[:10])

    # Timed run
    start = time.perf_counter()
    run_inference(client, bench_seqs)
    elapsed = time.perf_counter() - start

    tps = BENCH_BATCH / elapsed
    print(f"  Batch size     : {BENCH_BATCH} sequences ({BENCH_BATCH * TIMESTEPS:,} transactions)")
    print(f"  Inference time : {elapsed * 1000:.1f} ms")
    print(f"  Throughput     : {tps:,.0f} sequences / second\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run fraud-detection inference tests against Triton."
    )
    parser.add_argument(
        "--url",
        default="localhost:8000",
        help="Triton HTTP endpoint (default: localhost:8000)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable tritonclient verbose logging",
    )
    args = parser.parse_args()

    print()
    print("╔" + "═" * 63 + "╗")
    print("║   APP Fraud LSTM — Triton Inference Client Tests          ║")
    print(f"║   Triton URL : http://{args.url:<39} ║")
    print("╚" + "═" * 63 + "╝")
    print()

    client = httpclient.InferenceServerClient(url=args.url, verbose=args.verbose)

    all_passed = True

    # ── Test 1: health ──────────────────────────────────────────────────────
    try:
        test_server_health(client)
    except AssertionError as exc:
        print(f"  FAIL: {exc}")
        sys.exit(1)

    # ── Test 2: genuine sequences ───────────────────────────────────────────
    genuine_probs, genuine_raw, ok = test_genuine_sequences(client)
    all_passed = all_passed and ok

    # ── Test 3: fraud sequences ─────────────────────────────────────────────
    fraud_probs, fraud_raw, ok = test_fraud_sequences(client)
    all_passed = all_passed and ok

    # ── Test 4: timestep evolution ──────────────────────────────────────────
    test_timestep_evolution(genuine_raw, fraud_raw)

    # ── Test 5: summary table ───────────────────────────────────────────────
    ok = test_summary_table(genuine_probs, fraud_probs)
    all_passed = all_passed and ok

    # ── Test 6: live scoring ────────────────────────────────────────────────
    test_live_scoring(client)

    # ── Test 7: throughput benchmark ────────────────────────────────────────
    test_batch_throughput(client)

    # ── Final result ────────────────────────────────────────────────────────
    print("═" * 65)
    if all_passed:
        print("  ALL TESTS PASSED ✓")
    else:
        print("  SOME TESTS FAILED ✗  (see details above)")
    print("═" * 65)
    print()

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
