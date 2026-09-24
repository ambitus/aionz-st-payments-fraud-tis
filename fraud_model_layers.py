"""
fraud_model_layers.py
=====================
Reusable, generic building blocks not specific to this APP fraud solution template but 
rather common functions to address onnx export and onnx preprocessing parts.

Exports
-------
Keras layers (all registered for serialisation):
    OnnxVocabOneHot     — String → float32 one-hot  (ONNX-compatible StringLookup replacement)
    OnnxVocabOrdinal    — String → float32 ordinal  (ONNX-compatible StringLookup replacement)
    CyclicalEncoding    — periodic feature → (sin, cos) pair
    LogTransform        — sign-preserving log1p transform
    TimeOfDayEncoding   — (hour, minute) → (sin, cos) time-of-day pair

Training helpers:
    focal_loss(gamma, alpha)  — focal loss closure for use with model.compile()
    TP / FP / FN / TN         — Keras metric wrappers that slice to the final timestep

ONNX post-processing:
    fix_onnx_export(model_proto, preprocessing_model, onnx_output_path)
        Fixes two tf2onnx export bugs and saves the corrected proto in-place.
"""

import numpy as np
import tensorflow as tf
import keras
from keras import layers, ops
import onnx
from onnx import numpy_helper, TensorProto


# ── Custom Keras preprocessing layers ────────────────────────────────────────

@keras.saving.register_keras_serializable()
class OnnxVocabOneHot(layers.Layer):
    """String → float32 one-hot (replaces StringLookup output_mode='one_hot').

    ONNX-compatible: uses only tf.equal / tf.cast — opset-13 primitives.
    """

    def __init__(self, vocab_size=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._vocab = None
        self._vocab_size = vocab_size

    def compute_output_spec(self, inputs):
        return keras.KerasTensor(shape=(inputs.shape[0], self._vocab_size), dtype='float32')

    def adapt(self, data):
        vals = sorted(tf.unique(tf.reshape(data, [-1]))[0].numpy().tolist(), key=lambda b: b.decode())
        if b'Unknown' in vals:
            vals.remove(b'Unknown')
            vals = [b'Unknown'] + vals
        self._vocab = tf.constant(vals, dtype=tf.string)
        self._vocab_size = len(vals)

    def set_vocabulary(self, vocab):
        vocab = [v.encode() if isinstance(v, str) else v for v in vocab]
        self._vocab = tf.constant(vocab, dtype=tf.string)
        self._vocab_size = len(vocab)

    def call(self, inputs):
        return tf.cast(tf.squeeze(tf.equal(inputs[:, :, tf.newaxis], self._vocab), axis=1), tf.float32)

    def get_config(self):
        cfg = super().get_config()
        if self._vocab is not None:
            cfg['vocabulary'] = [v.decode() for v in self._vocab.numpy().tolist()]
        return cfg

    @classmethod
    def from_config(cls, config):
        vocab = config.pop('vocabulary', None)
        obj = cls(**config)
        if vocab is not None:
            obj.set_vocabulary(vocab)
        return obj


@keras.saving.register_keras_serializable()
class OnnxVocabOrdinal(layers.Layer):
    """String → float32 ordinal index (replaces StringLookup output_mode='int').

    ONNX-compatible: uses only tf.equal / tf.cast — opset-13 primitives.
    """

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._vocab = None

    def compute_output_spec(self, inputs):
        return keras.KerasTensor(shape=(inputs.shape[0], 1), dtype='float32')

    def adapt(self, data):
        vals = sorted(tf.unique(tf.reshape(data, [-1]))[0].numpy().tolist(), key=lambda b: b.decode())
        if b'Unknown' in vals:
            vals.remove(b'Unknown')
            vals = [b'Unknown'] + vals
        self._vocab = tf.constant(vals, dtype=tf.string)

    def set_vocabulary(self, vocab):
        vocab = [v.encode() if isinstance(v, str) else v for v in vocab]
        self._vocab = tf.constant(vocab, dtype=tf.string)

    def call(self, inputs):
        matches = tf.squeeze(tf.equal(inputs[:, :, tf.newaxis], self._vocab), axis=1)
        return tf.cast(tf.expand_dims(tf.argmax(tf.cast(matches, tf.int32), axis=-1), -1), tf.float32)

    def get_config(self):
        cfg = super().get_config()
        if self._vocab is not None:
            cfg['vocabulary'] = [v.decode() for v in self._vocab.numpy().tolist()]
        return cfg

    @classmethod
    def from_config(cls, config):
        vocab = config.pop('vocabulary', None)
        obj = cls(**config)
        if vocab is not None:
            obj.set_vocabulary(vocab)
        return obj


@keras.saving.register_keras_serializable()
class CyclicalEncoding(layers.Layer):
    """Encodes a periodic feature as (sin, cos) pair."""
    def __init__(self, max_value, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_value = float(max_value)
    def call(self, inputs):
        angle = 2 * np.pi * inputs / self.max_value
        return ops.concatenate([ops.sin(angle), ops.cos(angle)], axis=-1)
    def get_config(self):
        return {**super().get_config(), "max_value": self.max_value}


@keras.saving.register_keras_serializable()
class LogTransform(layers.Layer):
    """Sign-preserving log1p transform."""
    def call(self, inputs):
        return ops.sign(inputs) * ops.log(ops.abs(inputs) + 1.0)
    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable()
class TimeOfDayEncoding(layers.Layer):
    """Encodes (hour, minute) as a (sin, cos) time-of-day pair."""
    def call(self, inputs):
        seconds = inputs[:, 0:1] * 3600 + inputs[:, 1:2] * 60
        angle   = 2 * np.pi * seconds / 86400
        return ops.concatenate([ops.sin(angle), ops.cos(angle)], axis=-1)
    def get_config(self):
        return super().get_config()


# ── Training helpers ──────────────────────────────────────────────────────────
# A lower threshold is needed to register any TP/FP during training.  Tune alongside 
# CLASS_WEIGHT_CAP (in the training script) to find the ight precision/recall balance.
EVAL_THRESHOLD = 0.2

# ── Focal loss ────────────────────────────────────────────────────────────────
# Binary cross-entropy with sample_weight on an imbalanced dataset drives the
# model into a base-rate attractor.
def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss closure for use with model.compile(loss=focal_loss())."""
    def loss_fn(y_true, y_pred):
        eps     = keras.backend.epsilon()
        y_pred  = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        ce      = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, gamma) * ce)
    return loss_fn


# Custom metrics — evaluate only the last (most recent) timestep.
# All confusion-matrix metrics use EVAL_THRESHOLD (not the Keras default of 0.5)
# because focal loss compresses fraud scores well below 0.5.

class TP(keras.metrics.TruePositives):
    def __init__(self, **kwargs): super().__init__(thresholds=EVAL_THRESHOLD, **kwargs)
    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true[:,-1,:], y_pred[:,-1,:], sample_weight)

class FP(keras.metrics.FalsePositives):
    def __init__(self, **kwargs): super().__init__(thresholds=EVAL_THRESHOLD, **kwargs)
    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true[:,-1,:], y_pred[:,-1,:], sample_weight)

class FN(keras.metrics.FalseNegatives):
    def __init__(self, **kwargs): super().__init__(thresholds=EVAL_THRESHOLD, **kwargs)
    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true[:,-1,:], y_pred[:,-1,:], sample_weight)

class TN(keras.metrics.TrueNegatives):
    def __init__(self, **kwargs): super().__init__(thresholds=EVAL_THRESHOLD, **kwargs)
    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true[:,-1,:], y_pred[:,-1,:], sample_weight)

class F1Score(keras.metrics.Metric):
    """F1 at the last timestep using EVAL_THRESHOLD — balances precision and recall.

    Maximising F1 prevents the model from trading false negatives for false
    positives (or vice versa) as epochs accumulate on an imbalanced dataset.
    """
    def __init__(self, name='F1', **kwargs):
        super().__init__(name=name, **kwargs)
        self._tp = keras.metrics.TruePositives(thresholds=EVAL_THRESHOLD)
        self._fp = keras.metrics.FalsePositives(thresholds=EVAL_THRESHOLD)
        self._fn = keras.metrics.FalseNegatives(thresholds=EVAL_THRESHOLD)

    def update_state(self, y_true, y_pred, sample_weight=None):
        yt = y_true[:, -1, :]
        yp = y_pred[:, -1, :]
        self._tp.update_state(yt, tf.cast(yp >= EVAL_THRESHOLD, tf.float32), sample_weight)
        self._fp.update_state(yt, tf.cast(yp >= EVAL_THRESHOLD, tf.float32), sample_weight)
        self._fn.update_state(yt, tf.cast(yp >= EVAL_THRESHOLD, tf.float32), sample_weight)

    def result(self):
        tp = self._tp.result()
        fp = self._fp.result()
        fn = self._fn.result()
        precision = tp / (tp + fp + keras.backend.epsilon())
        recall    = tp / (tp + fn + keras.backend.epsilon())
        return 2 * precision * recall / (precision + recall + keras.backend.epsilon())

    def reset_state(self):
        self._tp.reset_state()
        self._fp.reset_state()
        self._fn.reset_state()


# ── ONNX post-processing ──────────────────────────────────────────────────────

def fix_onnx_export(model_proto, preprocessing_model, onnx_output_path):
    """Patch tf2onnx export to embed preprocessing layers correctly along with the model.
    e.g. Leaked preprocessing constants and invalid cast(STRING→INT32) in vocab lookup subgraphs.
    """
    # 1. Collect vocab and norm stats from the Keras preprocessing layers
    _layer_vocab, _layer_is_onehot, _layer_norm = {}, {}, {}
    for layer in preprocessing_model.layers:
        if isinstance(layer, OnnxVocabOneHot) and layer._vocab is not None:
            _layer_vocab[layer.name]     = [v.decode() for v in layer._vocab.numpy()]
            _layer_is_onehot[layer.name] = True
        elif isinstance(layer, OnnxVocabOrdinal) and layer._vocab is not None:
            _layer_vocab[layer.name]     = [v.decode() for v in layer._vocab.numpy()]
            _layer_is_onehot[layer.name] = False
        elif isinstance(layer, layers.Normalization):
            weights = {w.name.split('/')[-1]: w.numpy() for w in layer.weights}
            mk = next((k for k in weights if 'mean'     in k), None)
            vk = next((k for k in weights if 'variance' in k), None)
            if mk and vk:
                _layer_norm[layer.name] = (weights[mk].reshape(1,-1).astype(np.float32),
                                           weights[vk].reshape(1,-1).astype(np.float32))

    # 2. Reload proto and ensure ai.onnx.ml opset is declared
    model_proto = onnx.load(onnx_output_path)
    graph = model_proto.graph
    if not any(op.domain == 'ai.onnx.ml' for op in model_proto.opset_import):
        oi = model_proto.opset_import.add(); oi.domain = 'ai.onnx.ml'; oi.version = 3

    # 3. Replace broken vocab lookup subgraphs with LabelEncoder + OHE/Cast
    out_to_node        = {o: n for n in graph.node for o in n.output}
    ONNX_PREFIX        = "fraud_detection_lstm_1/preprocessing_1"
    leaked_vocab_names = {inp.name for inp in graph.input if 'Equal/y:0' in inp.name}

    def _keras_name(leaked):
        lws = leaked.split(f"{ONNX_PREFIX}/")[-1].split('/')[0]
        return lws[:-2] if lws.endswith('_1') else lws

    _new_nodes, _to_remove = [], set()
    _replaced = 0

    for leaked_name, keras_name in sorted({n: _keras_name(n) for n in leaked_vocab_names}.items()):
        if keras_name not in _layer_vocab:
            print(f"  WARNING: no vocab for {keras_name!r}"); continue
        vocab_tokens = _layer_vocab[keras_name]
        is_onehot    = _layer_is_onehot[keras_name]
        vocab_size   = len(vocab_tokens)

        vc_node = next((n for n in graph.node if n.op_type=='Cast' and n.input[0]==leaked_name), None)
        if not vc_node: continue
        vc_out  = vc_node.output[0]

        for eq_node in [n for n in graph.node if n.op_type=='Equal' and vc_out in n.input]:
            fc_out  = eq_node.input[0] if eq_node.input[1]==vc_out else eq_node.input[1]
            fc_node = out_to_node.get(fc_out)
            if not fc_node or fc_node.op_type != 'Cast': continue
            sl_node = out_to_node.get(fc_node.input[0])
            if not sl_node or sl_node.op_type != 'Slice': continue
            un_node = out_to_node.get(sl_node.input[0])
            if not un_node or un_node.op_type != 'Unsqueeze': continue
            str_in = un_node.input[0]   # [batch,1] string tensor — our new input

            sq_node = next((n for n in graph.node if n.op_type=='Squeeze' and eq_node.output[0] in n.input), None)
            if not sq_node: continue
            ps_node = next((n for n in graph.node if sq_node.output[0] in n.input), None)
            if not ps_node or ps_node.op_type != 'Cast': continue

            ax_node = next((n for n in graph.node if n.op_type=='ArgMax' and ps_node.output[0] in n.input), None)
            marks   = [fc_node, sl_node, un_node, eq_node, sq_node, ps_node]

            if ax_node:
                ex_node = next((n for n in graph.node if n.op_type=='Unsqueeze' and ax_node.output[0] in n.input), None)
                c1_node = next((n for n in graph.node if n.op_type=='Cast' and ex_node.output[0] in n.input), None) if ex_node else None
                if not ex_node or not c1_node: continue
                final_out = c1_node.output[0]
                marks.extend([ax_node, ex_node, c1_node])
            else:
                final_out = ps_node.output[0]

            sid     = f"_vf_{_replaced}"
            sh_name = f"{sid}_sh";  rs_out = f"{sid}_rs"
            graph.initializer.append(numpy_helper.from_array(np.array([-1], dtype=np.int64), name=sh_name))
            _new_nodes.append(onnx.helper.make_node('Reshape', [str_in, sh_name], [rs_out], name=f"{sid}_R"))
            le_out = f"{sid}_le"
            _new_nodes.append(onnx.helper.make_node('LabelEncoder', [rs_out], [le_out],
                name=f"{sid}_LE", domain='ai.onnx.ml',
                keys_strings=vocab_tokens, values_int64s=list(range(vocab_size)), default_int64=-1))

            if ax_node is None:
                _new_nodes.append(onnx.helper.make_node('OneHotEncoder', [le_out], [final_out],
                    name=f"{sid}_OHE", domain='ai.onnx.ml',
                    cats_int64s=list(range(vocab_size)), zeros=1))
            else:
                sh2 = f"{sid}_sh2"; r2 = f"{sid}_r2"
                graph.initializer.append(numpy_helper.from_array(np.array([-1,1], dtype=np.int64), name=sh2))
                _new_nodes.append(onnx.helper.make_node('Reshape', [le_out, sh2], [r2],     name=f"{sid}_R2"))
                _new_nodes.append(onnx.helper.make_node('Cast',    [r2], [final_out],        name=f"{sid}_CF", to=TensorProto.FLOAT))

            for n in marks: _to_remove.add(n.name)
            _replaced += 1

    for n in graph.node:
        if n.op_type == 'Cast' and n.input[0] in leaked_vocab_names:
            _to_remove.add(n.name)

    # Rebuild node list in topological order
    def _topo_sort(nodes, produced):
        sorted_nodes, remaining = [], list(nodes)
        while remaining:
            ready = [n for n in remaining if all(i == '' or i in produced for i in n.input)]
            if not ready:
                ready = [remaining[0]]   # cycle-breaker: force-emit the first stuck node
            for n in ready:
                sorted_nodes.append(n); produced.update(n.output)
            remaining = [n for n in remaining if n not in ready]
        return sorted_nodes

    all_nodes = [n for n in graph.node if n.name not in _to_remove] + _new_nodes
    produced  = {i.name for i in graph.input} | {i.name for i in graph.initializer}
    del graph.node[:]; graph.node.extend(_topo_sort(all_nodes, produced))

    # Rebuild graph.input: strip leaked vocab and normalization constants
    feature_inputs_only = [inp for inp in list(graph.input)
                           if 'Equal/y:0' not in inp.name
                           and 'Sub/y:0'   not in inp.name
                           and 'Sqrt/x:0'  not in inp.name]
    del graph.input[:]
    graph.input.extend(feature_inputs_only)

    print(f"  Replaced {_replaced} vocab subgraphs ({len(_to_remove)} nodes removed, {len(_new_nodes)} added).")

    # 4. Embed normalization mean/variance as ONNX initializers (Bug 1)
    graph_value_info   = list(graph.input) + list(graph.value_info) + list(graph.output)
    graph_tensor_names = {vi.name for vi in graph_value_info} | {n for node in graph.node for n in node.input}
    _norm_feed = {}
    for ln, (m, v) in _layer_norm.items():
        for tensor_name in sorted(n for n in graph_tensor_names
                                  if n.endswith(f"/{ln}_1/Sub/y:0") or n.endswith(f"/{ln}_1/Sqrt/x:0")):
            _norm_feed[tensor_name] = m if tensor_name.endswith('/Sub/y:0') else v

    for name in sorted(_norm_feed):
        if not any(init.name == name for init in graph.initializer):
            graph.initializer.append(numpy_helper.from_array(_norm_feed[name], name=name))
    new_inp = [i for i in graph.input if i.name not in _norm_feed]
    del graph.input[:]
    graph.input.extend(new_inp)

    print(f"  Embedded {len(_norm_feed)} norm constants. Remaining graph inputs: {len(graph.input)}")

    onnx.checker.check_model(model_proto)
    onnx.save(model_proto, onnx_output_path)
