"""AdaRound weight-rounding optimisation for an emitted XINT8 file (torch, imported lazily).

Transcribes Quark 0.11rc1's FastFinetune AdaRound path
(``quark/onnx/algorithm/finetuning``: ``fast_finetune.py``, ``onnx_subgraph.py``,
``torch_utils.py``, ``train_torch/train_model.py``, ``train_torch/train_model_loss.py``,
``create_torch/base_qdq_quantizers.py``, ``create_torch/create_model_ops.py``) for the
folded ResNet XINT8 dialect. It runs *after* emission and refinement, on the finished
file, exactly where Quark's post-process runs it. Only ``<w>_quantized`` initializers
change; scales, zero points, biases and topology are untouched.

Per Conv/Gemm layer, in the quantized file's node order and sequentially (layer i sees
the rounding chosen for layers < i):

1. Data: the layer's pre-QuantizeLinear input from the *current* quantized graph
   (ONNX Runtime CPU, ``ORT_DISABLE_ALL``); the float input and the float output
   (Relu output when the layer feeds a Relu) from the equalized float graph (ONNX
   Runtime CPU, default optimization) for the first ``DataSize`` calibration images.
2. Module: ``nn.Conv2d``/``nn.Linear`` built with Quark's kwargs and then
   ``reset_parameters()`` once more, because Quark's wrapper runs the torch init twice
   (measured: the global RNG advances twice per layer); float weight and bias loaded;
   input fake-quantized with the file's UINT8/zp128 scale (round half to even, clamp
   0..255); weight fake-quantized with the file's INT8 scale through the AdaRound
   rounding variable (floor plus rectified sigmoid, gamma -0.1, zeta 1.1); bias
   fake-quantized with the file's INT8 bias scale; Relu when present.
3. Loss: reconstruction against the float output (squared Frobenius norm over dim 1,
   averaged) plus the rounding regulariser after a warm start, beta annealed by
   cosine; Adam on the rounding variable; random ``BatchSize`` batches drawn with
   ``torch.randperm``; early stop on the windowed rounding loss.
4. Readout: hard rounding (``alpha >= 0``) clamped to the dtype range [-128, 127],
   which is Quark's clamp here, not the producer's [-127, 127].

Parity with Quark is measured, never assumed: the same seed, module construction,
batch draws, data pipeline and loss are transcribed so that a bitwise comparison is
*possible*; the gate reports whatever the graph diff says. This module imports torch
only inside ``finetune``; ``python -m quant quantize`` never imports it.
"""
from dataclasses import asdict, dataclass, field
import random
import time

import numpy as np
import onnx
from onnx import helper
import onnxruntime as ort

from .graph import Graph
from .sources import ImageFolderSource

GAMMA, ZETA = -0.1, 1.1
ACT_OPS = ("Relu", "PRelu", "LeakyRelu", "Gelu", "Tanh", "Clip", "Sigmoid", "Softmax")
QRANGE = {"uint8": (0, 255), "int8": (-128, 127)}


@dataclass
class FastFinetuneConfig:
    """Field names are Quark's ``extra_options["FastFinetune"]`` keys (custom_config.py DEFAULT_ADAROUND_PARAMS)."""
    DataSize: int = 1000
    FixedSeed: int = 1705472343
    BatchSize: int = 2
    NumIterations: int = 1000
    LearningRate: float = 0.1
    OptimAlgorithm: str = "adaround"
    OptimDevice: str = "cpu"
    InferDevice: str = "cpu"
    EarlyStop: bool = True
    # Fixed inside Quark's TrainParameters, not exposed by the preset:
    RegParam: float = 0.01
    BetaRange: tuple = (20, 2)
    WarmStart: float = 0.2
    DropRatio: float = 1.0

    def check(self) -> None:
        if self.OptimAlgorithm != "adaround":
            raise NotImplementedError("Only the adaround algorithm is transcribed")
        if self.OptimDevice != "cpu" or self.InferDevice != "cpu":
            raise NotImplementedError("Only cpu optimisation/inference is transcribed")
        if self.DataSize < 1 or self.BatchSize < 1 or self.NumIterations < 1:
            raise ValueError("DataSize, BatchSize and NumIterations must be positive")
        if self.DropRatio < 1:
            raise NotImplementedError("Mixed float/quantized inputs (DropRatio < 1) are not transcribed")


@dataclass
class QParams:
    scale: float
    zero_point: int
    dtype: str


@dataclass
class Layer:
    index: int
    name: str
    op_type: str
    start: str            # pre-QuantizeLinear input tensor; same name in both graphs
    end: str              # quantized file: Relu output when has_act, else the compute output
    f_end: str            # the same tensor in the float graph (a graph output is renamed by its QDQ)
    has_act: bool
    act_op: str | None
    weight_quantized: str
    weight_float: str
    bias_float: str | None
    input_q: QParams
    weight_q: QParams
    bias_q: QParams | None
    attrs: dict


@dataclass
class LayerReport:
    index: int
    name: str
    start: str
    end: str
    has_act: bool
    images: int
    iterations: int
    early_stop: bool
    recons_before: float
    recons_after: float
    changed_elements: int
    max_lsb: int
    data_seconds: float
    train_seconds: float


@dataclass
class AdaRoundReport:
    config: dict
    images: int
    layers: list = field(default_factory=list)
    torch_version: str = ""
    torch_threads: int = 0
    onnxruntime_version: str = ""
    float_graph_optimization: str = "ORT default (ENABLE_ALL), as Quark's float reference session"
    quant_graph_optimization: str = "ORT_DISABLE_ALL, as Quark's quantized data session"
    peak_rss_bytes: int | None = None
    data_seconds: float = 0.0
    train_seconds: float = 0.0
    wall_seconds: float = 0.0


def _qparams(g: Graph, node: onnx.NodeProto) -> QParams:
    scale, zp = g.initializer(node.input[1]), g.initializer(node.input[2])
    if scale is None or zp is None or scale.shape != () or zp.shape != ():
        raise ValueError(f"Expected scalar scale/zero point at {node.name}")
    return QParams(float(scale), int(zp), str(zp.dtype))


def layer_targets(qg: Graph, fg: Graph) -> list[Layer]:
    """Conv/Gemm layers of the quantized file, in file order, with their QDQ parameters.

    Quark's Subgraph takes a layer when its input comes through Q->DQ and its weight
    through a DQ; anything else is skipped silently there and rejected here.
    """
    layers = []
    float_nodes = {n.name: n for n in fg.nodes()}
    for node in qg.nodes():
        if node.op_type not in ("Conv", "Gemm"):
            if node.op_type in ("ConvTranspose", "MatMul", "InstanceNormalization", "LayerNormalization"):
                raise NotImplementedError(f"{node.op_type} is a Quark finetune target not transcribed here")
            continue
        dq_in = qg.producer(node.input[0])
        q_in = qg.producer(dq_in.input[0]) if dq_in is not None and dq_in.op_type == "DequantizeLinear" else None
        if q_in is None or q_in.op_type != "QuantizeLinear":
            raise ValueError(f"{node.name}: input is not Q->DQ quantized")
        dq_w = qg.producer(node.input[1])
        if dq_w is None or dq_w.op_type != "DequantizeLinear" or qg.initializer(dq_w.input[0]) is None:
            raise ValueError(f"{node.name}: weight is not a DQ of an initializer")
        bias_q, bias_float = None, None
        if len(node.input) == 3:
            dq_b = qg.producer(node.input[2])
            if dq_b is None or dq_b.op_type != "DequantizeLinear" or qg.initializer(dq_b.input[0]) is None:
                raise ValueError(f"{node.name}: bias is not a DQ of an initializer")
            bias_q = _qparams(qg, dq_b)
        consumers = qg.consumers(node.output[0])
        if len(consumers) != 1:
            raise ValueError(f"{node.name}: expected one consumer of the compute output")
        follower = consumers[0]
        if follower.op_type == "QuantizeLinear":
            end, has_act, act_op = node.output[0], False, None
        elif follower.op_type in ACT_OPS:
            end, has_act, act_op = follower.output[0], True, follower.op_type
        else:
            end, has_act, act_op = node.output[0], False, None
        fnode = float_nodes.get(node.name)
        if fnode is None or fnode.op_type != node.op_type or fnode.input[0] != q_in.input[0]:
            raise ValueError(f"{node.name}: float graph has no matching node")
        f_end = fnode.output[0]
        if has_act:
            facts = fg.consumers(fnode.output[0])
            if len(facts) != 1 or facts[0].op_type != act_op:
                raise ValueError(f"{node.name}: float graph activation differs from the quantized file")
            f_end = facts[0].output[0]
        if len(fnode.input) == 3:
            bias_float = fnode.input[2]
            if fg.initializer(bias_float) is None:
                raise ValueError(f"{node.name}: float bias is not an initializer")
        if fg.initializer(fnode.input[1]) is None:
            raise ValueError(f"{node.name}: float weight is not an initializer")
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        layers.append(Layer(len(layers), node.name, node.op_type, q_in.input[0], end, f_end, has_act, act_op,
                            dq_w.input[0], fnode.input[1], bias_float, _qparams(qg, dq_in), _qparams(qg, dq_w),
                            bias_q, attrs))
    if not layers:
        raise ValueError("No Conv/Gemm layers to finetune")
    return layers


def _sub_model(model: onnx.ModelProto, starts: list[str], ends: list[str]) -> onnx.ModelProto:
    """Quark's extract_sub_model: shape inference, then onnx.utils.Extractor."""
    inferred = onnx.shape_inference.infer_shapes(model)
    return onnx.utils.Extractor(inferred).extract_model(starts, ends)


def _run_quantized_inputs(qg: Graph, layer: Layer, images: list[np.ndarray], input_name: str) -> list[np.ndarray]:
    if layer.start == input_name:
        # Quark runs the whole quantized model and reads back its own input; the values
        # are the images themselves.
        return [np.array(image) for image in images]
    sub = _sub_model(qg.model, [input_name], [layer.start])
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(sub.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    return [np.array(session.run([layer.start], {input_name: image})[0]) for image in images]


def _run_float(fg: Graph, layer: Layer, images: list[np.ndarray], input_name: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    sub = _sub_model(fg.model, [input_name], [layer.f_end])
    outputs = [layer.f_end]
    if layer.start != input_name:
        if layer.start not in {v.name for v in sub.graph.output}:
            sub.graph.output.extend([onnx.ValueInfoProto(name=layer.start)])
        outputs = [layer.start, layer.f_end]
    session = ort.InferenceSession(sub.SerializeToString(), providers=["CPUExecutionProvider"])
    f_in, f_out = [], []
    for image in images:
        result = session.run(outputs, {input_name: image})
        if layer.start == input_name:
            f_in.append(np.array(image))
            f_out.append(np.array(result[0]))
        else:
            f_in.append(np.array(result[0]))
            f_out.append(np.array(result[1]))
    return f_in, f_out


def _stack(samples: list[np.ndarray]) -> np.ndarray:
    """Quark: np.array(list of (1, ...)) then reshape((-1, *shape[2:]))."""
    array = np.array(samples)
    return array.reshape((-1, *array.shape[2:]))


def _print(*args) -> None:
    print(*args, flush=True)


def finetune(float_graph: Graph, quant_graph: Graph, source: ImageFolderSource,
             cfg: FastFinetuneConfig | None = None, log=_print) -> AdaRoundReport:
    """Optimise every Conv/Gemm weight rounding in ``quant_graph`` in place; return the report."""
    import torch
    import torch.nn.functional as F

    cfg = cfg or FastFinetuneConfig()
    cfg.check()
    wall_start = time.perf_counter()
    input_name = quant_graph.model.graph.input[0].name
    if float_graph.model.graph.input[0].name != input_name:
        raise ValueError("Float and quantized graphs must share the input name")
    layers = layer_targets(quant_graph, float_graph)
    images = []
    for image in source:
        if len(images) >= cfg.DataSize:
            break
        images.append(np.ascontiguousarray(image, dtype=np.float32))
    if not images:
        raise ValueError("No finetune images")
    report = AdaRoundReport(config=asdict(cfg), images=len(images), torch_version=torch.__version__,
                            torch_threads=torch.get_num_threads(), onnxruntime_version=ort.__version__)
    # Quark's setup_seed, once, before any module is built.
    random.seed(cfg.FixedSeed)
    np.random.seed(cfg.FixedSeed)
    torch.manual_seed(cfg.FixedSeed)
    torch.cuda.manual_seed(cfg.FixedSeed)
    torch.cuda.manual_seed_all(cfg.FixedSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log(f"ADAROUND layers={len(layers)} images={len(images)} seed={cfg.FixedSeed} torch={torch.__version__} "
        f"threads={torch.get_num_threads()} ort={ort.__version__}")

    def qdq(tensor, params: QParams, round_func):
        low, high = QRANGE[params.dtype]
        scale = torch.from_numpy(np.array(params.scale, dtype=np.float32))
        zero_point = torch.from_numpy(np.array(params.zero_point, dtype=np.int64))
        min_q, max_q = torch.from_numpy(np.array(low)), torch.from_numpy(np.array(high))
        quant_t = round_func(tensor / scale) + zero_point
        return (torch.clamp(quant_t, min_q, max_q) - zero_point) * scale

    class RoundHalfToEven(torch.autograd.Function):
        @staticmethod
        def forward(ctx, t):
            return torch.round(t)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.clone()

    class Module(torch.nn.Module):
        def __init__(self, layer: Layer, weight: np.ndarray, bias: np.ndarray | None):
            super().__init__()
            self.layer = layer
            a = layer.attrs
            if layer.op_type == "Conv":
                pads = tuple(a.get("pads", (0,) * (2 * len(a["kernel_shape"]))))
                half = len(pads) // 2
                if pads[:half] != pads[half:]:
                    raise NotImplementedError("Asymmetric Conv pads (Quark inserts ConstantPad) are not transcribed")
                if a.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET"):
                    raise NotImplementedError("auto_pad is not transcribed")
                groups = int(a.get("group", 1))
                kwargs = dict(in_channels=weight.shape[1] * groups, out_channels=weight.shape[0],
                              kernel_size=tuple(a["kernel_shape"]), stride=tuple(a.get("strides", (1,) * half)),
                              padding=pads[:half], dilation=tuple(a.get("dilations", (1,) * half)), groups=groups,
                              bias=bias is not None)
                self.compute = torch.nn.Conv2d(**kwargs)
                self.w_alpha, self.b_beta = 1.0, 1.0
            else:
                self.transA, self.transB = int(a.get("transA", 0)), int(a.get("transB", 0))
                self.w_alpha, self.b_beta = float(a.get("alpha", 1.0)), float(a.get("beta", 1.0))
                if self.transB == 0:
                    kwargs = dict(in_features=weight.shape[0], out_features=weight.shape[1] if bias is None else bias.shape[0])
                else:
                    kwargs = dict(in_features=weight.shape[1], out_features=weight.shape[0] if bias is None else bias.shape[0])
                if bias is None:
                    # Quark's QGemm always builds nn.Linear with a bias and then leaves the
                    # random init in place when the Gemm has none; fail closed instead.
                    raise NotImplementedError("Gemm without bias is not transcribed")
                self.compute = torch.nn.Linear(**kwargs)
            # Quark's QuantizeWrapper.__init__ calls the torch init through super() and the
            # subclass calls it again: two RNG draws per layer (tools/quant_adaround_rng_probe).
            self.compute.reset_parameters()
            self.compute.weight.data = torch.tensor(weight, dtype=torch.float)
            if bias is not None:
                self.compute.bias.data = torch.tensor(bias, dtype=torch.float)
            self.compute.weight.data = torch.tensor(weight)
            if bias is not None:
                self.compute.bias.data = torch.tensor(bias)
            self.alpha = None
            self.use_soft_rounding = True
            self.act = torch.nn.ReLU(inplace=True) if layer.act_op == "Relu" else None
            if layer.has_act and self.act is None:
                raise NotImplementedError(f"Activation {layer.act_op} is not transcribed")

        def initialize_alpha(self):
            scale = torch.from_numpy(np.array(self.layer.weight_q.scale, dtype=np.float32))
            tensor = self.compute.weight.data
            tensor_floor = torch.floor(tensor / scale)
            tensor_diff = (tensor / scale) - tensor_floor
            alpha = -torch.log((ZETA - GAMMA) / (tensor_diff - GAMMA) - 1)
            self.alpha = torch.nn.Parameter(alpha.float(), requires_grad=True)

        def h_alpha(self):
            if self.use_soft_rounding:
                return torch.clamp(torch.sigmoid(self.alpha) * (ZETA - GAMMA) + GAMMA, 0, 1)
            return self.alpha >= 0

        def weight_round(self):
            h_alpha = self.h_alpha().to(self.compute.weight.dtype)
            return lambda t: torch.floor(t) + h_alpha

        def quantized_weight(self):
            """Quark's get_modules_optimized_weight: hard rounding, integer values as float."""
            self.use_soft_rounding = False
            params = self.layer.weight_q
            low, high = QRANGE[params.dtype]
            scale = torch.from_numpy(np.array(params.scale, dtype=np.float32))
            zero_point = torch.from_numpy(np.array(params.zero_point, dtype=np.int64))
            quant_t = self.weight_round()(self.compute.weight.detach() / scale) + zero_point
            return torch.clamp(quant_t, torch.from_numpy(np.array(low)), torch.from_numpy(np.array(high))).numpy()

        def forward(self, x):
            x = qdq(x, self.layer.input_q, RoundHalfToEven.apply)
            weight = qdq(self.compute.weight, self.layer.weight_q, self.weight_round())
            weight = weight * self.w_alpha
            if self.layer.op_type == "Conv":
                c = self.compute
                out = F.conv2d(x, weight, bias=None, stride=c.stride, padding=c.padding, dilation=c.dilation, groups=c.groups)
            else:
                A = x.transpose(-1, -2) if self.transA else x
                B = weight.transpose(-1, -2) if self.transB else weight
                out = torch.matmul(A, B)
            if self.compute.bias is not None:
                bias = qdq(self.compute.bias, self.layer.bias_q, RoundHalfToEven.apply) * self.b_beta
                if out.dim() >= 2 and out.shape[1] == bias.shape[0]:
                    bias = bias.view(1, -1, *([1] * (out.dim() - 2)))
                out = out + bias
            if self.act is not None:
                out = self.act(out)
            return out

    def recon_loss(quant_output, float_output):
        return (torch.norm(quant_output - float_output, p="fro", dim=1) ** 2).mean()

    def beta_at(cur_iter):
        start_beta, end_beta = cfg.BetaRange
        warm_end = cfg.WarmStart * cfg.NumIterations
        rel_iter = (cur_iter - warm_end) / (cfg.NumIterations - warm_end)
        return end_beta + 0.5 * (start_beta - end_beta) * (1 + np.cos(rel_iter * np.pi))

    def round_loss(module, cur_iter):
        if cur_iter < cfg.NumIterations * cfg.WarmStart:
            return torch.tensor(0.0)
        h_alpha = torch.clamp(torch.sigmoid(module.alpha) * (ZETA - GAMMA) + GAMMA, 0, 1)
        reg_term = torch.add(1, -(torch.add(2 * h_alpha, -1).abs()).pow(beta_at(cur_iter))).sum()
        return cfg.RegParam * reg_term

    def recons_metric(module, inp, out):
        """Quark's _recons_metrics for adaround: hard, then soft; the hard value is reported."""
        module.use_soft_rounding = False
        with torch.no_grad():
            hard = float(F.mse_loss(module(inp), out))
        module.use_soft_rounding = True
        with torch.no_grad():
            F.mse_loss(module(inp), out)
        return hard

    for layer in layers:
        data_start = time.perf_counter()
        q_inputs = _run_quantized_inputs(quant_graph, layer, images, input_name)
        f_inputs, f_outputs = _run_float(float_graph, layer, images, input_name)
        q_input, f_input, f_output = _stack(q_inputs), _stack(f_inputs), _stack(f_outputs)
        if q_input.shape != f_input.shape:
            raise ValueError(f"{layer.name}: quantized/float input shapes differ")
        data_seconds = time.perf_counter() - data_start
        log(f"Quark_latency_profiler: finetuning layer {layer.index}, onnx inference (quantized model + float model) "
            f"time consumed {data_seconds:1f}s")
        train_start = time.perf_counter()
        weight = np.array(float_graph.initializer(layer.weight_float))
        bias = None if layer.bias_float is None else np.array(float_graph.initializer(layer.bias_float)).reshape(-1)
        module = Module(layer, weight, bias)
        module.initialize_alpha()
        log(f"Module ({layer.start})->({layer.end}) will be optimized by adaround on cpu")
        all_q = torch.from_numpy(q_input)
        all_f_out = torch.from_numpy(f_output)
        before = recons_metric(module, all_q, all_f_out)
        batch_size = cfg.BatchSize
        if batch_size < 1 or batch_size > q_input.shape[0]:
            log(f"The batch size {batch_size} is invalid, set it to 1")
            batch_size = 1
        inputs_q = [torch.from_numpy(np.expand_dims(q_input[i], axis=0)) for i in range(q_input.shape[0])]
        inputs_f = [torch.from_numpy(np.expand_dims(f_input[i], axis=0)) for i in range(f_input.shape[0])]
        outputs_f = [torch.from_numpy(np.expand_dims(f_output[i], axis=0)) for i in range(f_output.shape[0])]
        module.use_soft_rounding = True
        optimizer = torch.optim.Adam([module.alpha], lr=cfg.LearningRate)
        best_loss, mean_loss = float("inf"), 0.0
        num_batches = cfg.NumIterations / 10          # Quark: float, NumBatches default 1
        log_period = cfg.NumIterations / 10
        iterations, early_stop = 0, False
        for iteration in range(cfg.NumIterations):
            indices = torch.randperm(len(inputs_q))[:batch_size].tolist()
            inp_q = torch.cat([inputs_q[i] for i in indices], dim=0)   # DropRatio 1: quantized inputs only
            out_f = torch.cat([outputs_f[i] for i in indices], dim=0)
            optimizer.zero_grad()
            out_q = module(inp_q)
            recons = recon_loss(out_q, out_f)
            rounding = round_loss(module, iteration)
            total = recons + rounding
            if cfg.EarlyStop and iteration >= cfg.NumIterations * cfg.WarmStart:
                if iteration % num_batches == num_batches - 1:
                    mean_loss = mean_loss / num_batches
                    if mean_loss < best_loss:
                        best_loss = mean_loss
                    else:
                        log(f"adaround Iterations={iteration}, mean loss {mean_loss:5f} (in {int(num_batches)} batches) "
                            f"is not better than best loss {best_loss:5f}, early stop")
                        early_stop = True
                        break
                    mean_loss = 0.0
                else:
                    mean_loss += float(rounding)
            total.backward()
            optimizer.step()
            iterations = iteration + 1
            if iteration % log_period == 0 or iteration == cfg.NumIterations - 1:
                lr = optimizer.param_groups[0]["lr"]
                log(f"adaround iterations={iteration}, lr={lr:f}, loss={float(total):5f} "
                    f"(Recons loss={float(recons):5f}, Rounding loss={float(rounding):5f})")
        module.use_soft_rounding = False
        after = recons_metric(module, all_q, all_f_out)
        log(f"Module ({layer.start})->({layer.end}) recons metrics was optimized from {before:f} to {after:f} "
            f"(diff={after - before:f})")
        previous = quant_graph.initializer(layer.weight_quantized)
        optimized = module.quantized_weight().astype(previous.dtype)
        if optimized.shape != previous.shape:
            raise ValueError(f"{layer.name}: optimized weight shape changed")
        difference = np.abs(optimized.astype(np.int16) - previous.astype(np.int16))
        quant_graph.set_initializer(layer.weight_quantized, optimized, np.int8)
        train_seconds = time.perf_counter() - train_start
        log(f"Quark_latency_profiler: finetuning layer {layer.index}, torch training time consumed {train_seconds:1f}s")
        report.layers.append(LayerReport(layer.index, layer.name, layer.start, layer.end, layer.has_act, len(images),
                                         iterations, early_stop, before, after, int(np.count_nonzero(difference)),
                                         int(difference.max(initial=0)), data_seconds, train_seconds))
        report.data_seconds += data_seconds
        report.train_seconds += train_seconds
        log(f"LAYER {layer.index}/{len(layers)} {layer.name} changed={int(np.count_nonzero(difference))}/{difference.size} "
            f"max_lsb={int(difference.max(initial=0))} iterations={iterations} early_stop={early_stop}")
    log(f"ONNX inference costs {report.data_seconds:.1f}s and Torch training costs {report.train_seconds:.1f}s")
    try:
        import psutil
        report.peak_rss_bytes = getattr(psutil.Process().memory_info(), "peak_wset", None)
    except ImportError:
        report.peak_rss_bytes = None
    report.wall_seconds = time.perf_counter() - wall_start
    return report
