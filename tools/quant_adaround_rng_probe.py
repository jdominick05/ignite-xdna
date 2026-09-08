"""How many times does Quark's finetune wrapper draw from the torch RNG per layer?

Quark's QConv2d/QGemm (`create_torch/quant_conv_ops.py`, `quant_gemm_ops.py`) inherit
from QuantizeWrapper and the torch layer; QuantizeWrapper.__init__ calls the torch init
through super() and the subclass then calls it again, so the global generator advances
twice per module before the first `torch.randperm` batch draw. Ignition's transcription
(`quant/adaround.py`) constructs the torch layer and calls reset_parameters() once more
to consume the same stream. This probe measures that claim: after the same seed, it
compares the first randperm after Quark's wrapper with the first randperm after the
single- and double-init constructions. Runs in resnet_env because it imports Quark.
"""
import torch
from quark.onnx.algorithm.finetuning.create_torch.quant_conv_ops import QConv2d
from quark.onnx.algorithm.finetuning.create_torch.quant_gemm_ops import QGemm
from quark.onnx.algorithm.finetuning.torch_utils import setup_seed

SEED = 1705472343


def first_draw(build):
    setup_seed(SEED)
    build()
    return torch.randperm(64)[:8].tolist()


def main():
    verdicts = []
    conv = dict(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                dilation=(1, 1), groups=1, bias=True)
    for label, kwargs in (("conv_bias", conv), ("conv_nobias", dict(conv, bias=False)),
                          ("conv_1x1_2048", dict(conv, in_channels=512, out_channels=2048, kernel_size=(1, 1),
                                                stride=(1, 1), padding=(0, 0)))):
        quark = first_draw(lambda: QConv2d(**kwargs))
        double = first_draw(lambda: torch.nn.Conv2d(**kwargs).reset_parameters())
        single = first_draw(lambda: torch.nn.Conv2d(**kwargs))
        verdict = "double" if quark == double else "single" if quark == single else "neither"
        verdicts.append(verdict)
        print(f"RNG_PROBE {label} quark={quark} double={double} single={single} -> {verdict}")
    gemm = dict(in_features=2048, out_features=1000, bias=True)
    quark = first_draw(lambda: QGemm(transB=1, **gemm))
    double = first_draw(lambda: torch.nn.Linear(**gemm).reset_parameters())
    single = first_draw(lambda: torch.nn.Linear(**gemm))
    verdict = "double" if quark == double else "single" if quark == single else "neither"
    verdicts.append(verdict)
    print(f"RNG_PROBE gemm quark={quark} double={double} single={single} -> {verdict}")
    print("TORCH", torch.__version__, "SEED", SEED)
    print("RNG_PROBE_PASS", all(v == "double" for v in verdicts))


if __name__ == "__main__":
    main()
