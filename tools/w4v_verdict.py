#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""W4V: the screen's pre-registration, its pins, the 500-image cut and the K-D verdict, from one set of constants.

PREREG below is the accepted pre-registration text, verbatim: v2.4 (PREREG_V24, LF 7ff9c2c4...) and the v2.5
amendment after the screen1 STOP (PREREG_V25, LF 2d309756...), joined by one newline. Every W4V log prints the
frozen triple and TOOLS_JSON: PREREG_TEXT_SHA256 (that text), PROTOCOL_JSON_SHA256 (protocol(): the models, the
arms, every hyperparameter, the calibration and H pins, the cut and K-D), VERDICT_CODE_SHA256 (the whole LF source
of this file and scripts/w4v-eval.sh, plus verdict_constants()) and TOOLS_JSON (the LF shas of every repo file in
the import graph of TOOL_ROOTS, the code that makes or judges an arm, and scripts/lib.sh; src/ is recorded by the
gate stage's SRC_TREE instead). --pins, --screen, --advanced, --c8 and --verdict refuse unless all four equal those
of the one committed, clean prereg log; pins are read only from committed, clean prereg and pins logs. No stage runs
until every earlier stage's logs are committed and clean, carry their success lines, and hold only hash blocks
equal to this tool's, nor while a log of the stage itself exists (ready(), --ready). v2.5 section E: a CARRIED_V24
log stands in for its stage's v2.5 run while rules 1-5 hold (carry_problems()), and a SUPERSEDED_V24 log is skipped
by every glob that would count it, at exactly its listed sha (live()).

    python tools/w4v_verdict.py --hashes
    python tools/w4v_verdict.py --prereg --files <models, captures, tier-1 arm files>   tier 1's pins, before any mAP
    python tools/w4v_verdict.py --pins --files <tier-2 arm files>                      tier 2's pins, before its screen
    python tools/w4v_verdict.py --pinned <file>             exit 0 iff the committed prereg/pins logs pin this file
    python tools/w4v_verdict.py --ready <stage>             exit 0 iff every earlier stage's logs are as ready() wants
                                                            and the stage has no log yet
    python tools/w4v_verdict.py --src-tree                  SRC_TREE <HEAD:src>; exit 0 iff src/ is clean against HEAD
    python tools/w4v_verdict.py --dets-sha <dets json>      the DETS_SHA256 line an eval row's log ends with
    python tools/w4v_verdict.py --c9 --logs <repro eval logs>   C9: B8 and RTN at 5,000 against gate D (STOP if not)
    python tools/w4v_verdict.py --screen --logs <screen eval logs> [--tier 1|2]
    python tools/w4v_verdict.py --advanced                  the arms the committed cut logs advance, one per line
    python tools/w4v_verdict.py --c8 --logs <W4V_GATE lines>    C8's int32 coverage marker, after the engine gate
    python tools/w4v_verdict.py --verdict --logs <repro and decisive eval logs> --controls <controls, parity, repro
        C9, emit and gate logs>
    python tools/w4v_verdict.py --selftest                  the rule, the H scoring and the dets check, synthetic

An eval row's log is results/int4/w4v_<screen|repro|decisive>_<model>_<arm>_n<500|5000>_<machine>_<date>.log:
pipelines/yolov8n/5_eval_map.py's output, the frozen triple before it and a DETS_SHA256 line after it.
Nothing here was tuned after seeing a mAP: the constants are the ones --prereg prints.
"""
import argparse
import ast
import contextlib
import fnmatch
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "int4"

# v2.4: v2.3, with the gate's F4 ruling (TOOLS_JSON's roots, SRC_TREE) and S2 (a stage's own logs); v2.5: v2.4 plus
# the amendment after the screen1 STOP (zero-detection rows, C5b, the carry-across rule), accepted on 2026-09-26.
# PREREG_SHA is the whole text's. tools/w4_recover.py:1010 (C0) still labels its check "text == the accepted v2.4";
# under v2.5 that line compares with this PREREG_SHA, the accepted v2.4 + v2.5 text.
PREREG_V24_SHA = "7ff9c2c46e9280e5002aafb6cb573646d0c920b6257b6aba4e2ff47a08caee49"
PREREG_V25_SHA = "2d30975688379c02c243538f35cb361923e57e08330617a9b3aa78cb5b3aa1bd"
PREREG_SHA = "56a169e84fe79d3e6b31b6b878ae2fa32ef7ab62f2239b52eebd86c00d2f2f98"

MODELS = ("yolov8n", "yolov8s")
MODEL_PIN = {   # gate D's prereg log :41-42, :47-48 (results/int4/w4a8_accuracy_prereg_desktop2_20260923.log)
    "yolov8n": {"float": ("models/yolov8n_cut.onnx", "11b52c2530def3c99bccc4bee39a40683aab8619d4278a334b5f0f78c43fd0a1"),
                "xint8": ("models/yolov8n_cut_xint8.onnx",
                          "f02e86ba1bf61ff3fdc159e29b15259567e279df06f68c6d9adeb966fe56c885")},
    "yolov8s": {"float": ("models/yolov8s_cut.onnx", "f713208d79578f41240d2ffaca96dfca711ad45bcf251326adb68da212ab04d5"),
                "xint8": ("models/yolov8s_cut_xint8.onnx",
                          "065f62523000b806e47e227f0b9d71a689f10d8c5368a7c5b6d0ff3b9f733fcc")},
}
GATE_D_FILES = {   # C1: w8check (w4a8_controls_*_desktop2_20260923.log:76); C2: the W4 files (prereg log :43-46, :49-52)
    "yolov8n": {"w8check": "ec48c462cae898af2ce5f6e12f3071477c571ba47c12c3d07e9a4876485e14ec",
                "e1": "4b0c957edb101091d28c02bbc544a8c980fedec9e4d6ca25c1136322b9f96733",
                "e2": "b90d25b6f780296df48c6ad633a26a0af3240ef8d5985fbcb782a9b12400a37d",
                "e1h": "f600d16e19fd13639071afa4d9de1da759af7cfba7332e48d3316a1edbc70a4b",
                "u": "b1fd9f528f54e524c80bc826d325ba88c4f88024d5d63131179eb063bc35346f"},
    "yolov8s": {"w8check": "434ffa8bc3e42d409e61ead968fe7eb384e3ed208a5ab77f4e157059ac964395",
                "e1": "92633c4528637553de18a83821f69c560e57597b080426ce3c02d2a53b8130a3",
                "e2": "15bdc6caf9363f5da6f572c95c50c0ce4bb0485c4fddb352b2ac9a99f8108475",
                "e1h": "b531a131f543ce1d15735987de803b3879293d6a3a90b76bc8d599d638b713ea",
                "u": "946bf15230b7b5c2921d6d13556410f8341bae585fe9b73e0ec2df85f4ff17cc"},
}
GATE_D_MAP = {   # C9: gate D's 5,000-image figures (results/int4/w4a8_accuracy_verdict_desktop2_20260923.log)
    "yolov8n": {"b8": "27.10", "rtn_e1": "0.06", "rtn_e2": "0.02", "rtn_u": "1.58"},
    "yolov8s": {"b8": "37.21", "rtn_e1": "2.76", "rtn_e2": "2.76", "rtn_u": "2.47"},
}

MATRIX = (   # (tier, method, forms), section 3.7's row order; it breaks headline ties
    (1, "rtn", ("e1", "e2", "u", "f")),
    (1, "bc", ("e1", "e2", "u")),
    (1, "vc", ("e1", "e2", "u")),
    (1, "ar", ("e1", "e2", "u")),
    (1, "gptq", ("e1", "e2", "u", "f")),
    (2, "arbc", ("e1", "e2", "u")),
    (2, "gptqbc", ("e1", "e2", "u")),
    (2, "mxar", ("e1", "e2", "u")),
    (2, "mxgptq", ("e1", "e2", "u")),
)
METHOD_TEXT = {"rtn": "RTN", "bc": "BC", "vc": "VC (folded variance correction)", "ar": "AdaRound", "gptq": "GPTQ",
               "arbc": "AdaRound + BC", "gptqbc": "GPTQ + BC", "mxar": "MX-AdaRound", "mxgptq": "MX-GPTQ"}
FORM_TAG = {"e1": "E1", "e2": "E2", "u": "U", "f": "F"}
FORM_COST = {"e1": "E1 lowers today", "e2": "E2 needs a shift per packet (a compiler change, phase 2)",
             "u": "U needs a kernel change (phase 2)", "f": "F is an upper bound only"}
DECIDING_FORMS = ("e1", "e2", "u")
INT32_BIAS = ("bc", "vc", "arbc", "gptqbc")
MX_METHODS = ("mxar", "mxgptq")
C9_ARMS = ("rtn_e1", "rtn_e2", "rtn_u")

GRID, WINDOW, GROUP, K_MAX = (-7, 7), (-1, 5), 32, 4
CALIB = {"dir": "data/coco_calib", "n": 64, "order": "sorted file names (quant.sources.CocoSource)",
         "first": "000000000139.jpg", "last": "000000119995.jpg",
         "listing_sha256": "07fcde156cd48c6a7b5e1fd726a1d93c649021d355cada6122281f8c84871a45",
         "listing_form": "'<name> <sha256>\\n' per file, the first n by name",
         "all_count": 300, "all_names_sha256": "41f5539df88fa7d97a8ba39d97551f8ce4c93e85747f0b1d375bbffdcb15e6df",
         "preprocess": "npu.yolo.letterbox, the graph input's size (640), as 5_eval_map.py"}
ANN = {"path": "data/coco/annotations/instances_val2017.json", "bytes": 19987840,
       "sha256": "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"}
H_PIN = {"definition": "sorted val2017 image ids, minus every id of data/coco_calib (all 300)",
         "id_text": "'<id>\\n' per id, ascending",
         5000: (4700, "ae0670418aa729575165614527596c559c74be2dc3c39ea3bf84da7200f11e02"),
         500: (468, "2e297c3beba1eebfdbc4042f07ae3d8a4e8582f1652aedadf9c06834c6f940cd")}
SCREEN_N, DECISIVE_N = 500, 5000
CUT_LINE, KILL_LINE, NARROW_LINE = 2.0, 1.0, 2.0
EVAL = {"script": "pipelines/yolov8n/5_eval_map.py", "ep": "cpu", "ort_opt": "disable_all", "threads": "8",
        "conf": 0.001, "iou": 0.7, "max_det": 300, "images": "sorted(coco.getImgIds())[:n]; --n 0 for all 5,000"}
ORT_VERSION, ENV = "1.23.3.dev20260320", "resnet_env17"
HYPER = {
    "order": "sequential; GPTQ, BC, VC and MX-GPTQ in the XINT8 graph's Conv order (quant.graph topological); "
             "AdaRound and MX-AdaRound in quant/adaround's vendor_order; the ORDER control checks that both orders "
             "are topological",
    "capture": "GPTQ, BC and VC: a sub-model to each Conv's input from the current QDQ graph, ORT_DISABLE_ALL, 8 "
               "intra-op threads; y_f from the float head-cut graph, the same session options. AdaRound: "
               "quant/adaround's own sessions, as in its parity runs (ORT's default intra-op threads; the float "
               "reference at ORT's default graph optimisation); torch at 8 threads",
    "reference": "the float graph's Conv output on the float network's own input (DropRatio 1); GPTQ uses its "
                 "own sequential quantized input",
    "bias": "RTN/GPTQ/AdaRound keep the shipped int8 bias; BC and VC write int32 at s_x * s_w (per channel in "
            "form a), rint in float64",
    "bc": "b'_c = b_c + mean(y_f,c) - mean(y_q,c) over every output position of the calibration images",
    "vc": "a_c = std(y_f,c) / std(y_q,c) (population std; a_c = 1 where std(y_q,c) = 0); w' = a_c * w; RTN with "
          "pos4 re-chosen on w' by MinMSE; then the bias as BC on the same captured input",
    "adaround": {"FixedSeed": 1705472343, "BatchSize": 2, "NumIterations": 1000, "LearningRate": 0.1,
                 "EarlyStop": True, "RegParam": 0.01, "BetaRange": [20, 2], "WarmStart": 0.2, "DropRatio": 1.0,
                 "zeta_gamma": [1.1, -0.1], "DataSize": 64, "OptimDevice": "cpu",
                 "override": "quant.adaround.WeightGrid per layer: 2**-pos4 per output channel, [-7, 7]; "
                             "None keeps an MX W8 layer",
                 "sessions": "quant/adaround's own: ORT's default intra-op threads, the float reference at ORT's "
                             "default graph optimisation; torch at 8 threads"},
    "gptq": {"hessian": "2 X X^T / N, float64, every output position of the 64 images", "damp": 0.01,
             "dead_column": "H_ii = 0 -> H_ii = 1 and W[:, i] = 0", "act_order": True, "block": 128,
             "quantizer": "the arm's per-row scale, clip(rint(w / s), -7, 7)", "layout": "ONNX (cin, kh, kw)"},
    "f": {"alphas": "0.50, 0.51, ..., 1.00", "scale": "alpha * max|w_c| / 7, float32, first minimum wins"},
    "mx": {"budget": 0.05, "sensitivity": "sum ||y_f - y_q||^2 / sum ||y_f||^2 on B8's captured Conv inputs: y_f "
                                          "the float weights, y_q the RTN-W4 weights, both with the float bias; "
                                          "once per model and form",
           "rule": "most sensitive first; stop at the first conv that would take the W8 share of conv weight "
                   "elements over the budget"},
    "torch_threads": 8,
}

# v2.5 section E: the v2.4 values a carried log's hash blocks must equal (rule 2; its TOOLS_JSON is rule 3's), the
# carried logs and the superseded ones (name -> HEAD LF sha256), generated from HEAD's logs, never retyped
V24_FROZEN = {   # the v2.4 prereg log's one hash block (results/int4/w4v_prereg_desktop2_20260926.log, LF 35c758ec...)
    "PREREG_TEXT_SHA256": "7ff9c2c46e9280e5002aafb6cb573646d0c920b6257b6aba4e2ff47a08caee49",
    "PROTOCOL_JSON_SHA256": "46297af64e23a10df62c363f766bacc3f78faf121b5194ea8e6077b364710872",
    "VERDICT_CODE_SHA256": "2fc815d93451a3be1f8403d768cddb5737218d9e58abe282cc25e77fb7eb7c64",
    "TOOLS_JSON": {
        "ignite_xdna/__init__.py": "ef016e665caaf8a1f43c196632b0cddf6e8b693a4b10efba3ac2fe7301198ab2",
        "npu/__init__.py": "7c8079f009174f9992dc12578079959704d0cb08710f8d68c666925559b8c599",
        "npu/modnet.py": "d6239523a4041df1194afc6b78a24bdf9c0f6e03039e4de4126d7938f9d015ca",
        "npu/paths.py": "e5e05f6a3e7bbc070b059a603281478ce76ee28f036361e90985b3c424c788a1",
        "npu/preprocess.py": "d981eb2c9cee4b941670ebd41ceb8d4c6636e14ea5a1e78bbaa3dd7f1bbf09ab",
        "npu/session.py": "1f7ec8e0aba8e70000d4c704aeb11a7ece6a33b8d57b9c8c7b34642eaca0edcc",
        "npu/yolo.py": "04222f86d1e2da8ca2cc68e0675cd75883ce541a804be4ba916df7c4ed3abd04",
        "npu/yolo_decode.py": "a245363a16fc845384611c125ba60725c106622f07b69b8567afd211b2c235fb",
        "pipelines/yolov8n/5_eval_map.py": "fb25e914e1b28078e0a99413c14b491dfcb57a17c6fb5b740ac0eb51145a753f",
        "quant/__init__.py": "6a457c2b7f7de2ed3e56fa9020d171291bff4b2633a4ce16b4b028e37b7289ec",
        "quant/__main__.py": "b67e8ef3ca621732d04cc7f54627011e89e00ed964b932561198519f7d1cd2af",
        "quant/adaround.py": "60f06bd2a3f28bfcb99f55ae462c2f41b3469f9fd3565877418dde95bb6a0704",
        "quant/calib.py": "a68f24d9bc39fcbc51988ce7b6b9dd79bf62f7d454689886d866ab8d04a2dcd6",
        "quant/cle.py": "6b876fc61b66ce348cbc9053974927a16888798305096bd2ccd23ebde2285e35",
        "quant/cli.py": "fb05ead47ee3d0fc25c41d305ec3d6bd7e57fafde074c9266c5737f59c87a194",
        "quant/graph.py": "35171c86a1535a34fbfaac40e54a1c8646dadec25af287df710b9feb5ae89342",
        "quant/passes.py": "63df9a473fe7fc8b576e0313b8ba8124a17a0541f212b3bde1befc17d8846397",
        "quant/pow2.py": "2212eaa226d72ad6b54a75f97a854ec9a9589861cd92baff6fea1d8d2d752fab",
        "quant/qdq.py": "4c4cf421ade75c578762e373f99ca2aba2daed1bf1ab8590d8dd044783958286",
        "quant/quantize.py": "6fbdc88bd88bdbfbb8882f53cb3c3586e7af916dcf0d665f45b2e284bcc3b327",
        "quant/refine.py": "2f10e3bcaaa57c441f029af91fbe2bc87c3b07b20d63b74eaf5aca45b58fadfd",
        "quant/report.py": "f186e05bf38d4cb7aff66514e059e0fc7fb02abe3fd4981bfd4980272aa2d4e9",
        "quant/shift_cut.py": "80fa8bfc76c7c13e526107ea6917cc3cee4a0da65caa8a4305ccbbf43fe4246a",
        "quant/sources.py": "0ef5e110296022af5d4b9e36493099fe7be009fe49687c16a1dc30c148ed8286",
        "quant/verify.py": "1263884ebbd17fbbfe1a83b42e971f84a67b847e22fabd132638f83721f6dd18",
        "scripts/lib.sh": "325383a38b0f8cc2d5d30cab3cef0c261dca41c40b8f79c7b5e771de6daa90fd",
        "tools/quant_compare.py": "d193031b0e238fb9ea92bb7c8492bf08d1830c9fa17a6eb5260080a5cbc6f1bf",
        "tools/w4_recover.py": "8a9562972a5183689a95ab405783e59387788f459d54dcd7643b56d4a928b155",
        "tools/w4a8_emulate.py": "298129b5c2de794dc680f18537db7e0a07659637fe8d342d78a44694a3fceeff",
        "tools/w4a8_engine_gate.py": "a8a61cb805fc40aa2b5cd2a166ac32c8296ffb42c3e3b492b05fbaf94b7972c4",
    },
}
CARRIED_V24 = {
    "load_w4v_repro_desktop2_20260925.log": "1a665b29cbd7c30a5994a9e36ecbe42c894363b7726e08cf2224999f86accba9",
    "w4v_controls_yolov8n_desktop2_20260925.log": "3230ae56d3275ece32933480ce0a8f80b8ecfaeac2f00bca7c45a7def5f6ec98",
    "w4v_controls_yolov8s_desktop2_20260925.log": "851fa188434c0f9e0b2a09adb9386d205480bb0b973a6bccd2a563a0fcee7a54",
    "w4v_emit1_yolov8n_desktop2_20260926.log": "8a70e4bc1f6e1e037f0c11570923a5c08f88dba592eb0703c4db89da6f4c9d1f",
    "w4v_emit1_yolov8s_desktop2_20260926.log": "605379a310388e509c0606b9081090ff2405e04260b4516e444a8093d5443a5a",
    "w4v_parity_c6a_yolov8n_desktop2_20260925.log": "d5b83902878341423e84245def7daae8396e1bc6dee599dd776f640a737053b6",
    "w4v_parity_c6a_yolov8s_desktop2_20260925.log": "df44ee761f19f40b9c92d639ff1986d159d55eccf99f4b57df87b69a2c35234b",
    "w4v_parity_c6b_yolov8n_desktop2_20260925.log": "1b0b49ebd013004a649edfaadba60fa614d1c2c2dbe9042a3f192a618536fe39",
    "w4v_parity_c6c_env17_yolov8n_desktop2_20260925.log":
        "6bc22cbc430f740f4d49b33ca2dfaa7de5cfa1ddd826a1e12989ef16331a6d2e",
    "w4v_parity_c6c_yolov8n_desktop2_20260925.log": "f7a14d7c739443783ca5672f648677fc552aa7120b67c55470c21993af5861da",
    "w4v_repro_c9_desktop2_20260925.log": "e174c2c7096d02a1747ea8cd5469a3d899d2efcc2d5d6cb203fbe835804dc348",
    "w4v_repro_yolov8n_b8_n5000_desktop2_20260925.log":
        "cbdb9ae5b861b4de177f20edbeb459683bbe19e296edff8ba33a2c97c3f85a21",
    "w4v_repro_yolov8n_rtn_e1_n5000_desktop2_20260925.log":
        "4e2eca4165cbf578f8cc19e430712a0e5f081d4a90beb5f9f25ac99cec2dfb7b",
    "w4v_repro_yolov8n_rtn_e2_n5000_desktop2_20260925.log":
        "4f3b3d649fd49b766b40cea72cc9a82f17079732a1d90a00b37620edc8ddee42",
    "w4v_repro_yolov8n_rtn_u_n5000_desktop2_20260925.log":
        "0e70b22215c876f98c3364ee74cc252599676c1af4bc8c92ae750afb362d7a2a",
    "w4v_repro_yolov8s_b8_n5000_desktop2_20260925.log":
        "74471da4518272671612dfa80beabc1fb075e77efcc464db8d0b733470c7383e",
    "w4v_repro_yolov8s_rtn_e1_n5000_desktop2_20260925.log":
        "d0268253cf11f9c70344032a8539e24548b5fc435cd1639763f86279d2129493",
    "w4v_repro_yolov8s_rtn_e2_n5000_desktop2_20260925.log":
        "bfd776faa6695ab7b2fe0b451bcd79b61e3079ee6fd0197cf0327105ed022363",
    "w4v_repro_yolov8s_rtn_u_n5000_desktop2_20260925.log":
        "50f7f0e0829a64773e6d418cbea2ce1de2dbd13474bff3d535969d243054ef3e",
    "w4v_sensitivity_yolov8n_desktop2_20260926.log": "4a6e64be7adc4644d2cd2639f646da8cbf4fc8489c6e3deed89c4f1851fc45e8",
    "w4v_sensitivity_yolov8s_desktop2_20260926.log": "beff7fbb90c264ed46a84eaf91eb8bc463e325bfa0b42d0930d5bbec1eb478a1",
}
SUPERSEDED_V24 = {
    "w4v_prereg_desktop2_20260926.log": "35c758eceafd53d4c99c62fadc0fe541188f694dbcd27741f620fb49567acc6e",
    "w4v_selftest_desktop2_20260925.log": "55f34e43e42fd736c36e2d23f6a679fd73130fe26bdc6140644697fa25f3c5dd",
}
# v2.5 sections B and C: the zero-detection record's audits and C5b's stages
AUDIT_SCRIPT_SHA = "dbc4b7bb4d2d0195f4becf479b4711732f62d0a9f3952fcbfc3c06e34741bd89"   # w4v_audit/w4v_bc_audit.py
C5B_TOOL = "tools/w4v_bias_audit.py"
AUDIT_LOGS = {"yolov8n": "results/int4/w4v_audit/recompute_yolov8n_rerun_desktop2_20260926.log",   # GENUINE, bc_e1
              "yolov8s": "results/int4/w4v_audit/recompute_yolov8s_rerun2_desktop2_20260926.log"}
BIAS1_ARMS = ("bc_e2", "bc_u", "vc_e1", "vc_e2", "vc_u")   # the tier-1 int32-bias arms not yet audited on (a)


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(path) -> str:
    return sha_bytes(Path(path).read_bytes())


def sha_lf(path) -> str:
    return sha_bytes(Path(path).read_bytes().replace(b"\r\n", b"\n"))


def shown(path) -> str:
    """A path relative to the repo when it is inside it, with forward slashes; never a profile path."""
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def say(tag: str, obj) -> None:
    print(f"{tag} " + json.dumps(obj, sort_keys=True), flush=True)


def arms(tier: int | None = None) -> list[str]:
    """Every arm label, <method>_<form>, in section 3.7's row order (optionally one tier)."""
    return [f"{m}_{f}" for t, m, forms in MATRIX if tier in (None, t) for f in forms]


def bias_arms(tier: int) -> list[str]:
    """The arms C5b audits in stage bias<tier>: bias1, BIAS1_ARMS; bias2, every tier-2 int32-bias arm (the +BC arms)."""
    return list(BIAS1_ARMS) if tier == 1 else [a for a in arms(2) if a.split("_")[0] in INT32_BIAS]


def audits() -> dict:
    """'<model> <arm>' -> the audit a zero-detection record cites (v2.5 B.3, B.4): a committed GENUINE log's path for
    bc_e1, else the C5b stage whose one committed log per model covers the arm."""
    return {**{f"{m} bc_e1": AUDIT_LOGS[m] for m in MODELS},
            **{f"{m} {a}": f"bias{t}" for t in (1, 2) for m in MODELS for a in bias_arms(t)}}


def v25() -> dict:
    """protocol()'s one added key (v2.5 E rule 4: the protocol minus it is v2.4's)."""
    return {"v24_frozen": V24_FROZEN, "carried_v24": CARRIED_V24, "superseded_v24": SUPERSEDED_V24,
            "audits": audits(), "audit_script_sha256": AUDIT_SCRIPT_SHA, "c5b_tool": C5B_TOOL,
            "c5b_base": {"bc": "rtn_<form>", "arbc": "ar_<form>", "gptqbc": "gptq_<form>",
                         "vc": "its own weights, from the shipped biases"},
            "bias_arms": {"bias1": bias_arms(1), "bias2": bias_arms(2)},
            "zero_detection": {
                "row": "5_eval_map.py printed '0 detections over <N> images in <s>s' (N the row's image count) and "
                       "'wrote <its --dets path>', then exited with 'no detections at all -- something is broken "
                       "upstream'; its dets file parses as an empty list; no DETS_SHA256 line; its model sha256 line "
                       "is the arm's pin; no skip unreadable line",
                "figures": "AP 0.00 on its N images and 0.00 on H, by rule (B.2); nothing is rescored",
                "recorded": "an int32-bias arm whose audit (audits) reads GENUINE for its exact pinned sha, from "
                            "dbc4b7bb or the frozen C5b tool, with no w4_recover or quant import",
                "stop": "any other arm (D2: RTN, AdaRound, GPTQ, MX), B8, C9's RTN rows, a DEFECT, or no audit",
                "record": "results/int4/w4v_zerodet_<model>_<arm>_n<N>_<machine>_<date>.log from --zerodet; the row "
                          "log stays byte-identical"}}


def protocol() -> dict:
    """v2.4's protocol JSON plus the one key v2.5 adds (v2.5 E rule 4)."""
    return {**protocol_v24(), "v2_5": v25()}


def protocol_v24() -> dict:
    return {"models": MODEL_PIN, "gate_d_files": GATE_D_FILES, "gate_d_map": GATE_D_MAP,
            "matrix": [[t, m, list(f)] for t, m, f in MATRIX], "method_text": METHOD_TEXT,
            "grid": list(GRID), "window": list(WINDOW), "group": GROUP, "k_max": K_MAX,
            "calibration": CALIB, "annotations": ANN,
            "h": {k if isinstance(k, str) else str(k): v for k, v in H_PIN.items()},
            "screen_n": SCREEN_N, "decisive_n": DECISIVE_N, "cut_line": CUT_LINE, "kill_line": KILL_LINE,
            "narrow_line": NARROW_LINE, "deciding_forms": list(DECIDING_FORMS), "int32_bias": list(INT32_BIAS),
            "mx": list(MX_METHODS), "c9_arms": list(C9_ARMS), "eval": EVAL, "onnxruntime": ORT_VERSION, "env": ENV,
            "hyper": HYPER,
            "rules": {"cut": "advance iff B8_500 - arm_500 <= cut_line, printed 2-decimal values",
                      "flag": "report-only: cut on the 500 but H_500 delta <= cut_line",
                      "kd": "pass iff B8_5000 - arm_5000 <= kill_line AND B8_H - arm_H <= kill_line; K-D delta = "
                            "max of the two; PASS per model iff a decision-eligible arm passes",
                      "eligible": "forms e1/e2/u only; a form-b file needs its own C8 PASS, any form (a FAIL loses "
                                  "the arm its eligibility); e1 needs form b",
                      "best": "smallest K-D delta among decision-eligible arms, ties in section 3.7's row order",
                      "narrow": "no decision-eligible arm passes and kill_line < best K-D delta <= narrow_line",
                      "pair": "EXTEND iff one method passes on both models",
                      "incomplete": "a row missing, < n images or any skipped, H not its pinned ids, ORT or threads "
                                    "differ, a control fails or is missing, the frozen triple or TOOLS_JSON "
                                    "mismatches, a log is not committed and clean, lacks its success line or holds a "
                                    "hash block that differs"}}


# ---------------------------------------------------------------------------------------------- the verdict code

LOG_NAME = re.compile(r"^w4v_(screen|repro|decisive)_(yolov8[ns])_(b8|[a-z]+_(?:e1|e2|u|f))_n(\d+)_[a-z0-9]+_\d{8}\.log$")
HASH_LINE = r"^(PREREG_TEXT_SHA256|PROTOCOL_JSON_SHA256|VERDICT_CODE_SHA256) ([0-9a-f]{64})$"
ARM_FILE = re.compile(r"^(yolov8[ns])_w4v_([a-z]+_(?:e1|e2|u|f))(_ortonly)?\.onnx$")
AR_METHODS = ("ar", "arbc", "mxar")   # the arms that run AdaRound: NARROW's 10k clause reads only these
CODE_FILES = ("tools/w4v_verdict.py", "scripts/w4v-eval.sh")   # VERDICT_CODE_SHA256: their whole LF source
# TOOLS_JSON: the repo import graph (import_graph) of the code that makes or judges an arm, this file excepted (it is
# in VERDICT_CODE_SHA256), plus the runner's scripts/lib.sh. src/ is left out: the gate stage records it as
# SRC_TREE. The selftest checks TOOL_FILES against the graph by set equality.
TOOL_ROOTS = ("tools/w4_recover.py",                # makes every arm
              "pipelines/yolov8n/5_eval_map.py",    # writes every row's detections
              "tools/w4a8_engine_gate.py",          # C8's eligibility
              "tools/quant_compare.py", "quant/__main__.py", "quant/cli.py",   # C6
              C5B_TOOL)                             # C5b (v2.5 section C)
TOOL_FILES = (C5B_TOOL,
              "tools/w4_recover.py", "tools/w4a8_emulate.py", "tools/w4a8_engine_gate.py", "tools/quant_compare.py",
              "pipelines/yolov8n/5_eval_map.py",
              "quant/__init__.py", "quant/__main__.py", "quant/adaround.py", "quant/calib.py", "quant/cle.py",
              "quant/cli.py", "quant/graph.py", "quant/passes.py", "quant/pow2.py", "quant/qdq.py",
              "quant/quantize.py", "quant/refine.py", "quant/report.py", "quant/shift_cut.py", "quant/sources.py",
              "quant/verify.py",
              "npu/__init__.py", "npu/modnet.py", "npu/paths.py", "npu/preprocess.py", "npu/session.py",
              "npu/yolo.py", "npu/yolo_decode.py",
              "ignite_xdna/__init__.py",            # the repo-root forwarder into src/
              "scripts/lib.sh")


def import_graph(roots=TOOL_ROOTS, root: Path = ROOT) -> tuple[set[str], set[str]]:
    """The repo files the roots reach, read statically (nothing is imported): every import statement, module level or
    inside a function, and importlib.import_module / __import__ of a literal name. Absolute names resolve as the
    root's own run resolves them, against its directory (a script's sys.path[0]) and the repo root; relative ones
    against their package. Nothing under src/. Also returns the files holding a non-literal dynamic import."""
    root = Path(root).resolve()
    src = root / "src"

    def package(f: Path) -> tuple[list[str], Path]:   # (the dotted package a file is in, the dir it hangs from)
        parts, d = [], f.parent
        while d != root and (d / "__init__.py").is_file():
            parts.insert(0, d.name)
            d = d.parent
        return parts, d

    def resolve(dotted: str, bases: list[Path]) -> list[Path]:   # each package on the way, then the module
        names, files = dotted.split("."), []
        d = next((b for b in bases if (b / names[0] / "__init__.py").is_file() or (b / f"{names[0]}.py").is_file()),
                 None)
        for n in names if d else []:
            if (d / n / "__init__.py").is_file():
                files.append(d / n / "__init__.py")
                d = d / n
            else:
                files += [d / f"{n}.py"] if (d / f"{n}.py").is_file() else []
                break
        return files

    found, dynamic = set(), set()
    for first in (root / r for r in roots):
        bases = ([first.parent] if not package(first)[0] else []) + [root]
        seen, queue = set(), [first]
        while queue:
            f = queue.pop()
            if f in seen or src in f.parents:
                continue
            seen.add(f)
            pkg, anchor = package(f)
            for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
                targets = []
                if isinstance(node, ast.Import):
                    targets = [(a.name, bases) for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        mod = ".".join(pkg[:len(pkg) - node.level + 1] + ([node.module] if node.module else []))
                        tb = [anchor]
                    else:
                        mod, tb = node.module, bases
                    targets = [(mod, tb)] + [(f"{mod}.{a.name}", tb) for a in node.names if a.name != "*"]
                elif (isinstance(node, ast.Call)
                      and getattr(node.func, "attr", getattr(node.func, "id", None)) in ("import_module", "__import__")):
                    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                        targets = [(node.args[0].value, bases)]
                    else:
                        dynamic.add(f.relative_to(root).as_posix())
                for mod, tb in targets:
                    queue += [r for r in resolve(mod, tb) if r not in seen]
        found |= {p.relative_to(root).as_posix() for p in seen if src not in p.parents}
    return found, dynamic


def hash_blocks(text: str) -> list[dict]:
    """Every hash block a log printed, in order. PREREG_TEXT_SHA256 opens one; PROTOCOL_JSON_SHA256,
    VERDICT_CODE_SHA256 and TOOLS_JSON fill it. A log of several tool calls (an emit log) holds several. A line
    outside a block, or twice in one, leaves a block that matches nothing."""
    blocks = []
    for line in text.splitlines():
        m = re.match(HASH_LINE, line)
        t = re.match(r"^TOOLS_JSON (\{.*\})$", line)
        if not (m or t):
            continue
        if m:
            key, value = m.group(1), m.group(2)
        else:
            try:
                key, value = "TOOLS_JSON", json.loads(t.group(1))
            except ValueError:   # an unreadable TOOLS_JSON line: the block matches nothing, a PROBLEM, not a traceback
                key, value = "TOOLS_JSON", None
        if key == "PREREG_TEXT_SHA256" or not blocks:
            blocks.append({})
        if key in blocks[-1]:
            blocks[-1]["repeated"] = True
        blocks[-1][key] = value
    return blocks


def logged_hashes(text: str) -> dict:
    """The one set of frozen values a log printed: its hash blocks when they all agree, else {} (which matches
    nothing)."""
    blocks = hash_blocks(text)
    return blocks[0] if blocks and all(b == blocks[0] for b in blocks) else {}


def parse_eval_log(path: Path) -> dict:
    """One eval row: the frozen hashes, 5_eval_map.py's lines and the wrapper's DETS_SHA256 line."""
    m = LOG_NAME.match(path.name)
    if not m:
        raise ValueError(f"{path.name}: not w4v_<stage>_<model>_<arm>_n<N>_<machine>_<date>.log")
    text = path.read_text(encoding="utf-8")

    def one(pattern):
        found = re.findall(pattern, text, flags=re.M)
        if len(found) != 1:
            raise ValueError(f"{path.name}: expected one match of {pattern!r}, found {len(found)}")
        return found[0]
    dets_sha, dets = one(r"^DETS_SHA256 ([0-9a-f]{64}) (\S+)$")
    return {"stage": m.group(1), "model": m.group(2), "arm": m.group(3), "n": int(m.group(4)), "log": path.name,
            "map": one(r"^mAP@50-95\s+([0-9]+\.[0-9]{2})$"),
            "images": int(one(r"^=== \S+ \| CPU \| \d+px \| (\d+) images ===$")),
            "ort_opt": one(r"^ort-opt : (\S+)"), "threads": one(r"ort-threads : (.+)$").strip(),
            "ort": one(r"^onnxruntime (\S+)$"), "sha256": one(r"^model sha256 ([0-9a-f]{64})$"),
            "dets": dets, "dets_sha256": dets_sha,
            "skipped": len(re.findall(r"^\s*skip unreadable", text, flags=re.M)),
            "triple": logged_hashes(text)}


def h_ids(ids: list[int], calib_ids: set[int]) -> list[int]:
    return [i for i in ids if i not in calib_ids]


def id_text_sha(ids: list[int]) -> str:
    return sha_bytes("".join(f"{i}\n" for i in ids).encode())


def score(coco, dets_path, id_sets: list[list[int]]) -> list[str]:
    """mAP@50-95 x 100 at 2 decimals for each id set, from one detections file: 5_eval_map.py's pycocotools calls."""
    from pycocotools.cocoeval import COCOeval
    out = []
    with contextlib.redirect_stdout(io.StringIO()):
        res = coco.loadRes(str(dets_path))
        for ids in id_sets:
            ev = COCOeval(coco, res, "bbox")
            ev.params.imgIds = list(ids)
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
            out.append(f"{ev.stats[0] * 100:.2f}")
    return out


def dets_check(row: dict, coco, ids: list[int], h: list[int], root: Path) -> dict:
    """The detections file must be the logged one, and re-scoring it on the row's images must give the logged
    figure; only then is H read from the same file (a failed check carries no H figure)."""
    path = root / row["dets"]
    got = {"dets_found": path.is_file()}
    if not got["dets_found"]:
        return {**got, "ok": False}
    got["dets_sha_equal"] = sha_file(path) == row["dets_sha256"]
    if not got["dets_sha_equal"]:
        return {**got, "ok": False}
    rescored, h_map = score(coco, path, [ids, h])
    got.update({"rescored": rescored, "rescored_equal": rescored == row["map"]})
    got["ok"] = got["rescored_equal"]
    if got["ok"]:
        got.update({"h": h_map, "h_images": len(h)})
    return got


# v2.5 section B: zero-detection rows. B.1's three lines, 5_eval_map.py:233, :240 and :243's exit message
ZERO_LINES = ((r"^0 detections over (\d+) images in \S+s$", "0 detections over <N> images"),
              (r"^wrote (\S+)$", "wrote <dets>"),
              (r"^(no detections at all -- something is broken upstream)$", "no detections at all"))


def row_of(log_name: str) -> str:
    """w4v_<stage>_<model>_<arm>_n<N> from its log's name (the runner's row name; its dets file is <row>.json)."""
    return log_name[:-len(".log")].rsplit("_", 2)[0]


def zero_row(path: Path, root: Path = ROOT) -> tuple[dict, list[str]]:
    """v2.5 B.1, line by line, on an eval row log with no DETS_SHA256 line: 5_eval_map.py printed '0 detections over
    <N> images' (N the row's n) and 'wrote <the runner's dets path>', then exited with 'no detections at all'; the
    dets file parses as an empty list; no skip unreadable line. The row's fields, with its figures 0.00 by rule B.2,
    and the problems (none iff B.1 holds)."""
    m = LOG_NAME.match(path.name)
    if not m:
        return {}, [f"{path.name}: not w4v_<stage>_<model>_<arm>_n<N>_<machine>_<date>.log"]
    text = Path(path).read_text(encoding="utf-8")
    problems = []

    def one(pattern, what):
        found = re.findall(pattern, text, flags=re.M)
        if len(found) != 1:
            problems.append(f"B.1: {len(found)} '{what}' lines, want one")
            return None
        return found[0]
    n = int(m.group(4))
    got_n, wrote, _ = (one(p, w) for p, w in ZERO_LINES)
    sha, ort_v = one(r"^model sha256 ([0-9a-f]{64})$", "model sha256"), one(r"^onnxruntime (\S+)$", "onnxruntime")
    opt, threads = one(r"^ort-opt : (\S+)", "ort-opt"), one(r"ort-threads : (.+)$", "ort-threads")
    if got_n is not None and int(got_n) != n:
        problems.append(f"B.1: {got_n} images, the row's n is {n}")
    if re.search(DETS_LINE, text, re.M):
        problems.append("B.1: a DETS_SHA256 line: the eval succeeded")
    if re.search(r"^\s*skip unreadable", text, re.M):
        problems.append("B.1: a skip unreadable line")
    want = f"scratch/w4v/dets/{row_of(path.name)}.json"
    if wrote is not None and wrote != want:
        problems.append(f"B.1: wrote {wrote}, not the runner's {want}")
    dets = Path(root) / want
    count = None
    try:
        got = json.loads(dets.read_text(encoding="utf-8"))
        count = len(got) if isinstance(got, list) else None
    except (OSError, ValueError):
        pass
    if count != 0:
        problems.append(f"B.1: {want} is {'missing or not a JSON list' if count is None else f'{count} detections'}")
    row = {"stage": m.group(1), "model": m.group(2), "arm": m.group(3), "n": n, "log": path.name, "map": "0.00",
           "images": int(got_n) if got_n is not None else 0, "ort_opt": opt, "threads": (threads or "").strip(),
           "ort": ort_v, "sha256": sha, "dets": want, "dets_sha256": sha_file(dets) if dets.is_file() else None,
           "skipped": 0, "triple": logged_hashes(text), "zero": True}
    return row, problems


def audit_for(model: str, arm: str, root: Path = ROOT) -> tuple[Path | None, list[str]]:
    """The audit a zero-detection record cites (audits()): the committed GENUINE log for bc_e1, else exactly one
    committed, clean w4v_bias<tier>_<model>_*.log (the gate's Q5 ruling)."""
    ref = audits().get(f"{model} {arm}")
    if ref is None:
        return None, [f"B.3: no audit covers {model} {arm}"]
    if ref.startswith("results/"):
        p, problems = Path(root) / ref, []
    else:
        found, problems = live(f"w4v_{ref}_{model}_*.log", root)
        if len(found) != 1:
            return None, problems + [f"B.3: {len(found)} w4v_{ref}_{model}_*.log, want exactly one"]
        p = found[0]
    if not p.is_file() or not committed_clean(p, root):
        problems.append(f"B.3: {shown(p)} is not committed and clean")
    return p, problems


def audit_problems(path: Path, model: str, arm: str, arm_sha: str, root: Path = ROOT) -> tuple[str | None, list[str]]:
    """v2.5 B.3: the arm's section of an audit log (from its AUDIT line to the next) is a recompute by the accepted
    script (dbc4b7bb) or the frozen C5b tool; it PIN_CHECKs exactly this arm file at its pinned sha; its HELD line
    reads GENUINE in full; no w4_recover or quant import; and the C5b tool's own C5B line reads PASS."""
    sections, cur = [], None
    for line in Path(path).read_text(encoding="utf-8").replace("\r\n", "\n").split("\n"):
        if line.startswith("AUDIT "):
            cur = [line]
            sections.append(cur)
        elif cur is not None:
            cur.append(line)
    mine = [s for s in sections if any(x.startswith(f"HELD {model} {arm} (a) ") for x in s)]
    if len(mine) != 1:
        return None, [f"B.3: {path.name} has {len(mine)} sections with a full HELD line for {model} {arm}, want one"]
    s, problems = mine[0], []
    script = re.match(r"^AUDIT recompute \S.*\| script (LF )?sha256 ([0-9a-f]{64}) \|", s[0])
    c5b_sha = tools_json(root).get(C5B_TOOL)
    if not script or script.group(2) not in (AUDIT_SCRIPT_SHA, c5b_sha):
        problems.append(f"B.3: {path.name}'s {model} {arm} section is not a recompute by dbc4b7bb or the frozen C5b "
                        f"tool")
    pins_seen = re.findall(rf"^PIN_CHECK scratch/w4v/arms/{model}_w4v_{arm}(?:_ortonly)?\.onnx sha256 ([0-9a-f]{{64}})"
                           rf" == pinned$", "\n".join(s), re.M)
    if pins_seen != [arm_sha]:
        problems.append(f"B.3: the section's PIN_CHECK of the arm file is not its pin {arm_sha[:12]}...")
    held = [x for x in s if x.startswith(f"HELD {model} {arm} (a) ")]
    if len(held) != 1 or not held[0].startswith(f"HELD {model} {arm} (a) False (ANCHORED agrees: True) (b) False "
                                                  f"(c) False (d) False -> GENUINE ("):
        problems.append(f"B.3: {model} {arm}'s HELD line does not read GENUINE in full")
    if sum(x.startswith("IMPORTS w4_recover or quant loaded: none |") for x in s) != 1:
        problems.append("B.3: the section's IMPORTS line does not read none")
    if script and script.group(2) == c5b_sha and f"C5B {model} {arm} PASS" not in s:
        problems.append(f"B.3: no C5B {model} {arm} PASS line")
    return held[0] if held else None, problems


def zerodet_problems(row_log: Path, dets: str, audit: str | None, root: Path = ROOT) -> tuple[list[str], list[str]]:
    """--zerodet (v2.5 B.4): B.1 on the row, B.3 on the arm and its audit. The record's lines, and the problems (none
    iff the row is recorded as AP 0.00 on N and 0.00 on H by rule B.2)."""
    row, problems = zero_row(row_log, root)
    if not row:
        return [], problems
    model, arm = row["model"], row["arm"]
    if row["stage"] == "repro" or arm == "b8":
        problems.append(f"B.3: a zero-detection {row['stage']} {arm} row is always a STOP")
    elif arm.split("_")[0] not in INT32_BIAS:
        problems.append(f"B.3 (D2): {arm} has no int32 bias; its zero-detection row is a STOP")
    if dets != row["dets"]:
        problems.append(f"B.1: --dets {dets} is not the row's {row['dets']}")
    try:
        pinned = pins(root)
    except SystemExit as e:
        pinned, problems = {}, problems + [str(e)]
    arm_sha = pinned.get(f"{model}_w4v_{arm}.onnx") or pinned.get(f"{model}_w4v_{arm}_ortonly.onnx")
    if arm_sha is None or row["sha256"] != arm_sha:
        problems.append(f"B.1: the row's model sha256 is not {model} {arm}'s pin")
    held, path = None, None
    if not problems:
        path, ap = audit_for(model, arm, root)
        problems += ap
        if path is not None and (audit is None or Path(audit).resolve() != path.resolve()):
            problems.append(f"B.4: --audit {audit} is not {model} {arm}'s audit {shown(path)}")
        if path is not None and path.is_file():
            held, hp = audit_problems(path, model, arm, arm_sha, root)
            problems += hp
    lines = [f"ZERO_DET_LOG {row_log.name} LF {sha_lf(row_log)}",
             f"ZERO_DET_DETS {row['dets']} sha256 {row['dets_sha256']} count 0",
             f"ZERO_DET_AUDIT {shown(path) if path else 'none'} LF {sha_lf(path) if path and path.is_file() else '-'}",
             f"ZERO_DET_HELD {held}",
             f"ZERO_DET_ARM {model} {arm} sha256 {arm_sha}",
             f"ZERO_DET_ROW {row_of(row_log.name)} AP_N 0.00 AP_H 0.00 (rule B.2)"]
    return lines, problems


def zero_record(row_log: Path, root: Path = ROOT, want: dict | None = None, need_commit: bool = True) -> tuple[
        dict, list[str]]:
    """A row log with no DETS_SHA256 line counts as present iff exactly one w4v_zerodet_<model>_<arm>_n<N>_*.log
    names it (committed and clean, need_commit), its hash blocks equal want, it names this row log at its current LF
    sha and the dets at their current sha, and B.1 re-verifies (v2.5 B.4). The row (0.00 by rule) and the problems."""
    row, problems = zero_row(row_log, root)
    if not row:
        return row, problems
    recs, sup = live(f"w4v_zerodet_{row['model']}_{row['arm']}_n{row['n']}_*.log", root)
    problems += sup
    if len(recs) != 1:
        return row, problems + [f"{len(recs)} zero-detection records name {row_of(row_log.name)}, want one"]
    rec = recs[0]
    text = rec.read_text(encoding="utf-8")
    if need_commit and not committed_clean(rec, root):
        problems.append(f"{rec.name} is not committed and clean")
    blocks = hash_blocks(text)
    if want is not None and (not blocks or any(b != want for b in blocks)):
        problems.append(f"{rec.name}: its hash blocks are not the frozen values")
    for line in (f"ZERO_DET_LOG {row_log.name} LF {sha_lf(row_log)}",
                 f"ZERO_DET_DETS {row['dets']} sha256 {row['dets_sha256']} count 0",
                 f"ZERO_DET_ROW {row_of(row_log.name)} AP_N 0.00 AP_H 0.00 (rule B.2)"):
        if not re.search(rf"^{re.escape(line)}$", text, re.M):
            problems.append(f"{rec.name} lacks '{line[:60]}...' as the row stands now")
    return row, problems


def delta(b8: str, arm: str) -> float:
    return round(float(b8) - float(arm), 2)


def advances(d500: float) -> bool:
    return d500 <= CUT_LINE


def flagged(d500: float, dh: float) -> bool:
    return d500 > CUT_LINE and dh <= CUT_LINE


def kd(d_all: float, d_h: float) -> dict:
    return {"d_all": d_all, "d_h": d_h, "kd": max(d_all, d_h), "pass": d_all <= KILL_LINE and d_h <= KILL_LINE}


def eligible(arm: str, ortonly: bool, gate: str | None) -> bool:
    """F never decides. A form-b file needs its own C8 PASS whatever its form: a FAIL, or no gate row, and the arm
    is not decision-eligible. A form-a file has no engine lowering to gate: E1 needs form b, E2 and U stay."""
    form = arm.split("_")[1]
    if form not in DECIDING_FORMS:
        return False
    if ortonly:
        return form != "e1"
    return gate == "PASS"


def cut_no_c8(row: dict, advanced: list[str]) -> bool:
    """A form-b row in a deciding form that the screen cut, so C8 never checked it: only RTN's 5,000-image rows
    (C9's) can be one. Not decision-eligible; the verdict names it on an INFO line."""
    return (row["arm"].split("_")[1] in DECIDING_FORMS and not row["ortonly"] and row["gate"] is None
            and row["arm"] not in advanced)


def row_order(arm: str) -> int:
    return arms().index(arm)


def phase2_cost(arm: str) -> str:
    method, form = arm.split("_")
    parts = [FORM_COST[form]]
    if method in MX_METHODS:
        parts.append("MX adds per-layer packet types")
    if method in INT32_BIAS:
        parts.append("an int32 bias, which lowers since 2299f0b")
    return "; ".join(parts)


def model_outcome(cands: list[dict]) -> dict:
    """cands: {arm, eligible, d_all, d_h} per 5,000-image arm of one model. ar_narrow: NARROW, and AdaRound's own
    best eligible arm is itself in (kill_line, narrow_line] (section 3.4's 10k clause)."""
    rows = [{**c, **kd(c["d_all"], c["d_h"])} for c in cands]
    elig = sorted((r for r in rows if r["eligible"]), key=lambda r: (r["kd"], row_order(r["arm"])))
    passing = [r for r in elig if r["pass"]]
    best = elig[0] if elig else None
    ar = [r for r in elig if r["arm"].split("_")[0] in AR_METHODS]
    ar_best = ar[0] if ar else None
    if passing:
        return {"outcome": "PASS", "best": passing[0], "passing": [r["arm"] for r in passing], "narrow": False,
                "ar_best": ar_best, "ar_narrow": False, "rows": rows}
    narrow = best is not None and KILL_LINE < best["kd"] <= NARROW_LINE
    ar_narrow = narrow and ar_best is not None and KILL_LINE < ar_best["kd"] <= NARROW_LINE
    return {"outcome": "KILLED", "best": best, "passing": [], "narrow": narrow, "ar_best": ar_best,
            "ar_narrow": ar_narrow, "rows": rows}


def pair_rule(outcomes: dict) -> dict:
    """EXTEND iff one method (the matrix row's method) passes on both models."""
    per_model = [{r.split("_")[0] for r in outcomes[m]["passing"]} for m in MODELS]
    common = sorted(set.intersection(*per_model), key=lambda m: [t[1] for t in MATRIX].index(m))
    return {"extend": bool(common), "methods": common}


def killed_text() -> str:
    return ("W4 killed at the tested post-training recovery set (RTN, BC, VC (folded variance correction), AdaRound, "
            "GPTQ, their BC stacks and MX; pow2 scales, this dialect); QAT untested.")


def c8_coverage(advanced: dict, arms_pinned: dict, gated: set) -> dict:
    """C8's int32 coverage (section 7): PASS iff at least one int32-bias form-b file that advanced was checked, or
    none advanced; each other int32-bias arm is named with what stopped it. gated: (model, arm) pairs with a W4V_GATE
    row. A per-file FAIL is not a coverage failure: it costs that arm its eligibility (eligible())."""
    form_b = [(m, a) for m in MODELS for a in advanced.get(m, [])
              if a.split("_")[0] in INT32_BIAS and arms_pinned[m].get(a) is False]
    checked = [x for x in form_b if x in gated]
    reasons = [f"{m} {a}: {'a form-a conv' if a in advanced.get(m, []) else 'the cut'}"
               for m in MODELS for a in arms() if a.split("_")[0] in INT32_BIAS and (m, a) not in form_b]
    return {"pass": bool(checked) or not form_b, "form_b": form_b, "checked": checked, "reasons": reasons}


def verdict_constants() -> dict:
    return {"MATRIX": [[t, m, list(f)] for t, m, f in MATRIX], "DECIDING_FORMS": DECIDING_FORMS,
            "INT32_BIAS": INT32_BIAS, "MX_METHODS": MX_METHODS, "AR_METHODS": AR_METHODS, "C9_ARMS": C9_ARMS,
            "CUT_LINE": CUT_LINE, "KILL_LINE": KILL_LINE, "NARROW_LINE": NARROW_LINE, "SCREEN_N": SCREEN_N,
            "DECISIVE_N": DECISIVE_N, "H_PIN": {str(k): v for k, v in H_PIN.items()}, "GATE_D_MAP": GATE_D_MAP,
            "EVAL": EVAL, "ORT_VERSION": ORT_VERSION, "FORM_COST": FORM_COST, "METHOD_TEXT": METHOD_TEXT,
            "CODE_FILES": CODE_FILES, "TOOL_FILES": TOOL_FILES}


def verdict_code_sha(root: Path = ROOT) -> str:
    """The whole LF source of this tool and the runner, plus the verdict constants: every line that decides."""
    return sha_bytes(json.dumps({"files": {rel: sha_lf(root / rel) for rel in CODE_FILES},
                                 "constants": verdict_constants()}, sort_keys=True).encode("utf-8"))


def tools_json(root: Path = ROOT) -> dict:
    """The LF shas of the code that makes the arm files: w4_recover.py and the quant/ grid."""
    return {rel: sha_lf(root / rel) for rel in TOOL_FILES}


def hashes(root: Path = ROOT) -> dict:
    return {"PREREG_TEXT_SHA256": sha_bytes(PREREG.encode("utf-8")),
            "PROTOCOL_JSON_SHA256": sha_bytes(json.dumps(protocol(), sort_keys=True).encode("utf-8")),
            "VERDICT_CODE_SHA256": verdict_code_sha(root)}


def frozen_values(root: Path = ROOT) -> dict:
    return {**hashes(root), "TOOLS_JSON": tools_json(root)}


def hash_lines(root: Path = ROOT) -> list[str]:
    v = frozen_values(root)
    return [f"{k} {v[k]}" for k in ("PREREG_TEXT_SHA256", "PROTOCOL_JSON_SHA256", "VERDICT_CODE_SHA256")] + [
        "TOOLS_JSON " + json.dumps(v["TOOLS_JSON"], sort_keys=True)]


def print_hashes() -> None:
    for line in hash_lines():
        print(line, flush=True)


def committed_clean(path, root: Path = ROOT) -> bool:
    """In HEAD and unchanged against it: a file only `git add`ed, or edited after its commit, is not committed."""
    p = Path(path)
    try:
        rel = (p if p.is_absolute() else Path.cwd() / p).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return False

    def ok(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True).returncode == 0
    return ok("cat-file", "-e", f"HEAD:{rel}") and ok("diff", "--quiet", "HEAD", "--", rel)


def live(pattern: str, root: Path = ROOT) -> tuple[list[Path], list[str]]:
    """The results/int4 files a glob matches, minus each SUPERSEDED_V24 name that is committed and clean at exactly
    its listed LF sha (v2.5 E: it stays committed and untouched, and no glob counts it). A listed name at any other
    sha, or not committed and clean, is a problem, so nothing can hide behind the list."""
    found, problems = [], []
    for p in sorted((Path(root) / "results" / "int4").glob(pattern)):
        want = SUPERSEDED_V24.get(p.name)
        if want is None:
            found.append(p)
        elif sha_lf(p) != want or not committed_clean(p, root):
            problems.append(f"{p.name} is listed as superseded at LF {want[:12]}..., but reads {sha_lf(p)[:12]}... "
                            f"or is not committed and clean")
    return found, problems


def frozen_state(root: Path = ROOT) -> list[str]:
    """Problems with the frozen state: the triple and TOOLS_JSON against the ONE committed, clean prereg log's, and
    the text against the accepted sha. Empty means frozen."""
    got = frozen_values(root)
    logs, problems = live("w4v_prereg_*.log", root)
    if got["PREREG_TEXT_SHA256"] != PREREG_SHA:
        problems.append("the embedded text is not the accepted one")
    if len(logs) != 1:
        problems.append(f"{len(logs)} w4v_prereg_*.log files, want exactly one")
    elif not committed_clean(logs[0], root):
        problems.append(f"{logs[0].name} is not committed and clean")
    else:
        want = logged_hashes(logs[0].read_text(encoding="utf-8"))
        problems += [f"{k} differs from {logs[0].name}'s" for k in got if want.get(k) != got[k]]
    say("FROZEN_JSON", {"prereg_logs": [p.name for p in logs], "frozen": not problems})
    return problems


def pins(root: Path = ROOT, allow_none: bool = False) -> dict:
    """basename -> sha256 from the one prereg log and the pins log, each committed and clean (a basename pinned twice
    must agree). allow_none: before any prereg log exists (tier 1's emit), no pins is an empty table."""
    (pre, p1), (extra, p2) = live("w4v_prereg_*.log", root), live("w4v_pins_*.log", root)
    if p1 + p2:
        raise SystemExit(f"pins refused: {'; '.join(p1 + p2)}")
    if allow_none and not pre and not extra:
        return {}
    if len(pre) != 1 or len(extra) > 1:
        raise SystemExit(f"pins refused: {len(pre)} prereg logs (want 1) and {len(extra)} pins logs (want <= 1)")
    for p in pre + extra:
        if not committed_clean(p, root):
            raise SystemExit(f"pins refused: {p.name} is not committed and clean")
    table, clash = {}, []
    for p in pre + extra:
        for sha, path in re.findall(r"^PIN ([0-9a-f]{64})  (\S+)$", p.read_text(encoding="utf-8"), re.M):
            name = Path(path).name
            if table.get(name, sha) != sha:
                clash.append(name)
            table[name] = sha
    if clash:
        raise SystemExit(f"a file is pinned twice with different shas: {sorted(set(clash))}")
    return table


def pinned_arms(pinned: dict) -> dict:
    """model -> {arm: its pinned file is *_ortonly} (twin and repeat files are not arm files)."""
    out = {m: {} for m in MODELS}
    for name in pinned:
        mm = ARM_FILE.match(name)
        if mm:
            if mm.group(2) in out[mm.group(1)]:
                raise SystemExit(f"{mm.group(1)} {mm.group(2)} is pinned as both a form-b and a form-a file")
            out[mm.group(1)][mm.group(2)] = bool(mm.group(3))
    return out


def read_cuts(pinned: dict, root: Path = ROOT, tiers=(1, 2)) -> tuple[dict, list[str]]:
    """The advanced arms, from exactly one committed, clean cut log per tier whose CUT_JSON follows SCREEN COMPLETE,
    and whose advanced plus cut arms are exactly the tier's pinned arms (every pinned arm was screened)."""
    by_model = pinned_arms(pinned)
    advanced, problems = {m: [] for m in MODELS}, []
    for tier in tiers:
        logs = sorted((root / "results" / "int4").glob(f"w4v_cut_tier{tier}_*.log"))
        if len(logs) != 1:
            problems.append(f"{len(logs)} w4v_cut_tier{tier}_*.log files, want exactly one")
            continue
        p = logs[0]
        if not committed_clean(p, root):
            problems.append(f"{p.name} is not committed and clean")
            continue
        text = p.read_text(encoding="utf-8")
        found = re.findall(r"^CUT_JSON (.+)$", text, re.M)
        done, first = re.search(r"^SCREEN COMPLETE$", text, re.M), re.search(r"^CUT_JSON ", text, re.M)
        if len(found) != 1 or not done or done.start() > first.start():
            problems.append(f"{p.name}: not exactly one CUT_JSON after SCREEN COMPLETE")
            continue
        try:
            cut, want = json.loads(found[0]), set(arms(tier))
        except ValueError:
            cut = None
        if not isinstance(cut, dict):
            problems.append(f"{p.name}: CUT_JSON is not a JSON object")
            continue
        if cut.get("tier") != tier:
            problems.append(f"{p.name}: CUT_JSON is tier {cut.get('tier')}")
            continue
        for m in MODELS:
            adv, drop = set(cut["advanced"].get(m, [])), set(cut["cut"].get(m, []))
            if adv & drop or adv | drop != {a for a in by_model[m] if a in want} or adv | drop != want:
                problems.append(f"{p.name}: {m}'s advanced plus cut arms are not its pinned tier-{tier} arms")
                continue
            advanced[m] += [a for a in arms(tier) if a in adv]
    return advanced, problems


STAGES = ("selftest", "controls", "parity", "repro", "emit1", "prereg", "screen1", "emit2", "screen2", "gate",
          "decisive", "verdict")   # w4v-eval.sh's stages, in run order
DETS_LINE = r"^DETS_SHA256 [0-9a-f]{64} \S+$"
SRC_TREE_LINE = r"^SRC_TREE [0-9a-f]{40,64}$"
CONTROL_FAIL = r"^W4V_CONTROL \S+ \S+( \S+)? FAIL"   # one scope token, or C5b's two (v2.5 C: <model> <arm>)
C9_TAG = "C9_REDERIVED"   # --verdict's C9 lines, re-derived from the repro rows (v2.5 E); never a W4V_CONTROL line
OWN_EXTRA = {   # the logs a stage writes beyond stage_logs' (report-only ones, the load witnesses, every decisive row)
    "parity": ["w4v_parity_c6c_env17_yolov8n_*.log"],
    "repro": ["load_w4v_repro_*.log"],
    "screen1": ["load_w4v_screen1_*.log"],
    "screen2": ["load_w4v_screen2_*.log"],
    "decisive": ["w4v_decisive_*.log", "load_w4v_decisive_*.log"],
    "verdict": ["w4v_verdict_*.log"],
}


def stage_logs(stage: str, advanced: dict | None = None) -> list[tuple[str, list[tuple[str, int]]]]:
    """The logs a stage leaves in results/int4/: a glob that must match exactly one file, and the success lines
    (pattern, least count) that file must carry. The decisive rows follow the cut (advanced)."""
    def rows(kind, n, by_model):
        return [(f"w4v_{kind}_{m}_{a}_n{n}_*.log", [(DETS_LINE, 1)]) for m in MODELS for a in by_model[m]]
    if stage == "selftest":
        return [("w4v_selftest_*.log", [(r"^\[selftest\] PASS$", 2)])]
    if stage == "controls":
        return [(f"w4v_controls_{m}_*.log", [(rf"^\[controls\] {m} PASS$", 1)]) for m in MODELS]
    if stage == "parity":
        return ([(f"w4v_parity_c6a_{m}_*.log", [(rf"^W4V_CONTROL C6a {m} PASS", 1)]) for m in MODELS]
                + [(f"w4v_parity_{c}_yolov8n_*.log", [(rf"^W4V_CONTROL {name} yolov8n PASS", 1)])
                   for c, name in (("c6b", "C6b"), ("c6c", "C6c"))])
    if stage == "repro":
        return (rows("repro", DECISIVE_N, {m: ("b8",) + C9_ARMS for m in MODELS})
                + [("w4v_repro_c9_*.log", [(rf"^W4V_CONTROL C9 {m} PASS", 1) for m in MODELS])])
    if stage == "emit1":
        return [(f"w4v_{k}_{m}_*.log", [(rf"^W4V_DONE {k} {m}$", 1)])
                for m in MODELS for k in ("sensitivity", "emit1")]
    if stage == "prereg":
        return [("w4v_prereg_*.log", [(r"^PINNED \d+ files$", 1)])]
    if stage == "emit2":
        return ([(f"w4v_emit2_{m}_*.log", [(rf"^W4V_DONE emit2 {m}$", 1)]) for m in MODELS]
                + [("w4v_pins_*.log", [(r"^PINNED \d+ files$", 1)])])
    if stage in ("screen1", "screen2"):
        tier = int(stage[-1])
        return (rows("screen", SCREEN_N, {m: (["b8"] if tier == 1 else []) + arms(tier) for m in MODELS})
                + [(f"w4v_cut_tier{tier}_*.log", [(r"^SCREEN COMPLETE$", 1)])])
    if stage == "gate":
        return [("w4v_gate_*.log", [(SRC_TREE_LINE, 1), (r"^W4V_CONTROL C8 int32 PASS", 1), (r"^W4V_DONE gate$", 1)])]
    if stage == "decisive":
        return rows("decisive", DECISIVE_N, {m: ["b8"] + [a for a in (advanced or {}).get(m, []) if a not in C9_ARMS]
                                             for m in MODELS})
    return []


def carry_global(root: Path = ROOT) -> list[str]:
    """v2.5 E rules 3 and 4, which every carried log needs: each file of v2.4's TOOLS_JSON at the same LF sha in
    v2.5's (v2.5 may add files), and v2.5's protocol JSON minus its one added key equal to v2.4's."""
    now = tools_json(root)
    out = [f"rule 3: {rel} is not at v2.4's LF sha {s[:12]}..." for rel, s in sorted(V24_FROZEN["TOOLS_JSON"].items())
           if now.get(rel) != s]
    rest = {k: v for k, v in protocol().items() if k != "v2_5"}
    if sha_bytes(json.dumps(rest, sort_keys=True).encode("utf-8")) != V24_FROZEN["PROTOCOL_JSON_SHA256"]:
        out.append("rule 4: the protocol JSON minus v2_5 is not v2.4's")
    return out


def carry_problems(path: Path, success: list[tuple[str, int]], root: Path = ROOT) -> list[str]:
    """v2.5 E: a CARRIED_V24 log stands in for its v2.5 re-run iff (1) it is listed and reads its listed LF sha,
    (2) every hash block in it equals V24_FROZEN, (3, 4) carry_global() holds, and (5) it is committed and clean,
    carries its success lines and holds no W4V_CONTROL FAIL line."""
    out = []
    want = CARRIED_V24.get(path.name)
    if want is None or sha_lf(path) != want:
        out.append(f"rule 1: not carried at its listed LF sha ({(want or 'unlisted')[:12]}...)")
    if not committed_clean(path, root):
        out.append("rule 5: is not committed and clean")
    text = Path(path).read_text(encoding="utf-8")
    for pattern, least in success:
        if len(re.findall(pattern, text, re.M)) < least:
            out.append(f"rule 5: lacks its success line {pattern}" + (f" ({least} wanted)" if least > 1 else ""))
    if re.search(CONTROL_FAIL, text, re.M):
        out.append("rule 5: holds a W4V_CONTROL FAIL line")
    blocks = hash_blocks(text)
    if not blocks or any(b != V24_FROZEN for b in blocks):
        out.append(f"rule 2: {sum(b != V24_FROZEN for b in blocks) if blocks else 'no'} of {len(blocks)} hash blocks "
                   f"unlike V24_FROZEN")
    return out + carry_global(root)


def log_problems(path: Path, success: list[tuple[str, int]], want: dict, root: Path = ROOT,
                 need_commit: bool = True) -> list[str]:
    """One stage log: committed and clean, its success lines, no W4V_CONTROL FAIL line, and at least one hash
    block, every one equal to want (the frozen values). A CARRIED_V24 log is judged by carry_problems() instead."""
    if Path(path).name in CARRIED_V24:
        return carry_problems(Path(path), success, root)
    out = []
    if need_commit and not committed_clean(path, root):
        out.append("is not committed and clean")
    text = Path(path).read_text(encoding="utf-8")
    for pattern, least in success:
        if len(re.findall(pattern, text, re.M)) < least:
            out.append(f"lacks its success line {pattern}" + (f" ({least} wanted)" if least > 1 else ""))
    if re.search(CONTROL_FAIL, text, re.M):
        out.append("holds a W4V_CONTROL FAIL line")
    blocks = hash_blocks(text)
    if not blocks:
        out.append("prints no hash block")
    elif any(b != want for b in blocks):
        out.append(f"has {sum(b != want for b in blocks)} of {len(blocks)} hash blocks unlike the frozen values")
    return out


def zerodet_globs(stage: str) -> list[str]:
    """The zero-detection records a stage writes. Their names carry no stage, so each is matched by its arm and n:
    no arm is scored at n500 in both screens (tier 1 and tier 2 are disjoint), and n5000 rows are only decisive's."""
    if stage in ("screen1", "screen2"):
        return [f"w4v_zerodet_{m}_{a}_n{SCREEN_N}_*.log" for m in MODELS for a in arms(int(stage[-1]))]
    return [f"w4v_zerodet_*_n{DECISIVE_N}_*.log"] if stage == "decisive" else []


def own_logs(stage: str, root: Path = ROOT) -> list[Path]:
    """Every log the stage itself has written already, of any date, committed or not. A superseded v2.4 log at its
    listed sha is not the stage's; at any other sha it is (so the stage refuses to start)."""
    results = root / "results" / "int4"
    found = set()
    for g in [g for g, _ in stage_logs(stage)] + OWN_EXTRA.get(stage, []) + zerodet_globs(stage):
        for p in results.glob(g):
            if p.name not in SUPERSEDED_V24 or live(p.name, root)[1]:
                found.add(p)
    return sorted(found)


def ready(stage: str, root: Path = ROOT, want: dict | None = None, starting: bool = False) -> list[str]:
    """May this stage run? Every earlier stage's logs: exactly one per glob, committed and clean, with their
    success lines, no W4V_CONTROL FAIL, and hash blocks equal to the frozen values. Starting (the runner's --ready,
    before the stage writes anything): also no log of the stage itself yet. The in-stage checks (--c9, --screen,
    --c8, --prereg, --pins, --verdict) run inside their stage, once its --ready has passed, so they leave starting
    off. Empty means yes."""
    want = frozen_values(root) if want is None else want
    before = STAGES[:STAGES.index(stage)]
    problems, advanced = [], None
    if starting:
        problems += [f"{stage}: {p.name} exists already; a restart needs the gate's ruling"
                     for p in own_logs(stage, root)]
    if "decisive" in before:
        try:
            advanced, cut_problems = read_cuts(pins(root), root)
        except SystemExit as e:
            advanced, cut_problems = {}, [str(e)]
        problems += cut_problems
    for s in before:
        for pattern, success in stage_logs(s, advanced):
            found, sup = live(pattern, root)
            problems += [f"{s}: {p}" for p in sup]
            if len(found) != 1:
                problems.append(f"{s}: {len(found)} {pattern}, want exactly one")
                continue
            got = log_problems(found[0], success, want, root)
            if (DETS_LINE, 1) in success and any(DETS_LINE in p for p in got):   # a zero-detection row (v2.5 B.4)
                _, rec = zero_record(found[0], root, want=want)
                if rec:
                    got += [f"zero-detection record: {p}" for p in rec]
                else:
                    got = [p for p in got if DETS_LINE not in p]
            problems += [f"{s}: {found[0].name} {p}" for p in got]
    return problems


def src_tree(root: Path = ROOT) -> tuple[str, list[str]]:
    """The tree HEAD records for src/ (the compiler the engine gate imports; not in TOOLS_JSON), and why src/ is not
    clean against it: staged, unstaged or untracked (not ignored) paths."""
    def run(*a):
        return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True)
    tree = run("rev-parse", "HEAD:src")
    if tree.returncode:
        return "none", ["src/ is not in HEAD"]
    status = run("status", "--porcelain", "--untracked-files=all", "--", "src")
    if status.returncode:
        return tree.stdout.strip(), ["git status failed on src/"]
    dirty = [line for line in status.stdout.splitlines() if line.strip()]
    return tree.stdout.strip(), [f"src/ is not clean against HEAD: {len(dirty)} paths, e.g. {dirty[0].strip()}"] \
        if dirty else []


class Data:
    """The evaluation context: the annotation file, the sorted ids and coco_calib's ids."""

    def __init__(self, ann: Path, calib_dir: Path):
        from pycocotools.coco import COCO
        with contextlib.redirect_stdout(io.StringIO()):
            self.coco = COCO(str(ann))
        self.ann_sha = sha_file(ann)
        self.ids = sorted(self.coco.getImgIds())
        self.calib_ids = {int(p.stem) for p in Path(calib_dir).glob("*.jpg")}

    def sets(self, n: int) -> tuple[list[int], list[int]]:
        ids = self.ids[:n] if n else list(self.ids)
        return ids, h_ids(ids, self.calib_ids)


def screen_table(rows: list[dict], data, expected: dict, root: Path) -> dict:
    """The cut on the 500, H_500 beside it (report-only) and the flag. expected: model -> arms of this tier."""
    problems, table, advanced, cut, flags = [], [], {}, {}, {}
    for model in MODELS:
        b8 = [r for r in rows if r["model"] == model and r["arm"] == "b8"]
        if len(b8) != 1:
            problems.append(f"{model}: {len(b8)} B8 rows, want 1")
            continue
        ids, h = data.sets(SCREEN_N)
        base = dets_check(b8[0], data.coco, ids, h, root)
        if not base["ok"]:
            problems.append(f"{model} b8: dets check {base}")
            continue
        advanced[model], cut[model], flags[model] = [], [], []
        for arm in expected[model]:
            found = [r for r in rows if r["model"] == model and r["arm"] == arm]
            if len(found) != 1:
                problems.append(f"{model} {arm}: {len(found)} rows, want 1")
                continue
            r = found[0]
            chk = row_check(r, data.coco, ids, h, root)
            if not chk["ok"]:
                problems.append(f"{model} {arm}: dets check {chk}")
                continue
            d500, dh = delta(b8[0]["map"], r["map"]), delta(base["h"], chk["h"])
            adv, flag = advances(d500), flagged(d500, dh)
            table.append({"model": model, "arm": arm, "map": r["map"], "h": chk["h"], "d500": d500, "dh": dh,
                          "advances": adv, "flag": flag})
            (advanced if adv else cut)[model].append(arm)
            if flag:
                flags[model].append(arm)
        table.append({"model": model, "arm": "b8", "map": b8[0]["map"], "h": base["h"]})
    for r in rows:
        if r["images"] != SCREEN_N or r["skipped"]:
            problems.append(f"{r['log']}: {r['images']} images, {r['skipped']} skipped")
    return {"table": table, "advanced": advanced, "cut": cut, "flags": flags, "problems": problems}


def screen_lines(res: dict, problems: list[str], tier: int, expected: dict) -> list[str]:
    """What follows the table: the PROBLEM lines and SCREEN INCOMPLETE, or SCREEN COMPLETE and only then the cut."""
    if problems:
        return [f"PROBLEM {p}" for p in problems] + ["SCREEN INCOMPLETE"]
    out = ["SCREEN COMPLETE",
           "CUT_JSON " + json.dumps({"tier": tier, "advanced": res["advanced"], "cut": res["cut"]}, sort_keys=True),
           "FLAG_JSON " + json.dumps({"tier": tier, "flags": res["flags"], "meaning": "report-only: cut on the 500 "
                                      "but within the cut line on H_500; decides nothing"}, sort_keys=True)]
    return out + [f"{m}: {len(res['advanced'][m])} of {len(expected[m])} tier-{tier} arms advance; "
                  f"{len(res['flags'][m])} flagged" for m in MODELS]


def verdict_table(rows: list[dict], data, advanced: dict, gate: dict, arms_pinned: dict, root: Path) -> dict:
    """K-D per model on all 5,000 and on H. rows: repro and decisive 5,000-image rows. advanced: model -> arms the
    screens advanced. gate: arm file sha -> its C8 PASS or FAIL. arms_pinned: model -> {arm: file is *_ortonly}."""
    problems, outcomes, c9 = [], {}, {}
    ids, h = data.sets(0)
    if len(h) != H_PIN[DECISIVE_N][0] or id_text_sha(h) != H_PIN[DECISIVE_N][1]:
        problems.append(f"H holds {len(h)} ids (sha {id_text_sha(h)}), not its pinned {H_PIN[DECISIVE_N]}")
    for r in rows:
        if r["images"] != DECISIVE_N or r["skipped"]:
            problems.append(f"{r['log']}: {r['images']} images, {r['skipped']} skipped")
    for model in MODELS:
        def one(stage, arm):
            found = [r for r in rows if r["model"] == model and r["stage"] == stage and r["arm"] == arm]
            if len(found) != 1:
                problems.append(f"{model} {stage} {arm}: {len(found)} rows, want 1")
                return None
            return found[0]
        repro = {a: one("repro", a) for a in ("b8",) + C9_ARMS}
        c9[model] = {a: None if r is None else r["map"] for a, r in repro.items()}
        for a, want in GATE_D_MAP[model].items():
            if repro.get(a) is not None and repro[a]["map"] != want:
                problems.append(f"C9 {model} {a}: {repro[a]['map']} != gate D's {want}")
        b8 = one("decisive", "b8")
        if b8 is None:
            continue
        if b8["map"] != GATE_D_MAP[model]["b8"]:
            problems.append(f"{model}: the decisive B8 reads {b8['map']}, not {GATE_D_MAP[model]['b8']}")
        base = dets_check(b8, data.coco, ids, h, root)
        if not base["ok"]:
            problems.append(f"{model} decisive b8: dets check {base}")
            continue
        wanted = [(a, "repro") for a in C9_ARMS] + [(a, "decisive") for a in advanced.get(model, [])
                                                     if a not in C9_ARMS]
        cands = []
        for arm, stage in wanted:
            r = one(stage, arm)
            if r is None:
                continue
            chk = row_check(r, data.coco, ids, h, root)
            if not chk["ok"]:
                problems.append(f"{model} {arm}: dets check {chk}")
                continue
            orto = arms_pinned[model].get(arm)
            if orto is None:
                problems.append(f"{model} {arm}: no pinned file")
                continue
            g = gate.get(r["sha256"])
            if not orto and g is None and arm in advanced.get(model, []):
                problems.append(f"{model} {arm}: a form-b arm that advanced has no C8 row")
            cands.append({"arm": arm, "map": r["map"], "h": chk["h"], "d_all": delta(b8["map"], r["map"]),
                          "d_h": delta(base["h"], chk["h"]), "ortonly": orto, "gate": g,
                          "eligible": eligible(arm, orto, g)})
        outcomes[model] = {**model_outcome(cands), "b8": b8["map"], "b8_h": base["h"]}
    for field in ("threads", "ort"):
        seen = {r[field] for r in rows}
        if len(seen) > 1:
            problems.append(f"{field} differs between rows: {sorted(seen)}")
    return {"outcomes": outcomes, "c9": c9, "problems": problems,
            "pair": pair_rule(outcomes) if len(outcomes) == len(MODELS) else None}


def load_rows(paths: list[str], need_commit: bool = False, root: Path = ROOT) -> list[dict]:
    """The eval rows; one with no DETS_SHA256 line is read through its zero-detection record (v2.5 B.4), committed
    and clean when need_commit (after its stage; --screen reads the stage's own, not yet committed)."""
    out = []
    for p in map(Path, paths):
        if re.search(DETS_LINE, p.read_text(encoding="utf-8"), re.M):
            out.append(parse_eval_log(p))
            continue
        row, problems = zero_record(p, root, want=frozen_values(root), need_commit=need_commit)
        if problems:
            raise ValueError(f"{p.name}: no DETS_SHA256 line, and not a recorded zero-detection row: "
                             + "; ".join(problems))
        out.append(row)
    return out


def row_check(row: dict, coco, ids: list[int], h: list[int], root: Path) -> dict:
    """dets_check, or for a recorded zero-detection row its figures by rule B.2 (nothing is rescored)."""
    return {"ok": True, "zero": True, "h": "0.00"} if row.get("zero") else dets_check(row, coco, ids, h, root)


def common_problems(rows: list[dict], pinned: dict) -> list[str]:
    problems = []
    want = frozen_values()
    by_sha = {}
    for name, sha in pinned.items():
        by_sha.setdefault(sha, []).append(name)
    for r in rows:
        if r["ort"] != ORT_VERSION or r["ort_opt"] != EVAL["ort_opt"] or r["threads"] != EVAL["threads"]:
            problems.append(f"{r['log']}: ORT {r['ort']}, ort-opt {r['ort_opt']}, threads {r['threads']}")
        if r["triple"] != (V24_FROZEN if r["log"] in CARRIED_V24 else want):   # a carried row holds v2.4's (E rule 2)
            problems.append(f"{r['log']}: its frozen triple or TOOLS_JSON differs from this tool's"
                            + (" (v2.4's, as carried)" if r["log"] in CARRIED_V24 else ""))
        names = set(by_sha.get(r["sha256"], []))
        wanted = ({f"{r['model']}_cut_xint8.onnx"} if r["arm"] == "b8" else
                  {f"{r['model']}_w4v_{r['arm']}.onnx", f"{r['model']}_w4v_{r['arm']}_ortonly.onnx"})
        if not names & wanted:
            problems.append(f"{r['log']}: model sha {r['sha256'][:12]} is not the pinned file of {r['arm']}")
    return problems


def controls_status(paths: list[str]) -> dict:
    """W4V_CONTROL <name> <scope> <PASS|FAIL> markers (a W4V_REPORT line is report-only and not read)."""
    marks = {}
    for p in paths:
        for name, scope, status in re.findall(r"^W4V_CONTROL (\S+) (\S+) (PASS|FAIL)\b", Path(p).read_text(
                encoding="utf-8"), re.M):
            marks.setdefault((name, scope), []).append(status)
    return marks


def gate_status_text(text: str) -> dict:
    """arm file sha -> PASS or FAIL from the W4V_GATE lines (any FAIL for a sha is its result)."""
    gate = {}
    for sha, status in re.findall(r"^W4V_GATE ([0-9a-f]{64}) (PASS|FAIL)$", text, re.M):
        gate[sha] = "FAIL" if "FAIL" in (gate.get(sha), status) else "PASS"
    return gate


def gate_status(paths: list[str]) -> dict:
    return gate_status_text("\n".join(Path(p).read_text(encoding="utf-8") for p in paths))


def required_controls() -> list[tuple[str, str]]:
    req = []
    for m in MODELS:
        req += [("C0", m), ("ORDER", m), ("MINMSE", m), ("C1", m), ("C2", m), ("C3", f"{m}_rtn_e1"),
                ("C3", f"{m}_ar_e1"), ("C3", f"{m}_gptq_e1"), ("C4", m), ("C6a", m), ("C9", m)]
        req += [("C5", f"{m}_{a}") for a in arms()]
    req += [("C6b", "yolov8n"), ("C6c", "yolov8n"), ("C8", "int32")]
    req += [("C7", f"yolov8n_{a}") for a in ("gptq_e1", "bc_e1", "vc_e1", "ar_e1")]
    return req


def v24_pins(root: Path = ROOT) -> tuple[dict, list[str]]:
    """The superseded v2.4 prereg log's PIN lines (path -> sha256), read only while it is committed and clean at its
    listed LF sha."""
    name = next(n for n in SUPERSEDED_V24 if n.startswith("w4v_prereg_"))
    p = Path(root) / "results" / "int4" / name
    if not p.is_file() or sha_lf(p) != SUPERSEDED_V24[name] or not committed_clean(p, root):
        return {}, [f"{name}: not committed and clean at its listed LF sha, so its pins cannot be read"]
    return ({path: sha for sha, path in re.findall(r"^PIN ([0-9a-f]{64})  (\S+)$", p.read_text(encoding="utf-8"),
                                                    re.M)}, [])


def v24_pins_problems(files: list[str], root: Path = ROOT) -> list[str]:
    """v2.5 E: the v2.5 prereg pins exactly the v2.4 prereg's files, at equal shas, or refuses."""
    old, problems = v24_pins(root)
    new = {shown(f): sha_file(f) for f in files}
    print(f"V24_PINS_EQUAL {old == new} (v2.4 prereg {len(old)} files; this prereg {len(new)})", flush=True)
    if not problems and old != new:
        problems.append("the v2.5 prereg does not pin exactly the v2.4 prereg's files at equal shas")
    return problems


def refuse(problems: list[str], what: str) -> int:
    for p in problems:
        print(f"PROBLEM {p}")
    print(what)
    return 2


# ------------------------------------------------------------------------------------------------------- the CLI

def cmd_prereg(args, tier2: bool) -> int:
    """--prereg (tier 1, no prereg log exists yet) or --pins (tier 2, against the committed prereg log)."""
    import onnxruntime as ort
    if not tier2:
        print(PREREG)
        say("PROTOCOL_JSON", protocol())
    print_hashes()
    same = hashes()["PREREG_TEXT_SHA256"] == PREREG_SHA
    print(f"PREREG_TEXT_EQUALS_ACCEPTED {same} (the accepted text's LF sha256 {PREREG_SHA})", flush=True)
    problems = [] if same else ["the embedded text is not the accepted one"]
    if tier2:
        problems += frozen_state()
        # every stage before emit2 committed as ready() wants; the emit2 logs, from this same stage, are not
        # committed yet: their success lines and hash blocks are checked here, their commit by screen2's ready()
        problems += ready("emit2")
        want = frozen_values()
        for pattern, success in [x for x in stage_logs("emit2") if x[0].startswith("w4v_emit2_")]:
            found = sorted(RESULTS.glob(pattern))
            if len(found) != 1:
                problems.append(f"emit2: {len(found)} {pattern}, want exactly one")
                continue
            problems += [f"emit2: {found[0].name} {p}"
                         for p in log_problems(found[0], success, want, need_commit=False)]
    else:
        # every earlier stage's committed logs (controls, parity, repro, sensitivity, emit1 and the selftest):
        # their success lines, and every hash block equal to this tool's
        problems += ready("prereg")
        existing, sup = live("w4v_prereg_*.log")
        problems += sup
        if existing:
            problems.append(f"a prereg log exists already: {[p.name for p in existing]}")
        problems += v24_pins_problems(args.files)
        for m, files in MODEL_PIN.items():
            for kind, (path, want) in files.items():
                got = sha_file(ROOT / path)
                print(f"MODEL_PIN {m} {kind} {path} sha256 {got} {'== pinned' if got == want else 'MISMATCH'}")
                if got != want:
                    problems.append(f"{path} is not the pinned {kind} model")
    for rel in CODE_FILES + TOOL_FILES:
        ok = committed_clean(ROOT / rel)
        print(f"COMMITTED {'OK' if ok else 'REFUSED'} {rel}")
        if not ok:
            problems.append(f"{rel} is not committed and clean")
    if problems:
        return refuse(problems, "PINS REFUSED")
    head = git("rev-parse", "HEAD")
    dirty = [line for line in git("status", "--porcelain").splitlines() if line]
    print(f"git HEAD {head} (+{len(dirty)} uncommitted paths: this log's own commit)")
    print(f"onnxruntime {ort.__version__}")
    for f in args.files:
        print(f"PIN {sha_file(f)}  {shown(f)}")
    print(f"PINNED {len(args.files)} files")
    return 0


def cmd_pinned(path: str) -> int:
    table = pins()
    name, sha = Path(path).name, sha_file(path)
    ok = table.get(name) == sha
    print(f"PINNED {'OK' if ok else 'REFUSED'} {shown(path)} sha256 {sha}"
          f"{'' if ok else f' (pinned: {table.get(name)})'}", flush=True)
    return 0 if ok else 1


def cmd_ready(stage: str) -> int:
    """For the runner, before a stage writes anything: every earlier stage's logs as ready() wants them, and no log of
    the stage itself yet."""
    problems = ready(stage, starting=True)
    if problems:
        return refuse(problems, f"NOT READY {stage}: an earlier stage's logs do not hold, or the stage has run "
                                f"already; nothing may run")
    print(f"READY {stage}: every earlier stage's logs are committed and clean, carry their success lines, and "
          f"hold only hash blocks equal to this tool's; the stage has no log yet")
    return 0


def cmd_src_tree() -> int:
    """For the gate stage, before the engine gate runs: the compiler C8 judges with, as HEAD records it."""
    tree, problems = src_tree()
    print(f"SRC_TREE {tree}", flush=True)
    if problems:
        return refuse(problems, "SRC_TREE REFUSED: no commit records the compiler C8 would judge with")
    return 0


def c9_setup(r: dict, want_now: dict) -> bool:
    """C9's setup check on one repro row: all 5,000 images, none skipped, the protocol's ORT, ort-opt and threads,
    and the frozen triple with TOOLS_JSON (v2.4's when the row is carried, E rule 2; this tool's otherwise)."""
    return (r["images"] == DECISIVE_N and not r["skipped"] and r["ort"] == ORT_VERSION
            and r["ort_opt"] == EVAL["ort_opt"] and r["threads"] == EVAL["threads"]
            and r["triple"] == (V24_FROZEN if r["log"] in CARRIED_V24 else want_now))


def c9_rederive(rows: list[dict], data, root: Path = ROOT, tag: str = "C9") -> tuple[str, dict]:
    """C9's comparison, for --c9 and for v2.5's --verdict, which re-derives it from the carried repro rows (section E:
    their C9 PASS line was printed by v2.4's code): each gate D row's printed figure against gate D's, its setup
    against the protocol (its triple v2.4's when the row is carried, E rule 2) and its dets check with its H figure.
    The annotations line, and model -> (its row lines, PASS), every line led by tag."""
    ids, h = data.sets(0)
    h_ok = (len(h), id_text_sha(h)) == H_PIN[DECISIVE_N]
    ok = data.ann_sha == ANN["sha256"] and h_ok
    head = (f"{tag} annotations {'as pinned' if data.ann_sha == ANN['sha256'] else 'NOT the pinned file'}; H holds "
            f"{len(h)} ids, {'as pinned' if h_ok else 'NOT as pinned'}")
    want_now = frozen_values(root)
    out = {}
    for m in MODELS:
        good, lines = ok, []
        for a, want in GATE_D_MAP[m].items():
            found = [r for r in rows if r["model"] == m and r["arm"] == a and r["stage"] == "repro"]
            if len(found) != 1:
                lines.append(f"{tag} {m} {a}: {len(found)} rows, want 1")
                good = False
                continue
            r = found[0]
            setup = c9_setup(r, want_now)
            same = r["map"] == want
            chk = dets_check(r, data.coco, ids, h, root)
            lines.append(f"{tag} {m} {a}: {'matches' if same else 'DOES NOT MATCH'} gate D's {DECISIVE_N}-image "
                         f"figure; setup {'as pre-registered' if setup else 'DIFFERS'}; dets check "
                         + (f"ok, H ({len(h)} images) {chk['h']}" if chk["ok"] else "FAILED, no H figure")
                         + f" ({r['log']})")
            good &= same and setup and chk["ok"]
        out[m] = (lines, good)
    return head, out


def cmd_c9(args) -> int:
    """C9 at the end of the repro stage: each row's printed figure against gate D's, its setup against the protocol,
    and its H figure (section 7: RTN's decisive figures come from here). A mismatch is a STOP; --verdict re-derives
    it from the same logs (c9_rederive()). It runs before the prereg log exists, so it reads no pins."""
    print_hashes()
    problems = ready("repro")
    if problems:
        return refuse(problems, "C9 REFUSED: an earlier stage's logs do not hold; nothing is scored")
    rows = load_rows(args.logs)
    head, c9 = c9_rederive(rows, Data(ROOT / ANN["path"], ROOT / CALIB["dir"]))
    print(head)
    for m, (lines, good) in c9.items():
        for line in lines:
            print(line)
        print(f"W4V_CONTROL C9 {m} {'PASS' if good else 'FAIL'}")
    return 0 if all(good for _, good in c9.values()) else 1


def cmd_screen(args) -> int:
    print_hashes()
    problems = frozen_state() + ready(f"screen{args.tier}")
    if problems:
        return refuse(problems, "SCREEN REFUSED: the frozen state or an earlier stage's logs do not hold; nothing is "
                                "scored")
    try:
        rows = load_rows(args.logs)
    except ValueError as e:
        return refuse([str(e)], "SCREEN REFUSED: a row log is neither an eval nor a recorded zero-detection row; "
                                "nothing is scored")
    pinned = pins()
    tier = args.tier
    expected = {m: arms(tier) for m in MODELS}
    data = Data(ROOT / ANN["path"], ROOT / CALIB["dir"])
    ids, h = data.sets(SCREEN_N)
    problems = common_problems(rows, pinned)
    if data.ann_sha != ANN["sha256"]:
        problems.append(f"annotation sha {data.ann_sha} != pinned {ANN['sha256']}")
    if len(h) != H_PIN[SCREEN_N][0] or id_text_sha(h) != H_PIN[SCREEN_N][1]:
        problems.append(f"H_500 holds {len(h)} ids, not its pinned {H_PIN[SCREEN_N]}")
    if problems:
        return refuse(problems, "SCREEN REFUSED: the rows do not meet the pre-registered setup; nothing is scored")
    res = screen_table(rows, data, expected, ROOT)
    print(f"{'model':8s} {'arm':12s} {'mAP500':>7s} {'d500':>6s} {'H468':>6s} {'dH':>6s}  advance  flag")
    for t in res["table"]:
        if t["arm"] == "b8":
            print(f"{t['model']:8s} {'b8':12s} {t['map']:>7s} {'':>6s} {t['h']:>6s}")
        else:
            print(f"{t['model']:8s} {t['arm']:12s} {t['map']:>7s} {t['d500']:>6.2f} {t['h']:>6s} {t['dh']:>6.2f}  "
                  f"{'yes' if t['advances'] else 'no':7s}  {'FLAG' if t['flag'] else ''}")
    for line in screen_lines(res, res["problems"], tier, expected):
        print(line)
    return 2 if res["problems"] else 0


def cmd_advanced() -> int:
    """For the runner: the arms the committed cut logs advance, as ADVANCED <model> <arm>."""
    print_hashes()
    problems = frozen_state()
    advanced = {}
    if not problems:
        advanced, problems = read_cuts(pins())
    if problems:
        return refuse(problems, "ADVANCED REFUSED")
    for m in MODELS:
        for a in advanced[m]:
            print(f"ADVANCED {m} {a}")
    return 0


def cmd_c8(args) -> int:
    """C8's int32 coverage marker, from the engine gate's W4V_GATE lines and the committed cut logs."""
    print_hashes()
    problems = frozen_state() + ready("gate") + src_tree()[1]
    if problems:
        return refuse(problems, "C8 REFUSED: the frozen state, an earlier stage's logs or src/ do not hold")
    pinned = pins()
    advanced, cut_problems = read_cuts(pinned)
    problems += cut_problems
    gate = gate_status(args.logs)
    by_sha = {}
    for name, sha in pinned.items():
        mm = ARM_FILE.match(name)
        if mm:
            by_sha[sha] = (mm.group(1), mm.group(2))
    problems += [f"W4V_GATE {s[:12]} is no pinned arm file" for s in gate if s not in by_sha]
    cov = c8_coverage(advanced, pinned_arms(pinned), {by_sha[s] for s in gate if s in by_sha})
    print(f"[C8] int32-bias form-b arms that advanced: {', '.join(f'{m} {a}' for m, a in cov['form_b']) or 'none'}; "
          f"checked: {', '.join(f'{m} {a}' for m, a in cov['checked']) or 'none'}")
    for r in cov["reasons"]:
        print(f"[C8] not a checked int32-bias form-b arm: {r}")
    for p in problems:
        print(f"PROBLEM {p}")
    ok = cov["pass"] and not problems
    print(f"W4V_CONTROL C8 int32 {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def cmd_verdict(args) -> int:
    print_hashes()
    problems = frozen_state() + ready("verdict")
    problems += [f"{Path(p).name} is not committed and clean" for p in args.logs + args.controls
                 if not committed_clean(p)]
    if problems:
        return refuse(problems, "VERDICT REFUSED: the frozen state or the committed logs do not hold; nothing is "
                                "scored")
    pinned = pins()
    advanced, problems = read_cuts(pinned)
    try:
        rows = load_rows(args.logs, need_commit=True)
    except ValueError as e:
        return refuse(problems + [str(e)], "VERDICT REFUSED: a row log is neither an eval nor a recorded "
                                           "zero-detection row; nothing is scored")
    problems += common_problems(rows, pinned)
    marks = controls_status(args.controls)
    for key in required_controls():
        st = marks.get(key, [])
        if not st or "FAIL" in st:
            problems.append(f"control {key[0]} {key[1]}: {st or 'missing'}")
    problems += [f"control {k[0]} {k[1]} FAIL" for k, st in marks.items() if "FAIL" in st and k not in
                 required_controls()]
    data = Data(ROOT / ANN["path"], ROOT / CALIB["dir"])
    if data.ann_sha != ANN["sha256"]:
        problems.append(f"annotation sha {data.ann_sha} != pinned {ANN['sha256']}")
    head, c9 = c9_rederive(rows, data, ROOT, C9_TAG)   # v2.5 E: C9 from the carried rows, as --c9 made it
    print(head)
    for m, (lines, good) in c9.items():
        for line in lines:
            print(line)
        print(f"{C9_TAG} {m} {'PASS' if good else 'FAIL'}")
        if not good:
            problems.append(f"C9 re-derived from the repro rows: {m} FAIL")
    if problems:
        return refuse(problems, "VERDICT REFUSED: the rows, the cut logs, the controls or the re-derived C9 do not "
                                "meet the pre-registered setup; nothing is scored")
    res = verdict_table(rows, data, advanced, gate_status(args.controls), pinned_arms(pinned), ROOT)
    say("C9_JSON", res["c9"])
    for m, o in res["outcomes"].items():
        print(f"\n{m}: B8 {o['b8']} on all {DECISIVE_N}, {o['b8_h']} on H ({H_PIN[DECISIVE_N][0]})")
        print(f"  {'arm':12s} {'mAP':>6s} {'H':>6s} {'d_all':>6s} {'d_H':>6s} {'K-D':>6s}  eligible  pass")
        for r in sorted(o["rows"], key=lambda r: row_order(r["arm"])):
            print(f"  {r['arm']:12s} {r['map']:>6s} {r['h']:>6s} {r['d_all']:>6.2f} {r['d_h']:>6.2f} {r['kd']:>6.2f}  "
                  f"{'yes' if r['eligible'] else 'no':8s}  {'yes' if r['pass'] else 'no'}")
        for r in sorted(o["rows"], key=lambda r: row_order(r["arm"])):
            if cut_no_c8(r, advanced.get(m, [])):
                print(f"INFO {m} {r['arm']}: not decision-eligible: cut, no C8 row (report-only; decides nothing)")
    if res["problems"]:
        print()
        return refuse(res["problems"], "VERDICT INCOMPLETE: the rows above do not meet the pre-registered setup")
    print()
    for m, o in res["outcomes"].items():
        b = o["best"]
        if o["outcome"] == "PASS":
            method, form = b["arm"].split("_")
            print(f"VERDICT {m} PASS: {METHOD_TEXT[method]} {FORM_TAG[form]} ({b['arm']}), delta {b['d_all']:.2f} on "
                  f"all {DECISIVE_N} and {b['d_h']:.2f} on H; phase 2: {phase2_cost(b['arm'])}; passing: "
                  f"{', '.join(o['passing'])}")
        else:
            print(f"VERDICT {m} KILLED: {killed_text()}" +
                  ("" if b is None else f" Best: {b['arm']}, K-D delta {b['kd']:.2f}."))
            if o["narrow"]:
                print(f"{m}: fails narrowly; a recovery method is the untested next step")
            if o["ar_narrow"]:
                print(f"{m}: AdaRound's best eligible arm ({o['ar_best']['arm']}, K-D delta {o['ar_best']['kd']:.2f}) "
                      f"is in ({KILL_LINE}, {NARROW_LINE}]: a 10k-iteration AdaRound run is a new pre-registration, "
                      f"never a re-run")
    pair = res["pair"]
    print(f"EXTEND {'YES' if pair['extend'] else 'NO'}: " +
          (f"{', '.join(METHOD_TEXT[m] for m in pair['methods'])} passes on both models; YOLOv8n-pose and YOLO26n "
           f"open (the gate's reading of the pair rule)" if pair["extend"] else
           "no one method passes on both models; the extension does not open"))
    return 0


def cmd_zerodet(args) -> int:
    """v2.5 B.4, for the runner's eval_row when a row's eval exits with B.1's lines: the completion record. It prints
    the record's lines only when B.1 and B.3 hold; a refusal stops the stage exactly as in v2.4."""
    print_hashes()
    problems = frozen_state()
    lines, zp = zerodet_problems(Path(args.log), args.dets, args.audit)
    if problems + zp:
        return refuse(problems + zp, "ZERODET REFUSED: B.1 or B.3 does not hold; the stage stops (v2.5 B.3)")
    for line in lines:
        print(line)
    return 0


def cmd_dets_sha(path: str) -> int:
    p = Path(path)
    print(f"DETS_SHA256 {sha_file(p)} {shown(p)}")
    print(f"dets bytes {p.stat().st_size}")
    return 0


# ---------------------------------------------------------------------------------------------- the selftest

def selftest() -> int:
    """Synthetic data only: no model, COCO image, capture, arm file or detections JSON is read. The git checks run in
    a throwaway repository under the system temp directory, on copies of the tool files. Read-only, it also reads the
    repo's own sources (the frozen values, TOOLS_JSON's import graph), the committed carried and superseded v2.4 logs
    (v2.5 E), the archived yolov8s bc_e1 screen1 row and the committed audit logs (v2.5 B.1, B.3; the row's dets are
    given as the held [] in a temp directory), and the --advanced and --c8 children it starts read the committed W4V
    logs in results/int4/."""
    import numpy as np
    fails = []

    def expect(what, ok):
        print(f"[selftest] {'PASS' if ok else 'FAIL'} {what}", flush=True)
        if not ok:
            fails.append(what)

    h1 = hashes()
    expect(f"the embedded text is the accepted v2.4 + v2.5 (LF sha256 {PREREG_SHA[:8]}...): v2.4's part is "
           f"{PREREG_V24_SHA[:8]}... (V24_FROZEN's), v2.5's is {PREREG_V25_SHA[:8]}..., joined by one newline",
           h1["PREREG_TEXT_SHA256"] == PREREG_SHA and PREREG == PREREG_V24 + "\n" + PREREG_V25
           and sha_bytes(PREREG_V24.encode("utf-8")) == PREREG_V24_SHA == V24_FROZEN["PREREG_TEXT_SHA256"]
           and sha_bytes(PREREG_V25.encode("utf-8")) == PREREG_V25_SHA)
    expect("the triple is stable across calls", h1 == hashes())
    global KILL_LINE
    saved, KILL_LINE = KILL_LINE, 1.5
    try:
        expect("a changed constant changes VERDICT_CODE_SHA256", verdict_code_sha() != h1["VERDICT_CODE_SHA256"])
    finally:
        KILL_LINE = saved
    expect("the matrix has 17 tier-1 and 12 tier-2 arms", (len(arms(1)), len(arms(2))) == (17, 12))

    # v2.5 E on this checkout: rules 3 and 4, the carried and superseded tables against the committed logs, and the
    # widened FAIL regex (read-only: the committed W4V logs in results/int4/)
    rest = {k: v for k, v in protocol().items() if k != "v2_5"}
    expect("rule 4: the protocol JSON minus v2_5, its one added key, is v2.4's (PROTOCOL_JSON_SHA256 46297af6...)",
           sha_bytes(json.dumps(rest, sort_keys=True).encode("utf-8")) == V24_FROZEN["PROTOCOL_JSON_SHA256"]
           and set(protocol()) - set(rest) == {"v2_5"} and rest == protocol_v24())
    expect("rule 3: every file of v2.4's TOOLS_JSON is at its v2.4 LF sha, and v2.5 adds exactly the C5b tool",
           carry_global() == [] and set(tools_json()) - set(V24_FROZEN["TOOLS_JSON"]) == {C5B_TOOL}
           and set(V24_FROZEN["TOOLS_JSON"]) <= set(tools_json()))

    def success_of(name):   # the success lines ready() wants of a log with this name, or none (a witness)
        for s in STAGES:
            for g, succ in stage_logs(s, {m: [] for m in MODELS}):
                if fnmatch.fnmatch(name, g):
                    return succ
        return []
    carried = {n: carry_problems(RESULTS / n, success_of(n)) if (RESULTS / n).is_file() else ["missing"]
               for n in CARRIED_V24}
    expect(f"the {len(CARRIED_V24)} carried v2.4 logs pass rules 1-5: committed and clean at their listed LF shas, "
           f"every hash block V24_FROZEN, their success lines, no FAIL line under the widened regex",
           all(v == [] for v in carried.values()))
    expect(f"the {len(SUPERSEDED_V24)} superseded v2.4 logs are committed and clean at their listed LF shas, and no "
           f"glob counts them", all((RESULTS / n).is_file() and live(n) == ([], []) for n in SUPERSEDED_V24))
    fail_lines = ("W4V_CONTROL C5b yolov8n bc_e2 FAIL", "W4V_CONTROL C6c yolov8n FAIL (exit 1)",
                  "W4V_CONTROL C2 yolov8s FAIL")
    pass_lines = ("W4V_CONTROL C5b yolov8n bc_e2 PASS", "W4V_CONTROL C5 yolov8n_bc_e1 PASS", "W4V_REPORT x y FAIL")
    expect("CONTROL_FAIL reads one scope token or C5b's two: every FAIL form trips it, no PASS or report line does",
           all(re.search(CONTROL_FAIL, s, re.M) for s in fail_lines)
           and not any(re.search(CONTROL_FAIL, s, re.M) for s in pass_lines))

    expect("delta works on the printed values", delta("27.10", "25.10") == 2.0 and delta("27.10", "25.09") == 2.01)
    expect("the cut is <= 2.00", advances(2.0) and not advances(2.01))
    expect("the flag: cut on the 500, within the line on H", flagged(2.5, 1.9) and not flagged(2.5, 2.1)
           and not flagged(1.0, 0.5))
    expect("K-D needs both deltas", kd(1.0, 1.0)["pass"] and not kd(0.9, 1.1)["pass"] and kd(0.9, 1.1)["kd"] == 1.1)
    expect("eligibility: F never; E1 needs form b; any form-b file needs its own C8 PASS",
           not eligible("gptq_f", False, "PASS") and not eligible("ar_e1", True, "PASS")
           and not eligible("ar_e1", False, "FAIL") and not eligible("ar_e1", False, None)
           and eligible("ar_e1", False, "PASS") and eligible("ar_e2", True, None) and eligible("vc_u", True, None)
           and not eligible("gptq_u", False, "FAIL") and not eligible("bc_e2", False, None)
           and eligible("gptq_u", False, "PASS"))

    def cand(arm, d_all, d_h, elig=True):
        return {"arm": arm, "d_all": d_all, "d_h": d_h, "eligible": elig}
    o = model_outcome([cand("rtn_e1", 25.0, 25.1), cand("gptq_u", 0.8, 0.9), cand("ar_u", 0.8, 0.7),
                       cand("gptq_f", 0.1, 0.1, False)])
    expect("PASS picks the smallest K-D delta; F never decides", o["outcome"] == "PASS" and o["best"]["arm"] == "ar_u"
           and "gptq_f" not in o["passing"])
    o = model_outcome([cand("gptq_u", 0.9, 0.9), cand("ar_u", 0.9, 0.9)])
    expect("ties go to section 3.7's row order", o["best"]["arm"] == "ar_u")
    o = model_outcome([cand("gptq_u", 0.5, 1.5), cand("gptq_f", 0.1, 0.1, False)])
    expect("NARROW: no eligible pass, best K-D in (1, 2]", o["outcome"] == "KILLED" and o["narrow"])
    o = model_outcome([cand("gptq_u", 0.5, 2.5)])
    expect("KILLED without NARROW above 2", o["outcome"] == "KILLED" and not o["narrow"])
    o1 = model_outcome([cand("gptq_u", 1.2, 1.2), cand("ar_u", 1.9, 1.5)])
    o2 = model_outcome([cand("gptq_u", 1.2, 1.2), cand("mxar_e2", 2.4, 1.0)])
    o3 = model_outcome([cand("gptq_u", 1.2, 1.2)])
    o4 = model_outcome([cand("gptq_u", 0.9, 0.8), cand("ar_u", 1.5, 1.5)])
    expect("the 10k AdaRound clause: only when NARROW and AdaRound's own best eligible arm is in (1, 2]",
           o1["ar_narrow"] and o1["ar_best"]["arm"] == "ar_u" and not o2["ar_narrow"] and not o3["ar_narrow"]
           and o2["narrow"] and not o4["ar_narrow"])
    pr = pair_rule({"yolov8n": {"passing": ["ar_u"]}, "yolov8s": {"passing": ["ar_e2", "gptq_u"]}})
    expect("EXTEND when one method passes on both models", pr["extend"] and pr["methods"] == ["ar"])
    pr = pair_rule({"yolov8n": {"passing": ["ar_u"]}, "yolov8s": {"passing": ["mxar_u"]}})
    expect("no EXTEND across two methods", not pr["extend"])
    expect("phase-2 cost names the int32 bias and MX", "int32" in phase2_cost("vc_e1")
           and "per-layer packet" in phase2_cost("mxgptq_u") and phase2_cost("rtn_e1") == FORM_COST["e1"])

    # C8's coverage (E)
    orto = {m: {a: a.endswith("_u") for a in arms()} for m in MODELS}
    adv = {"yolov8n": ["bc_e1", "gptq_u"], "yolov8s": ["vc_u"]}
    c_hit = c8_coverage(adv, orto, {("yolov8n", "bc_e1")})
    c_miss = c8_coverage(adv, orto, set())
    c_none = c8_coverage({"yolov8n": ["gptq_u"], "yolov8s": ["vc_u"]}, orto, set())
    expect("C8 coverage: PASS when an advanced int32-bias form-b file was checked, whatever its result; FAIL when "
           "one advanced and none was checked; PASS with reasons when none advanced",
           c_hit["pass"] and not c_miss["pass"] and c_none["pass"] and c_none["form_b"] == []
           and "yolov8s vc_u: a form-a conv" in c_none["reasons"] and "yolov8n bc_e1: the cut" in c_none["reasons"])
    gs = gate_status_text("W4V_GATE " + "a" * 64 + " PASS\nW4V_GATE " + "a" * 64 + " FAIL\nW4V_GATE " + "b" * 64
                          + " PASS\n")
    expect("a W4V_GATE FAIL for a file is its result", gs == {"a" * 64: "FAIL", "b" * 64: "PASS"})

    # the screen prints the cut only after SCREEN COMPLETE (D)
    res = {"advanced": {"yolov8n": ["gptq_u"], "yolov8s": []}, "cut": {"yolov8n": ["rtn_e1"], "yolov8s": ["rtn_u"]},
           "flags": {"yolov8n": [], "yolov8s": ["rtn_u"]}}
    exp = {m: arms(1) for m in MODELS}
    bad, good = screen_lines(res, ["x"], 1, exp), screen_lines(res, [], 1, exp)
    expect("SCREEN INCOMPLETE prints no CUT_JSON or FLAG_JSON; SCREEN COMPLETE comes first, then the cut",
           not any(line.startswith(("CUT_JSON", "FLAG_JSON")) for line in bad) and bad[-1] == "SCREEN INCOMPLETE"
           and good[0] == "SCREEN COMPLETE" and good[1].startswith("CUT_JSON ") and good[2].startswith("FLAG_JSON "))

    # v2.5 B.2, B.4 and E: a zero row's figures by rule, and C9's setup and its re-derivation in --verdict
    expect("a recorded zero-detection row is 0.00 by rule and never rescored; B8 - 0.00 > 2.0 on the 500 and on H, "
           "so it is cut and not flagged (B.4)",
           row_check({"zero": True}, None, [], [], ROOT) == {"ok": True, "zero": True, "h": "0.00"}
           and not advances(delta("30.00", "0.00")) and not flagged(delta("30.00", "0.00"), delta("29.00", "0.00")))
    now = frozen_values()
    r9 = {"images": DECISIVE_N, "skipped": 0, "ort": ORT_VERSION, "ort_opt": EVAL["ort_opt"],
          "threads": EVAL["threads"]}
    carried9 = next(n for n in CARRIED_V24 if n.startswith("w4v_repro_yolov8n_rtn_e1_"))
    fresh9 = "w4v_repro_yolov8n_rtn_e1_n5000_desktop2_20260928.log"
    expect("C9's setup (v2.5 E): a carried repro row wants v2.4's triple, any other row this tool's; 4,999 images or "
           "one skipped image fails it",
           c9_setup({**r9, "log": carried9, "triple": V24_FROZEN}, now)
           and not c9_setup({**r9, "log": carried9, "triple": now}, now)
           and c9_setup({**r9, "log": fresh9, "triple": now}, now)
           and not c9_setup({**r9, "log": fresh9, "triple": V24_FROZEN}, now)
           and not c9_setup({**r9, "log": carried9, "triple": V24_FROZEN, "images": DECISIVE_N - 1}, now)
           and not c9_setup({**r9, "log": carried9, "triple": V24_FROZEN, "skipped": 1}, now))

    class NoData:   # stands in for Data: another annotation file and no H; with no rows nothing is scored
        ann_sha, coco = "0" * 64, None

        def sets(self, n):
            return [], []
    head9, c9 = c9_rederive([], NoData(), ROOT, C9_TAG)
    lines9 = [head9] + [x for ls, _ in c9.values() for x in ls]
    expect("--verdict re-derives C9 under its own tag, never a W4V_CONTROL line; another annotation file or a missing "
           "repro row is a FAIL for its model",
           all(x.startswith(C9_TAG + " ") for x in lines9) and "NOT the pinned file" in head9
           and not any(x.startswith("W4V_CONTROL") for x in lines9 + [f"{C9_TAG} {m} FAIL" for m in MODELS])
           and sorted(c9) == sorted(MODELS) and not any(good for _, good in c9.values())
           and sum(x.endswith(": 0 rows, want 1") for x in lines9) == sum(len(GATE_D_MAP[m]) for m in MODELS))

    # v2.5 B.1 and B.3 on the committed evidence, at the v2.4 prereg's pins: the archived yolov8s bc_e1 row (its dets
    # given as the held [] in a temp root) is B.1's zero row, and every committed audit log passes B.3 at its pin
    pins24, _ = v24_pins()

    def pin24(m, a):
        return (pins24.get(f"scratch/w4v/arms/{m}_w4v_{a}.onnx")
                or pins24.get(f"scratch/w4v/arms/{m}_w4v_{a}_ortonly.onnx"))
    with tempfile.TemporaryDirectory(prefix="w4v-selftest-zero-") as t:
        (Path(t) / "scratch" / "w4v" / "dets").mkdir(parents=True)
        (Path(t) / "scratch" / "w4v" / "dets" / "w4v_screen_yolov8s_bc_e1_n500.json").write_text("[]", encoding="utf-8")
        z_row, z_problems = zero_row(RESULTS / "w4v_v24_screen1_stop" /
                                     "w4v_screen_yolov8s_bc_e1_n500_desktop2_20260926.log", Path(t))
    z_real = [(audit_problems(RESULTS / "w4v_audit" / n, m, a, pin24(m, a))[1],
               audit_problems(RESULTS / "w4v_audit" / n, m, a, "0" * 64)[1])
              for n, m, a in (("recompute_yolov8n_rerun_desktop2_20260926.log", "yolov8n", "bc_e1"),
                              ("recompute_yolov8s_rerun2_desktop2_20260926.log", "yolov8s", "bc_e1"),
                              ("c5b_validate_yolov8n_desktop2_20260926.log", "yolov8n", "bc_e1"),
                              ("c5b_validate_yolov8s_desktop2_20260926.log", "yolov8s", "bc_e1"),
                              ("c5b_precheck_vc_yolov8n_desktop2_20260926.log", "yolov8n", "vc_e1"))]
    expect("B.1 and B.3 on the committed evidence: the archived yolov8s bc_e1 row is B.1's zero row at its v2.4 pin; "
           "both GENUINE recomputes, both C5b validations and the VC pre-check pass B.3 at their v2.4 pins, and each "
           "is refused at another pin",
           z_problems == [] and z_row["sha256"] == pin24("yolov8s", "bc_e1") and z_row["images"] == SCREEN_N
           and all(ok == [] and any("PIN_CHECK" in p for p in bad) for ok, bad in z_real))

    # git: committed and clean, the one prereg log, TOOLS_JSON, pins and the cut logs (A, B, C, D, G)
    tmp = Path(tempfile.mkdtemp(prefix="w4v-selftest-git-"))
    try:
        def g(*a):   # every git call here must succeed: a commit that did nothing would leave the cases untested
            r = subprocess.run(["git", "-c", "user.name=w4v-selftest", "-c", "user.email=selftest@invalid", *a],
                               cwd=tmp, capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(f"selftest repo: git {' '.join(a)} exited {r.returncode}: {r.stderr.strip()[:200]}")
            return r
        g("init", "-q")
        for rel in CODE_FILES + TOOL_FILES:
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp / rel).write_bytes((ROOT / rel).read_bytes())
        res_dir = tmp / "results" / "int4"
        res_dir.mkdir(parents=True)
        f = res_dir / "note.log"
        f.write_text("x\n", encoding="utf-8")
        s_untracked = committed_clean(f, tmp)
        g("add", "-A")
        s_staged = committed_clean(f, tmp)
        g("commit", "-q", "-m", "one")
        s_committed = committed_clean(f, tmp)
        f.write_text("y\n", encoding="utf-8")
        s_edited = committed_clean(f, tmp)
        g("checkout", "--", "results/int4/note.log")
        expect("committed and clean: untracked no, only `git add`ed no, committed yes, edited after commit no",
               (s_untracked, s_staged, s_committed, s_edited) == (False, False, True, False)
               and committed_clean(f, tmp))
        v0 = frozen_values(tmp)
        runner = tmp / "scripts" / "w4v-eval.sh"
        runner.write_bytes(runner.read_bytes() + b"# one more byte\n")
        pow2 = tmp / "quant" / "pow2.py"
        pow2.write_bytes(pow2.read_bytes() + b"\n")
        v1 = frozen_values(tmp)
        expect("VERDICT_CODE_SHA256 hashes the whole runner; TOOLS_JSON changes with quant/pow2.py",
               v1["VERDICT_CODE_SHA256"] != v0["VERDICT_CODE_SHA256"] and v1["TOOLS_JSON"] != v0["TOOLS_JSON"]
               and v1["PROTOCOL_JSON_SHA256"] == v0["PROTOCOL_JSON_SHA256"])
        g("checkout", "--", "scripts/w4v-eval.sh", "quant/pow2.py")
        expect("no prereg log: not frozen", frozen_state(tmp) != [])
        pins_a = {f"{m}_w4v_{a}{'_ortonly' if orto[m][a] else ''}.onnx": sha_bytes(f"{m}{a}".encode())
                  for m in MODELS for a in arms(1)}
        prereg = res_dir / "w4v_prereg_desktop2_20260928.log"   # not a SUPERSEDED_V24 name
        prereg.write_text("\n".join(hash_lines(tmp) + [f"PIN {s}  scratch/w4v/arms/{n}" for n, s in pins_a.items()])
                          + "\n", encoding="utf-8")
        s_uncommitted = frozen_state(tmp)
        g("add", "-A")
        g("commit", "-q", "-m", "prereg")
        s_frozen = frozen_state(tmp)
        pow2.write_bytes(pow2.read_bytes() + b"\n")
        s_tool = frozen_state(tmp)
        g("checkout", "--", "quant/pow2.py")
        second = res_dir / "w4v_prereg_desktop2_20260927.log"
        second.write_text(prereg.read_text(encoding="utf-8"), encoding="utf-8")
        s_two = frozen_state(tmp)
        second.unlink()
        expect("frozen: the one committed, clean prereg log with the same triple and TOOLS_JSON; refused when it is "
               "uncommitted, when a tool changed, or when there are two",
               s_frozen == [] and s_uncommitted != [] and any("TOOLS_JSON" in p for p in s_tool) and s_two != [])
        table = pins(tmp)
        pins_log = res_dir / "w4v_pins_desktop2_20260927.log"
        pins_log.write_text("PIN " + "c" * 64 + "  scratch/w4v/arms/yolov8n_w4v_arbc_e1.onnx\n", encoding="utf-8")
        try:
            pins(tmp)
            refused = False
        except SystemExit:
            refused = True
        pins_log.unlink()
        expect("pins come from committed, clean logs only (an uncommitted pins log refuses them all)",
               table == pins_a and refused and pins(tmp) == pins_a)

        # v2.5 E: a superseded log (a synthetic table) is skipped at each site at its listed sha, refused if edited
        old_pre = res_dir / "w4v_prereg_desktop2_20260930.log"
        old_pre.write_text(hash_lines(tmp)[0] + "\nPIN " + "d" * 64 + "  scratch/w4v/arms/old.onnx\n", encoding="utf-8")
        old_st = res_dir / "w4v_selftest_desktop2_20260930.log"
        old_st.write_text("[selftest] PASS\n", encoding="utf-8")
        new_st = res_dir / "w4v_selftest_desktop2_20260926.log"
        new_st.write_text(("\n".join(hash_lines(tmp)) + "\n[selftest] PASS\n") * 2, encoding="utf-8")
        g("add", "-A")
        g("commit", "-q", "-m", "superseded")
        saved_sup = SUPERSEDED_V24
        globals()["SUPERSEDED_V24"] = {old_pre.name: sha_lf(old_pre), old_st.name: sha_lf(old_st)}
        try:   # the table is in protocol(), so the hash blocks differ under the patch: only the counts are read here
            def st_problems():
                return [p for p in ready("controls", tmp) if "want exactly one" in p or "superseded" in p]

            def pre_problems():
                return [p for p in frozen_state(tmp) if "prereg_*.log files" in p or "superseded" in p]
            u_live = live("w4v_prereg_*.log", tmp)
            u_frozen, u_pins, u_own, u_ready = pre_problems(), pins(tmp), own_logs("prereg", tmp), st_problems()
            old_pre.write_text(old_pre.read_text(encoding="utf-8") + "edited\n", encoding="utf-8")
            old_st.write_text("edited\n", encoding="utf-8")
            e_live, e_frozen, e_own, e_ready = live("w4v_prereg_*.log", tmp), pre_problems(), own_logs(
                "prereg", tmp), st_problems()
            try:
                pins(tmp)
                e_pins = False
            except SystemExit:
                e_pins = True
        finally:
            globals()["SUPERSEDED_V24"] = saved_sup
            g("checkout", "--", old_pre.relative_to(tmp).as_posix(), old_st.relative_to(tmp).as_posix())
            g("rm", "-q", old_pre.relative_to(tmp).as_posix(), old_st.relative_to(tmp).as_posix(),
              new_st.relative_to(tmp).as_posix())
            g("commit", "-q", "-m", "superseded, removed")
        expect("superseded (v2.5 E): a listed log at its sha is skipped by live(), frozen_state() (one prereg log), "
               "pins() (its PIN lines unread), ready()'s stage_logs (one selftest log) and own_logs(); edited, it is a "
               "PROBLEM or a refusal at each",
               u_live == ([prereg], []) and u_frozen == [] and u_pins == pins_a and u_own == [prereg]
               and u_ready == [] and e_live[0] == [prereg] and len(e_live[1]) == 1
               and any("superseded" in p for p in e_frozen) and e_pins and old_pre in e_own
               and any("superseded" in p for p in e_ready))

        def cut_log(text):
            p = res_dir / "w4v_cut_tier1_desktop2_20260926.log"
            p.write_text(text, encoding="utf-8")
            g("add", "-A")
            g("commit", "-q", "-m", "cut")
            return read_cuts(pins_a, tmp, tiers=(1,))
        adv_ok = {"yolov8n": ["ar_u", "gptq_u"], "yolov8s": ["gptq_u"]}
        cut_ok = {m: [a for a in arms(1) if a not in adv_ok[m]] for m in MODELS}
        body = "CUT_JSON " + json.dumps({"tier": 1, "advanced": adv_ok, "cut": cut_ok}, sort_keys=True) + "\n"
        r_ok = cut_log("table\nSCREEN COMPLETE\n" + body)
        r_order = cut_log(body + "SCREEN COMPLETE\n")
        r_none = cut_log("table\nSCREEN INCOMPLETE\n")
        short = {m: v[1:] for m, v in cut_ok.items()}
        r_short = cut_log("SCREEN COMPLETE\nCUT_JSON " + json.dumps({"tier": 1, "advanced": adv_ok, "cut": short},
                                                                    sort_keys=True) + "\n")
        r_json = cut_log("SCREEN COMPLETE\nCUT_JSON {not: json}\n")
        (res_dir / "w4v_cut_tier1_desktop2_20260926.log").write_text("SCREEN COMPLETE\n" + body, encoding="utf-8")
        g("add", "-A")
        r_staged = read_cuts(pins_a, tmp, tiers=(1,))
        expect("the cut: one committed, clean log per tier, CUT_JSON after SCREEN COMPLETE, and every pinned arm "
               "either advanced or cut; refused otherwise, and a CUT_JSON that is not JSON is a PROBLEM",
               r_ok == ({"yolov8n": ["ar_u", "gptq_u"], "yolov8s": ["gptq_u"]}, []) and r_order[1] != []
               and r_none[1] != [] and r_short[1] != [] and r_json[1] != [] and r_staged[1] != [])
        g("commit", "-q", "-m", "cut")

        # the stage order: every earlier stage's committed log, its success line, its hash blocks (R2, R4)
        want = frozen_values(tmp)
        blk = "\n".join(hash_lines(tmp)) + "\n"
        blk_other = blk.replace(want["VERDICT_CODE_SHA256"], "0" * 64)

        def put(name, text, commit=True):
            p = res_dir / name
            p.write_text(text, encoding="utf-8")
            if commit:
                g("add", "-A")
                g("commit", "-q", "-m", name)
            return p
        put("w4v_selftest_desktop2_20260926.log", blk + "[selftest] PASS\n" + blk + "[selftest] PASS\n")
        for m in MODELS:
            put(f"w4v_controls_{m}_desktop2_20260926.log", blk + f"W4V_CONTROL C0 {m} PASS\n[controls] {m} PASS\n")
            put(f"w4v_parity_c6a_{m}_desktop2_20260926.log", blk + f"W4V_CONTROL C6a {m} PASS\n")
        put("w4v_parity_c6b_yolov8n_desktop2_20260926.log", blk + "W4V_CONTROL C6b yolov8n PASS\n")
        c6c = "w4v_parity_c6c_yolov8n_desktop2_20260926.log"
        good_c6c = blk + "W4V_CONTROL C6c yolov8n PASS\n"
        put(c6c, good_c6c)
        s_ready = ready("repro", tmp)
        put(c6c, blk + "W4V_CONTROL C6c yolov8n FAIL (exit 1)\n")
        s_fail = ready("repro", tmp)
        put(c6c, blk)
        s_trunc = ready("repro", tmp)
        put(c6c, good_c6c + blk_other)
        s_blocks = ready("repro", tmp)
        put(c6c, "W4V_CONTROL C6c yolov8n PASS\n")
        s_noblock = ready("repro", tmp)
        put(c6c, good_c6c, commit=False)
        s_uncommitted = ready("repro", tmp)
        put(c6c, good_c6c)
        s_again = ready("repro", tmp)
        expect("a stage waits for every earlier stage's committed log with its success line: ready; refused on a "
               "committed C6c FAIL, a truncated log, differing hash blocks, no hash block, an uncommitted log",
               s_ready == [] and s_again == [] and any("FAIL" in p for p in s_fail)
               and any("success line" in p for p in s_trunc) and any("hash blocks unlike" in p for p in s_blocks)
               and any("no hash block" in p for p in s_noblock)
               and any("not committed" in p for p in s_uncommitted))
        s_ctrl = ready("parity", tmp)
        put("w4v_controls_yolov8s_desktop2_20260926.log", blk + "W4V_CONTROL C2 yolov8s FAIL\n"
            "[controls] yolov8s FAIL (STOP at C2; no later control ran)\n")
        s_ctrl_fail = ready("parity", tmp)
        expect("parity waits for both controls logs' PASS lines; a committed controls FAIL log refuses it",
               s_ctrl == [] and any("w4v_controls_yolov8s" in p for p in s_ctrl_fail))
        emit = put("w4v_emit1_yolov8n_desktop2_20260928.log", (blk + "W4V_CONTROL C5 yolov8n_bc_e1 PASS\n") * 14
                   + "W4V_DONE emit1 yolov8n\n", commit=False)
        emit1_success = dict(stage_logs("emit1"))["w4v_emit1_yolov8n_*.log"]
        expect("an emit log's 14 equal hash blocks pass; --pins reads an uncommitted emit log (need_commit off), "
               "ready() wants it committed",
               log_problems(emit, emit1_success, want, tmp, need_commit=False) == []
               and log_problems(emit, emit1_success, want, tmp) != [])
        gate_success = dict(stage_logs("gate"))["w4v_gate_*.log"]
        gate_lines = "W4V_GATE " + "a" * 64 + " FAIL\nW4V_CONTROL C8 int32 PASS\n"
        gate_pass = blk + "SRC_TREE " + "e" * 40 + "\n" + gate_lines
        g_ok = put("w4v_gate_desktop2_20260926.log", gate_pass + "W4V_DONE gate\n")
        s_gate_ok = log_problems(g_ok, gate_success, want, tmp)
        g_fail = put("w4v_gate_desktop2_20260926.log", blk + "SRC_TREE " + "e" * 40 + "\nW4V_CONTROL C8 int32 FAIL\n")
        s_gate_fail = log_problems(g_fail, gate_success, want, tmp)
        g_cut = put("w4v_gate_desktop2_20260926.log", gate_pass)
        s_gate_cut = log_problems(g_cut, gate_success, want, tmp)
        g_nosrc = put("w4v_gate_desktop2_20260926.log", blk + gate_lines + "W4V_DONE gate\n")
        s_gate_nosrc = log_problems(g_nosrc, gate_success, want, tmp)
        expect("decisive waits for the gate log's SRC_TREE line, C8 int32 PASS and completion line (a file's W4V_GATE "
               "FAIL is not a stage failure): refused on C8 FAIL, on a log cut short and on one without SRC_TREE",
               s_gate_ok == [] and s_gate_fail != [] and s_gate_cut != [] and s_gate_nosrc != []
               and "gate" in STAGES[:STAGES.index("decisive")])

        # S2: the runner's --ready (starting) also refuses while any log of the stage itself exists
        def restart(ps):
            return [p for p in ps if "a restart needs the gate's ruling" in p]
        s_own = ready("parity", tmp, starting=True)   # the four committed parity logs above
        s_in = ready("parity", tmp)
        s_fresh = ready("repro", tmp, starting=True)
        old = put("w4v_repro_c9_desktop2_20260920.log", blk, commit=False)   # another date, never committed
        s_old = ready("repro", tmp, starting=True)
        old.unlink()
        wit = put("load_w4v_repro_desktop2_20260920.log", blk, commit=False)
        s_wit = ready("repro", tmp, starting=True)
        wit.unlink()
        expect("the runner's --ready also refuses a stage with a log of its own, of any date, committed or not (its "
               "load witness included); the in-stage checks leave that to it",
               len(restart(s_own)) == 4 and not restart(s_in) and not restart(s_fresh)
               and any("w4v_repro_c9_desktop2_20260920" in p for p in restart(s_old))
               and any("load_w4v_repro_desktop2_20260920" in p for p in restart(s_wit)))
        s_v0 = ready("verdict", tmp, starting=True)
        verdict_log = put("w4v_verdict_desktop2_20260920.log", blk, commit=False)
        s_v1 = ready("verdict", tmp, starting=True)
        verdict_log.unlink()
        expect("the verdict runs once: its --ready refuses while a verdict log exists, of any date",
               not restart(s_v0) and any("w4v_verdict_desktop2_20260920" in p for p in restart(s_v1)))

        # v2.5 E: a carried log (synthetic tables): accepted in place of its stage's v2.5 run, then each rule broken
        v24_syn = {"PREREG_TEXT_SHA256": "1" * 64, "PROTOCOL_JSON_SHA256": V24_FROZEN["PROTOCOL_JSON_SHA256"],
                   "VERDICT_CODE_SHA256": "2" * 64,
                   "TOOLS_JSON": {k: v for k, v in tools_json(tmp).items() if k != C5B_TOOL}}
        blk24 = "\n".join([f"{k} {v24_syn[k]}" for k in ("PREREG_TEXT_SHA256", "PROTOCOL_JSON_SHA256",
                                                           "VERDICT_CODE_SHA256")]
                          + ["TOOLS_JSON " + json.dumps(v24_syn["TOOLS_JSON"], sort_keys=True)]) + "\n"
        c_name = "w4v_controls_yolov8n_desktop2_20260920.log"
        c_succ = dict(stage_logs("controls"))["w4v_controls_yolov8n_*.log"]
        c_good = blk24 + "W4V_CONTROL C0 yolov8n PASS\n[controls] yolov8n PASS\n"
        saved_carry, saved_cut = (CARRIED_V24, V24_FROZEN), CUT_LINE
        try:
            globals()["V24_FROZEN"] = v24_syn

            def carried(text, commit=True):   # each call writes new content, so its commit is never empty
                p = put(c_name, text, commit)
                globals()["CARRIED_V24"] = {c_name: sha_lf(p)}
                return log_problems(p, c_succ, want, tmp)
            k_ok = carried(c_good)
            p_c = res_dir / c_name
            globals()["CARRIED_V24"] = {c_name: "0" * 64}
            k_1 = log_problems(p_c, c_succ, want, tmp)
            globals()["CARRIED_V24"] = {c_name: sha_lf(p_c)}
            pow2.write_bytes(pow2.read_bytes() + b"\n")
            k_3 = log_problems(p_c, c_succ, want, tmp)
            g("checkout", "--", "quant/pow2.py")
            globals()["CUT_LINE"] = 2.5
            k_4 = log_problems(p_c, c_succ, want, tmp)
            globals()["CUT_LINE"] = saved_cut
            k_2 = carried(blk + "W4V_CONTROL C0 yolov8n PASS\n[controls] yolov8n PASS\n")
            k_5b = carried(blk24 + "W4V_CONTROL C0 yolov8n PASS\n")
            k_5c = carried(c_good + "W4V_CONTROL C5b yolov8n bc_e2 FAIL\n")
            k_5a = carried(c_good + "x\n", commit=False)
            row = {"log": c_name, "ort": ORT_VERSION, "ort_opt": EVAL["ort_opt"], "threads": EVAL["threads"],
                   "triple": v24_syn, "sha256": "e" * 64, "model": "yolov8n", "arm": "b8"}
            k_rows = (common_problems([row], {"yolov8n_cut_xint8.onnx": "e" * 64}),
                      common_problems([{**row, "triple": frozen_values()}], {"yolov8n_cut_xint8.onnx": "e" * 64}))
            globals()["CARRIED_V24"] = {}
            k_rows += (common_problems([row], {"yolov8n_cut_xint8.onnx": "e" * 64}),)
        finally:
            (globals()["CARRIED_V24"], globals()["V24_FROZEN"]), globals()["CUT_LINE"] = saved_carry, saved_cut
            g("checkout", "--", "quant/pow2.py")
            g("rm", "-q", "-f", f"results/int4/{c_name}")
            g("commit", "-q", "-m", "carried, removed")
        expect("carried (v2.5 E): a listed log with V24_FROZEN blocks stands in for its v2.5 run, and its row's v2.4 "
               "triple is accepted; refused on rule 1 (another sha), 2 (a v2.5 block), 3 (a tool changed), 4 (the "
               "protocol minus v2_5 changed), 5 (uncommitted; no success line; a C5b FAIL line); an unlisted row "
               "with the v2.4 triple is refused",
               k_ok == [] and any(p.startswith("rule 1") for p in k_1) and any(p.startswith("rule 2") for p in k_2)
               and any(p.startswith("rule 3") for p in k_3) and any(p.startswith("rule 4") for p in k_4)
               and any("not committed" in p for p in k_5a) and any("success line" in p for p in k_5b)
               and any(p.startswith("rule 5: holds a W4V_CONTROL FAIL") for p in k_5c)
               and k_rows[0] == [] and k_rows[1] != [] and k_rows[2] != [])

        # v2.5 B.3 and B.4: --zerodet's record, its refusals (B.1 and B.3 each broken once), and a recorded row in
        # ready(), load_rows() and own_logs(); the audits are the committed logs, re-pinned to this repo's pins
        dets_dir = tmp / "scratch" / "w4v" / "dets"
        dets_dir.mkdir(parents=True)
        (res_dir / "w4v_audit").mkdir()

        def pin_a(m, a):
            return pins_a.get(f"{m}_w4v_{a}.onnx") or pins_a.get(f"{m}_w4v_{a}_ortonly.onnx")

        def zlines(stage, m, a, n=SCREEN_N):   # B.1's row log, in 5_eval_map.py's format (the archived row's)
            row = f"w4v_{stage}_{m}_{a}_n{n}"
            return row, [*hash_lines(tmp), f"onnxruntime {ORT_VERSION}",
                         f"ort-opt : {EVAL['ort_opt']}   ort-threads : {EVAL['threads']}",
                         f"model sha256 {pin_a(m, a) or 'f' * 64}", f"0 detections over {n} images in 46s",
                         f"wrote scratch/w4v/dets/{row}.json", "no detections at all -- something is broken upstream"]

        def zput(row, lines, dets="[]", commit=True):
            (dets_dir / f"{row}.json").write_text(dets, encoding="utf-8")
            return put(f"{row}_desktop2_20260928.log", "\n".join(lines) + "\n", commit)

        def zd(row_p, audit=None):
            return zerodet_problems(row_p, f"scratch/w4v/dets/{row_of(row_p.name)}.json",
                                    str(audit or tmp / AUDIT_LOGS["yolov8s"]), tmp)

        def undo(*ps):   # back to HEAD: every case below edits a committed file without committing
            g("checkout", "--", *[p.relative_to(tmp).as_posix() for p in ps])
        real_s = (ROOT / AUDIT_LOGS["yolov8s"]).read_text(encoding="utf-8")
        audit_p = put("w4v_audit/" + Path(AUDIT_LOGS["yolov8s"]).name,
                      real_s.replace(pin24("yolov8s", "bc_e1"), pin_a("yolov8s", "bc_e1")))
        z_name, z_lines = zlines("screen", "yolov8s", "bc_e1")
        row_p = zput(z_name, z_lines)
        z_dets = dets_dir / f"{z_name}.json"
        rec_lines, z_ok = zd(row_p)
        z_before = [p for p in ready("emit2", tmp) if z_name in p]
        try:
            load_rows([str(row_p)], True, tmp)
            z_raised = False
        except ValueError:
            z_raised = True
        rec_text = blk + "\n".join(rec_lines) + "\n"
        rec_p = put(f"w4v_zerodet_yolov8s_bc_e1_n{SCREEN_N}_desktop2_20260928.log", rec_text)
        z_rec = zero_record(row_p, tmp, want)[1]
        z_loaded = load_rows([str(row_p)], True, tmp)
        z_after = [p for p in ready("emit2", tmp) if z_name in p]
        z_own = (rec_p in own_logs("screen1", tmp), rec_p in own_logs("screen2", tmp))
        expect("zerodet (v2.5 B.4): a B.1 row of an int32-bias arm with its committed GENUINE audit at its pin gets "
               "the record's six lines; recorded and committed, the row reads 0.00 by rule in load_rows() and counts "
               "in ready(), which refused it before; screen1's own_logs() holds the record, screen2's not",
               z_ok == [] and [x.split()[0] for x in rec_lines] == ["ZERO_DET_LOG", "ZERO_DET_DETS", "ZERO_DET_AUDIT",
                                                                    "ZERO_DET_HELD", "ZERO_DET_ARM", "ZERO_DET_ROW"]
               and rec_lines[5] == f"ZERO_DET_ROW {z_name} AP_N 0.00 AP_H 0.00 (rule B.2)"
               and rec_lines[3].endswith(")") and "-> GENUINE (" in rec_lines[3]
               and z_raised and any("zero-detection record" in p for p in z_before) and z_after == [] and z_rec == []
               and z_loaded[0]["zero"] and z_loaded[0]["map"] == "0.00" and z_own == (True, False))

        # the record's refusals: uncommitted, the dets or the row changed after it, two records, another hash block
        rec_p.write_text(rec_text + "x\n", encoding="utf-8")
        r_unc = (zero_record(row_p, tmp, want)[1], zero_record(row_p, tmp, want, need_commit=False)[1])
        rec_p.write_text(blk_other + "\n".join(rec_lines) + "\n", encoding="utf-8")
        r_blk = zero_record(row_p, tmp, want, need_commit=False)[1]
        undo(rec_p)
        z_dets.write_text("[ ]", encoding="utf-8")
        r_dets = zero_record(row_p, tmp, want)[1]
        z_dets.write_text("[]", encoding="utf-8")
        row_p.write_text("\n".join(z_lines) + "\n# x\n", encoding="utf-8")
        r_row = zero_record(row_p, tmp, want)[1]
        undo(row_p)
        second = put(f"w4v_zerodet_yolov8s_bc_e1_n{SCREEN_N}_desktop2_20260929.log", rec_text, commit=False)
        r_two = zero_record(row_p, tmp, want)[1]
        second.unlink()
        expect("a record counts only while it holds: refused when it is uncommitted (need_commit), when its hash "
               "block is not the frozen one, when the dets or the row log changed after it, and when two name the row",
               any("not committed" in p for p in r_unc[0]) and r_unc[1] == []
               and any("hash blocks are not the frozen values" in p for p in r_blk)
               and any("ZERO_DET_DETS" in p for p in r_dets) and any("ZERO_DET_LOG" in p for p in r_row)
               and any("2 zero-detection records" in p for p in r_two) and zero_record(row_p, tmp, want)[1] == [])

        # --zerodet's refusals. B.1, line by line, and the dets file
        def b1(lines=None, dets="[]"):
            row_p.write_text("\n".join(lines or z_lines) + "\n", encoding="utf-8")
            z_dets.write_text(dets, encoding="utf-8")
            out = zd(row_p)[1]
            undo(row_p)
            z_dets.write_text("[]", encoding="utf-8")
            return out
        wrote = f"wrote scratch/w4v/dets/{z_name}.json"
        z_b1 = [b1([x.replace(f"over {SCREEN_N} images", "over 499 images") for x in z_lines]),
                b1(z_lines[:-1]),
                b1(z_lines + [f"DETS_SHA256 {'a' * 64} scratch/w4v/dets/{z_name}.json"]),
                b1(z_lines + ["  skip unreadable 000000000001.jpg"]),
                b1([x.replace(wrote, "wrote scratch/w4v/dets/other.json") for x in z_lines]),
                b1(dets='[{"image_id": 1}]'), b1(dets="{}")]
        want_b1 = ["499 images, the row's n is 500", "0 'no detections at all' lines", "a DETS_SHA256 line",
                   "a skip unreadable line", "not the runner's", "is 1 detections", "missing or not a JSON list"]
        # B.3: the arm's kind, its pin, and its audit (the static bc_e1 log: its commit, HELD, PIN_CHECK, IMPORTS and
        # script; --audit naming another log)
        z_kind = []
        for stage, m, a, n in (("screen", "yolov8s", "b8", SCREEN_N), ("repro", "yolov8n", "rtn_e1", DECISIVE_N),
                               ("screen", "yolov8n", "rtn_e1", SCREEN_N)):
            kn, kl = zlines(stage, m, a, n)
            kp = zput(kn, kl, commit=False)
            z_kind.append(zd(kp)[1])
            kp.unlink()
            (dets_dir / f"{kn}.json").unlink()
        row_p.write_text("\n".join(x.replace(pin_a("yolov8s", "bc_e1"), "e" * 64) for x in z_lines) + "\n",
                         encoding="utf-8")
        z_sha = zd(row_p)[1]
        undo(row_p)

        def audit_as(text):
            audit_p.write_text(text, encoding="utf-8")
            out = zd(row_p)[1]
            undo(audit_p)
            return out
        a_text = audit_p.read_text(encoding="utf-8")
        z_audit = [audit_as(a_text + "\n"), audit_as(a_text.replace("-> GENUINE (", "-> DEFECT (")),
                   audit_as(a_text.replace(pin_a("yolov8s", "bc_e1"), pin24("yolov8s", "bc_e1"))),
                   audit_as(a_text.replace("IMPORTS w4_recover or quant loaded: none |",
                                           "IMPORTS w4_recover or quant loaded: quant.qdq |")),
                   audit_as(a_text.replace(AUDIT_SCRIPT_SHA, "0" * 64)),
                   zd(row_p, res_dir / "note.log")[1]]
        want_audit = ["is not committed and clean", "does not read GENUINE in full", "PIN_CHECK of the arm file",
                      "IMPORTS line does not read none", "not a recompute by dbc4b7bb", "B.4: --audit"]
        expect("--zerodet refuses unless B.1 holds line by line (another N, no exit line, a DETS_SHA256 line, a skip "
               "line, another wrote path, a detection, a dets that is no list) and B.3 holds (B8, C9's RTN rows and "
               "an arm with no int32 bias are STOPs; the row's model sha is the pin; its audit is committed, reads "
               "GENUINE in full, PIN_CHECKs this pin, imports nothing, is by dbc4b7bb, and is the one --audit names)",
               all(any(w in p for p in got) for got, w in zip(z_b1, want_b1)) and len(z_b1) == len(want_b1)
               and any("always a STOP" in p for p in z_kind[0]) and any("always a STOP" in p for p in z_kind[1])
               and any("(D2)" in p for p in z_kind[2]) and any("is not yolov8s bc_e1's pin" in p for p in z_sha)
               and all(any(w in p for p in got) for got, w in zip(z_audit, want_audit))
               and len(z_audit) == len(want_audit) and zd(row_p)[1] == [])

        # a bias1 arm (v2.5 B.3, the gate's Q5): exactly one committed, clean w4v_bias1_<model>_*.log, whose section
        # for the arm is by the frozen C5b tool with its C5B PASS line
        bn, bl = zlines("screen", "yolov8s", "bc_e2")
        bp = zput(bn, bl, commit=False)
        c5b_s = (ROOT / "results" / "int4" / "w4v_audit" / "c5b_validate_yolov8s_desktop2_20260926.log").read_text(
            encoding="utf-8").replace(pin24("yolov8s", "bc_e1"), pin_a("yolov8s", "bc_e2")).replace("bc_e1", "bc_e2")
        bias_name = "w4v_bias1_yolov8s_desktop2_20260928.log"
        z_b_none = zd(bp, res_dir / bias_name)[1]
        bias_p = put(bias_name, c5b_s)
        z_b_ok = zd(bp, bias_p)[1]
        bias_p.write_text(c5b_s.replace("C5B yolov8s bc_e2 PASS", "C5B yolov8s bc_e2 FAIL"), encoding="utf-8")
        z_b_fail = zd(bp, bias_p)[1]
        undo(bias_p)
        other_bias = put("w4v_bias1_yolov8s_desktop2_20260929.log", c5b_s, commit=False)
        z_b_two = zd(bp, bias_p)[1]
        other_bias.unlink()
        bp.unlink()
        (dets_dir / f"{bn}.json").unlink()
        expect("a bias1 arm's audit is its one committed, clean w4v_bias1_<model>_*.log: none or two refuse; the "
               "frozen C5b tool's section passes with its C5B PASS line and is refused with a C5B FAIL line",
               any("0 w4v_bias1_yolov8s_*.log" in p for p in z_b_none) and z_b_ok == []
               and any("no C5B yolov8s bc_e2 PASS line" in p for p in z_b_fail)
               and any("2 w4v_bias1_yolov8s_*.log" in p for p in z_b_two))

        # F4: SRC_TREE is the tree HEAD records for src/, refused unless src/ is clean against it
        t_none = src_tree(tmp)
        mod = tmp / "src" / "ignite_xdna" / "mod.py"
        mod.parent.mkdir(parents=True)
        mod.write_text("x = 1\n", encoding="utf-8")
        g("add", "-A")
        g("commit", "-q", "-m", "src")
        head_src = g("rev-parse", "HEAD:src").stdout.strip()
        t_clean = src_tree(tmp)
        mod.write_text("x = 2\n", encoding="utf-8")
        t_edit = src_tree(tmp)
        g("checkout", "--", "src/ignite_xdna/mod.py")
        new = mod.with_name("new.py")
        new.write_text("y = 1\n", encoding="utf-8")
        t_new = src_tree(tmp)
        new.unlink()
        expect("SRC_TREE is HEAD:src and wants src/ clean: refused when src/ is not in HEAD, edited or holds an "
               "untracked file",
               t_none[1] != [] and t_clean == (head_src, []) and bool(re.match(SRC_TREE_LINE, f"SRC_TREE {head_src}"))
               and t_edit[1] != [] and t_new[1] != [] and src_tree(tmp) == (head_src, []))

        # S4: an unreadable TOOLS_JSON line is a PROBLEM line, not a traceback
        bad = put("w4v_bad_desktop2_20260926.log", "\n".join(hash_lines(tmp)[:3] + ["TOOLS_JSON {not: json}"])
                  + "\n", commit=False)
        s_bad = log_problems(bad, [], want, tmp, need_commit=False)
        bad.unlink()
        expect("an unreadable TOOLS_JSON line makes its block match nothing, a PROBLEM line and not a traceback",
               hash_blocks("TOOLS_JSON {not: json}\n") == [{"TOOLS_JSON": None}]
               and any("hash blocks unlike" in p for p in s_bad))
        two = {**want, "VERDICT_CODE_SHA256": "0" * 64}
        expect("a log's hash blocks: all of them read, and a log whose blocks disagree matches nothing",
               hash_blocks(blk + "x\n" + blk) == [want, want] and logged_hashes(blk + blk_other) == {}
               and hash_blocks(blk_other)[0] == two and logged_hashes(blk) == want)
    finally:
        rm_tree(tmp)

    # R1: the modes that end a stage log print the triple and TOOLS_JSON first
    heads = {}
    child_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    for mode in (["--advanced"], ["--c8", "--logs"]):
        run = subprocess.run([sys.executable, str(ROOT / "tools" / "w4v_verdict.py"), *mode], cwd=ROOT,
                             capture_output=True, text=True, encoding="utf-8", env=child_env)
        heads[mode[0]] = run.stdout.splitlines()[:4]
    expect("--advanced and --c8 print the triple and TOOLS_JSON first, so the gate log carries them",
           all(h == hash_lines() for h in heads.values()))

    # F4: the static import graph, first on a synthetic tree, then TOOLS_JSON against the graph of TOOL_ROOTS
    with tempfile.TemporaryDirectory(prefix="w4v-selftest-graph-") as t:
        t = Path(t)
        tree = {"tools/root.py": "import importlib\nimport helper\nfrom pkg import mod_a\nimport ignite_xdna.compiler\n"
                                 "def f(name):\n    import pkg.lazy\n    importlib.import_module('pkg.dyn')\n"
                                 "    return importlib.import_module(name)\n",
                "tools/helper.py": "X = 1\n", "pkg/__init__.py": "from .rel import Y\n", "pkg/mod_a.py": "",
                "pkg/rel.py": "Y = 1\n", "pkg/lazy.py": "", "pkg/dyn.py": "", "pkg/unused.py": "",
                "ignite_xdna/__init__.py": "", "src/ignite_xdna/__init__.py": "",
                "src/ignite_xdna/compiler.py": "import pkg.unused\n"}
        for rel, text in tree.items():
            (t / rel).parent.mkdir(parents=True, exist_ok=True)
            (t / rel).write_text(text, encoding="utf-8")
        got, dyn = import_graph(("tools/root.py",), t)
    expect("the static graph: plain, from-, relative, lazy and literal importlib imports, a package's __init__, the "
           "script's own directory; nothing under src/; a non-literal importlib call is reported",
           got == {"tools/root.py", "tools/helper.py", "pkg/__init__.py", "pkg/mod_a.py", "pkg/rel.py",
                   "pkg/lazy.py", "pkg/dyn.py", "ignite_xdna/__init__.py"} and dyn == {"tools/root.py"})
    graph, dynamic = import_graph()
    runner_src = (ROOT / "scripts" / "w4v-eval.sh").read_text(encoding="utf-8")
    eval_src = (ROOT / "pipelines" / "yolov8n" / "5_eval_map.py").read_text(encoding="utf-8")
    expect(f"TOOL_FILES == the import graph of its {len(TOOL_ROOTS)} roots ({len(graph - {'tools/w4v_verdict.py'})} "
           f"repo files; this tool is VERDICT_CODE's, src/ is SRC_TREE's) plus scripts/lib.sh, which the runner "
           f"sources; the one non-literal import is the eval's --decoder, which the runner never passes and whose "
           f"default module the eval imports by name",
           "scripts/lib.sh" in TOOL_FILES and len(set(TOOL_FILES)) == len(TOOL_FILES)
           and set(TOOL_FILES) == (graph - {"tools/w4v_verdict.py"}) | {"scripts/lib.sh"}
           and not any(p.startswith("src/") for p in graph) and '/lib.sh"' in runner_src
           and dynamic == {"pipelines/yolov8n/5_eval_map.py"} and "--decoder" not in runner_src
           and 'default="npu.yolo_decode:decode_heads"' in eval_src and "npu/yolo_decode.py" in graph)

    # pycocotools on a tiny synthetic ground truth: H scoring, the dets check and the log parser
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory(prefix="w4v-selftest-") as tmp:
        root = Path(tmp)
        images, anns, k = [], [], 1
        for i in range(1, 13):
            images.append({"id": i, "file_name": f"{i:012d}.jpg", "width": 640, "height": 480})
            for _ in range(1 + i % 3):
                x, y = rng.uniform(0, 500), rng.uniform(0, 350)
                w, hh = rng.uniform(20, 120), rng.uniform(20, 120)
                anns.append({"id": k, "image_id": i, "category_id": 1 + k % 2, "bbox": [x, y, w, hh],
                             "area": w * hh, "iscrowd": 0})
                k += 1
        gt = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]}
        (root / "ann.json").write_text(json.dumps(gt), encoding="utf-8")
        calib = root / "calib"
        calib.mkdir()
        for i in (1, 5, 9):
            (calib / f"{i:012d}.jpg").write_bytes(b"")

        def dets(noise, drop):
            out = []
            for a in anns:
                if a["image_id"] in drop:
                    continue
                x, y, w, hh = a["bbox"]
                s = noise * (1 + a["image_id"] % 4)
                out.append({"image_id": a["image_id"], "category_id": a["category_id"],
                            "bbox": [x + s, y - s, w, hh], "score": 0.9})
            return out
        data = Data(root / "ann.json", calib)
        ids, h = data.sets(0)
        expect("H drops coco_calib's ids", h == [i for i in range(1, 13) if i not in (1, 5, 9)])
        files = {"b8": dets(0.0, set()), "gptq_u": dets(6.0, set()), "ar_u": dets(0.0, {1, 5})}
        logs = {}
        for arm, d in files.items():
            dp = root / f"dets_{arm}.json"
            dp.write_text(json.dumps(d), encoding="utf-8")
            all_map, h_map = score(data.coco, dp, [ids, h])
            log = root / f"w4v_decisive_yolov8n_{arm}_n5000_desktop2_20260926.log"
            text = ("\n".join(hash_lines()) + f"\nonnxruntime {ORT_VERSION}\n"
                    f"ort-opt : disable_all   ort-threads : 8\nmodel sha256 {'1' * 64}\n"
                    f"=== yolov8n | CPU | 640px | {len(ids)} images ===\nmAP@50-95  {all_map}\n"
                    f"DETS_SHA256 {sha_file(dp)} {dp.name}\n")
            log.write_text(text, encoding="utf-8")
            logs[arm] = (log, all_map, h_map)
        rows = {a: parse_eval_log(p) for a, (p, _, _) in logs.items()}
        expect("the parser reads the row, and its triple with TOOLS_JSON", rows["b8"]["map"] == logs["b8"][1]
               and rows["b8"]["images"] == 12 and rows["b8"]["dets"] == "dets_b8.json"
               and rows["b8"]["triple"] == frozen_values())
        chk = dets_check(rows["b8"], data.coco, ids, h, root)
        expect("the dets check re-scores the logged figure", chk["ok"] and chk["h"] == logs["b8"][2])
        expect("perfect detections score 100.00 on all and on H", chk["rescored"] == "100.00" and chk["h"] == "100.00")
        chk_ar = dets_check(rows["ar_u"], data.coco, ids, h, root)
        expect("H scores only its ids (detections missing on two coco_calib images lower all, not H)",
               chk_ar["ok"] and chk_ar["rescored"] != "100.00" and chk_ar["h"] == "100.00")
        chk_g = dets_check(rows["gptq_u"], data.coco, ids, h, root)
        expect("shifted boxes lower both figures", chk_g["ok"] and chk_g["rescored"] != "100.00"
               and chk_g["h"] != "100.00")
        bad_row = dict(rows["gptq_u"], map="99.99")
        bad_chk = dets_check(bad_row, data.coco, ids, h, root)
        expect("a logged figure the dets do not give fails the check, and the failure carries no H figure",
               not bad_chk["ok"] and "h" not in bad_chk)
        bad_row = dict(rows["gptq_u"], dets_sha256="0" * 64)
        expect("a dets file that is not the logged one fails the check",
               not dets_check(bad_row, data.coco, ids, h, root)["ok"])
        expect("the log name is enforced", not LOG_NAME.match("w4v_decisive_yolov8n_gptq_x_n5000_desktop2_20260926.log")
               and LOG_NAME.match("w4v_screen_yolov8s_mxgptq_e2_n500_desktop2_20260926.log") is not None)
    print(f"[selftest] {'PASS' if not fails else 'FAIL: ' + '; '.join(fails)}")
    return 0 if not fails else 1


def rm_tree(path: Path) -> None:
    """shutil.rmtree for a throwaway git repository: git's object files are read-only on Windows."""
    def writable(func, p, _):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=writable)
    else:
        shutil.rmtree(path, onerror=writable)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--hashes", action="store_true")
    mode.add_argument("--prereg", action="store_true")
    mode.add_argument("--pins", action="store_true")
    mode.add_argument("--pinned")
    mode.add_argument("--ready", choices=STAGES)
    mode.add_argument("--src-tree", action="store_true")
    mode.add_argument("--dets-sha")
    mode.add_argument("--c9", action="store_true")
    mode.add_argument("--screen", action="store_true")
    mode.add_argument("--advanced", action="store_true")
    mode.add_argument("--c8", action="store_true")
    mode.add_argument("--verdict", action="store_true")
    mode.add_argument("--zerodet", action="store_true")
    mode.add_argument("--selftest", action="store_true")
    ap.add_argument("--log")
    ap.add_argument("--dets")
    ap.add_argument("--audit")
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--logs", nargs="*", default=[])
    ap.add_argument("--controls", nargs="*", default=[])
    ap.add_argument("--tier", type=int, choices=(1, 2), default=1)
    args = ap.parse_args()
    if args.hashes:
        print_hashes()
        return 0
    if args.prereg or args.pins:
        return cmd_prereg(args, tier2=args.pins)
    if args.pinned:
        return cmd_pinned(args.pinned)
    if args.ready:
        return cmd_ready(args.ready)
    if args.src_tree:
        return cmd_src_tree()
    if args.dets_sha:
        return cmd_dets_sha(args.dets_sha)
    if args.c9:
        return cmd_c9(args)
    if args.screen:
        return cmd_screen(args)
    if args.advanced:
        return cmd_advanced()
    if args.c8:
        return cmd_c8(args)
    if args.verdict:
        return cmd_verdict(args)
    if args.zerodet:
        if not (args.log and args.dets):
            ap.error("--zerodet needs --log and --dets (and --audit)")
        return cmd_zerodet(args)
    print_hashes()
    return selftest()


# ------------------------------------------------------------ the plan (PREREG: the accepted text, verbatim)

PREREG_V24 = r"""# W4V step 5, pre-registration text, v2.4: post-training W4 recovery on YOLOv8n and YOLOv8s (2026-09-25)

Status: accepted by the gate (LF sha256 = PREREG_SHA) and embedded in `tools/w4v_verdict.py`. At acceptance no W4V
method had read an image, and nothing had run but the synthetic selftests.

The gate read v1 (LF `d42bb241`, kept as `scratch/w4v/w4v_screen_prereg_draft_v1.md`) and ruled on O1-O11. This
text applies every ruling (section 14 lists each and where it lands). From here the order is section 13's: the
step-5 commits, then START REQUEST 1.

No screen run happens before the freeze and a START REQUEST. Push is the gate's.

**This text amends W4V v2 in three places, each marked AMENDS:**
- the calibration line (v2 section 4), which is false. The user ruled (B) on the leakage (section 4);
- the "13-30 s per arm" cite (v2 section 4), which the gate asked to fix;
- the BN re-estimation route (v2 section 4), replaced by VC, a folded per-channel output-variance correction
  (section 3.3, the gate's O3 ruling).

The plan file stays unedited at LF `88b440fa`.

**Changes after the gate's read (v1 → v2):**
- **O1 (the user's (B)):**
  - K-D is on all 5,000 AND on H (sections 4 and 6);
  - the screen prints H's 468 beside the 500, and the cut stays on the 500 (section 5);
  - the (B) line goes into the DECISIONS entry (15.1a).
- **(B)'s mechanics, new in v2 (the gate has not seen these):**
  - every row writes its detections JSON through `--dets` into the git-ignored `scratch/w4v/dets/`, and its log
    prints the file's sha256;
  - the verdict re-scores all 5,000 from that JSON, and must match the logged figure, before it scores H from the
    same file;
  - H's id list is pinned (sections 6, 7 and 9);
  - a K-D delta per arm, max(Δ_all, Δ_H), orders the arms and defines NARROW (section 6);
  - PASS, the headline and NARROW count decision-eligible arms only, and F arms are printed report-only (section 6);
  - the annotation file both the eval and the verdict score against is pinned in C0 (section 7);
  - C9 also prints, report-only, whether each row's detections are byte-identical to gate D's surviving file
    (section 7);
  - the dets disk cost is MEASURED from gate D's surviving files (section 12).
- **O3:** BNR is renamed VC (folded variance correction) everywhere. It is defined as re-estimating no BatchNorm
  statistics.
- **O2:** the pair rule is the gate's reading, and the user may override it (section 6).
- **O6:** a NARROW AdaRound is a new pre-registration, never a re-run (section 3.4).
- **O10:** C8 must include at least one int32-bias form-b file, or its log states why none advanced (section 7).
- **O11:** `w4v-eval.sh` gains a `repro` stage for C9 (section 8). v1's stage list lacked it.
- **GPTQ's Hessian is written G** (section 3.5), so that H means only the held-out set.
- **Section 0:** the stale quant/DESIGN.md:709-714 gets a dated note in the step-5 commit (15.4).
- **15.1a:** the (B) line states the 64-image calibration set as the method's setting (accepted from v1), then the
  user's ruling. The ruling itself covers only K-D on all 5,000 and on H.
- **15.1b, withdrawn on the gate's later message:** the coco_calib pitfall lands on main, from the gate, in a
  docs commit that also covers main's AdaRound and GPTQ claims. It is not added on this branch, which would
  duplicate it and conflict at merge. 15.1a's (B) sentence points at it in plain text, "(see the coco_calib
  pitfall entry)". It is not a link: this branch cannot resolve the anchor before main is merged in.
- **Section 14:** the open points became the list of rulings.

Sources:
- **W4V v2**, accepted: hybrid-stack `scratch/w4v/w4v_plan_draft.md`, LF `88b440fa`, sections 4-6.
- **The census:** `eb66153` (tool) and `c4672a7` (log, LF `8185c5ee`), both CONFIRMED.
- **Gate D, frozen and read only:**
  - `tools/w4a8_emulate.py`, `tools/w4a8_verdict.py`, `tools/w4a8_engine_gate.py` and `scripts/w4a8-eval.sh`;
  - `results/int4/w4a8_accuracy_{prereg,verdict}_desktop2_20260923.log`, and the controls logs and sidecars.
- **The frozen-triple mechanics** of S1, S1b and F2-A: hybrid-stack `scratch/llm/hybrid_f2a_prereg_draft.md`,
  section 8.
- **quant/, read and not run:** `adaround.py`, `calib.py` and `pow2.py`, and the parity logs in `results/quant/`.

Labels: MEASURED, DERIVED, INFERRED, ESTIMATE, SPEC. Line cites are this branch at `c4672a7` (= main `dc6b850`
for the docs): B = docs/BENCHMARKS.md, D = docs/DECISIONS.md, MZ = docs/MODEL_ZOO_BENCHMARKS.md.

## 0. The question, and what W4V cannot change

- **The question:** does a post-training recovery method bring a W4 YOLOv8n or YOLOv8s within 1.0 point of
  mAP@50-95 of its B8, measured the gate-D way, in a scale form the engine can lower (E1, E2 or U, tag named)?
- **What decides:** K-D, per model, on the decisive 5,000-image run and on its held-out H (section 6). The
  500-image screen only cuts (section 5).
- **What cannot change:**
  - Gate B's and gate D's records stand as logged.
  - K-B is report-only, and the census is done.
  - Phase 2 (the lowering, and K-F) is decided after the verdict, on the user's ruling.
  - QAT and CLE are out (v2 section 4).
    - CLE finds 0 patterns on these SiLU graphs (MEASURED: `results/quant/prepare_probe_yolov8n_cut.log:269, :296`;
      YOLOv8s on Quark's side, `results/bench/quant_yolov8s_cut_xint8_c200.log:65`).
    - quant/DESIGN.md:709-714 still calls that a "design expectation", which is stale. The step-5 commit adds a
      dated one-line note beneath it (15.4). The paragraph itself stays as written.

## 1. Inherited from gate D, unchanged

- **The files:** the float head-cut `models/yolov8{n,s}_cut.onnx` and the shipped XINT8
  `models/yolov8{n,s}_cut_xint8.onnx`. They are pinned by gate D's prereg shas: `11b52c25…`, `f02e86ba…`,
  `f713208d…` and `065f6252…` (prereg log :41-42, :47-48). A mismatch is a STOP.
- **What changes:** only the Conv weights, plus the Conv bias for the bias-correcting methods (BC, VC). Every
  activation Q/DQ, every other scale and the graph stay as shipped.
- **The grid:** [-7, 7], 15 levels. -8 is left out because its int8 image, -128, is outside XINT8
  (`w4a8_emulate.py:7-8`).
- **pos4 per group:** Quark's MinMSE on the float weights, with bounds [-7, 7] and window -1:5; the first minimum wins.
  - This is gate D's `choose_pos` (`w4a8_emulate.py:101-121`, default window at :586).
  - It is fixed for every method except VC (section 3.3) and F (section 2).
- **Storage, per conv:**
  - form b (q4 << k under the shipped per-tensor pos8 scale, k in [0, 4]; a plain XINT8 file) when every group
    allows it;
  - otherwise form a (a per-channel scale vector, axis 0; the file is `*_ortonly.onnx`).
  - This is exactly `w4a8_emulate`'s `--form auto` (:20-26, :404).
- **The evaluation:**
  - `pipelines/yolov8n/5_eval_map.py --ep cpu --ort-opt disable_all --ort-threads 8` (conf 0.001, IoU 0.7,
    max_det 300, per-class NMS, pycocotools);
  - resnet_env17, ORT 1.23.3.dev20260320, serial, never `--fresh`;
  - the images are `sorted(coco.getImgIds())[:n]` (5_eval_map.py:169-171).
  - New for (B): every row passes `--dets scratch/w4v/dets/<row>.json` (:142, written at :239).
    - Without it the script writes `results/dets_<stem>_cpu.json` (:237), inside results/, and a later row would
      overwrite it.
    - `w4v-eval.sh` refuses an existing dets path before it calls the eval, which stays unedited.
- **"8 pinned threads" means `--ort-threads 8`,** which sets `intra_op_num_threads`, as gate D ran it
  (`w4a8-eval.sh:36`).
  - There is no CPU affinity mask in the eval path, and none is added.
  - Otherwise B8 and the RTN controls could not reproduce gate D's figures.

## 2. Scale forms and their tags (v2 section 4, unchanged)

The tag names the scale form, not the storage form: gate D's E2 and U files were both `*_ortonly`.
- **E1**, one pow2 scale per tensor. It lowers today: form b wherever k is in [0, 4], and gate D's E1 files were
  form b.
- **E2**, one pow2 scale per 32-output-channel group. It needs a shift per packet: a compiler change, phase 2.
- **U**, one pow2 scale per output channel. It needs a kernel change, phase 2. `graph_ir.scale_zp` reads channel 0
  (D:309-313), so U files are `*_ortonly`.
- **F**, a float scale per output channel. It is an upper bound only, cannot pass K-D, and is `*_ortonly`.
  - The scale is s_c = α · max|w_c| / 7, with α in {0.50, 0.51, …, 1.00} minimizing
    Σ (w_c − s_c · clip(rint(w_c / s_c), −7, 7))².
  - The arithmetic is float32 and the first minimum wins.

## 3. Methods

**Common to every method:**
- **Calibration:** section 4, the user's ruling (B).
  - The first 64 images of `data/coco_calib` by file name, letterboxed exactly as `5_eval_map.py` does
    (`npu/yolo.py:103-121`).
  - 64 is what quant/'s AdaRound used in both of its bitwise-parity runs.
  - AdaRound's memory is linear in the image count: 5,742,055,424 B peak on YOLOv8n at 64 (MEASURED,
    `results/quant/quant_yolov8n_cut_ignition_cle_adaround_c64.log:1107`).
- **Order and inputs: sequential.** Conv l's input comes from the QDQ model with every earlier conv already
  holding its new weights (and bias).
  - GPTQ, BC and VC walk the XINT8 graph's Conv order. Each captures through a sub-model to that conv's input,
    under ORT_DISABLE_ALL at 8 intra-op threads.
  - AdaRound runs quant/adaround's own sessions, as in its parity runs: ORT's default intra-op threads, and ORT's
    default graph optimisation for the float reference (`_run_quantized_inputs`, `_run_float`). It walks
    vendor_order; the ORDER control checks that both orders are topological. torch runs at 8 threads.
- **Reference:** the float head-cut model's output of the same conv, on the float network's own input
  (`adaround.py:586-587`; Quark's DropRatio 1). GPTQ is the exception: its objective uses the same quantized
  input, its published form.
- **Biases (O10, ruled):**
  - RTN, GPTQ and AdaRound keep the shipped int8 bias as it is, which is gate D's rule. So RTN rebuilds gate D's
    files byte for byte (C2).
  - BC and VC write the bias as int32 at s_x · s_w (per channel in form a).
  - The graph engine lowers an int32 conv bias since `2299f0b`: graph_ir takes int8 or int32, and the YOLOv8n
    containers rebuilt byte-identical.
  - The engine gate (C8) re-checks it on the form-b files that reach the decisive run: at least one int32-bias
    file among them whenever one advances, or else the reason none did (section 7).

### 3.1 RTN (the control)

Gate D's arms. They must rebuild gate D's files byte for byte (C2), and their mAPs must reproduce at 5,000 (C9).

### 3.2 BC: empirical bias correction

- Per conv, in order, per output channel c: b′_c = b_c + mean(y_f,c) − mean(y_q,c).
  - The mean runs over every output position of the calibration images.
  - y_q is the quantized model's conv output with the current weights and bias, on its sequential inputs.
  - y_f is the float model's conv output.
- In tier 1 it sits on RTN; in tier 2 it is a second pass over a finished AdaRound or GPTQ model.

### 3.3 VC (folded variance correction) (AMENDS v2's route; the gate's O3 ruling)

VC re-estimates no BatchNorm statistics. It is a per-channel output-variance correction on the folded conv. Per
conv, in order, on one capture of its sequential input x:
1. a_c = std(y_f,c) / std(y_q,c), with y_q from the current base-quantized layer.
2. Rescale the float weights, w′_c = a_c · w_c, and re-quantize them by the base method (tier 1: RTN, with pos4
   re-chosen on w′ by MinMSE).
3. Set the bias as BC does, on the rescaled layer. Its y_q′ is computed from the same captured x, so no new ORT run
   is needed.

**Why folded, and not v2's route through the .pt:**
- BN is already folded in both files. The head-cut ONNX has no BatchNormalization, and the head's last convs have
  none to begin with.
- v2's route re-estimated BN statistics in `models/yolov8{n,s}.pt`, refolded and re-exported. That moves every
  re-estimated channel by σ/σ′ after quantization, off the grid, so it would have to be re-quantized anyway.
- That route's export would also have to reproduce the shipped head-cut graph in every non-weight tensor.
  - Its ultralytics lives only in `resnet_env`, whose ORT is 1.22.1.
  - The shipped files were made on another host, with ORT 1.29.0 (`results/bench_s_probe.log:108, :116`).
- VC does the per-channel rescale in place, on the folded weights.

### 3.4 AdaRound at W4 (quant/, with the additive grid of section 8)

- **The call:** `quant/adaround.finetune`, the bitwise-parity transcription of Quark's fast finetune, with its
  config unchanged:
  - `FastFinetuneConfig`: seed 1705472343, batch 2, 1,000 iterations (O6, ruled), lr 0.1, early stop, reg 0.01,
    β 20→2, warm start 0.2, drop ratio 1.0, and ζ/γ 1.1/−0.1 (adaround.py:59, :64-83);
  - data size 64.
- **If a model is NARROW** (section 6) and its best decision-eligible AdaRound arm (AdaRound, AdaRound + BC or
  MX-AdaRound) is itself in (1.0, 2.0], a 10k-iteration run is a new pre-registration, never a re-run of this one.
- **The only change is the W4 override.** Per layer, the weight quantizer is the arm's form (scale 2^-pos4 per
  group, clamped to [-7, 7]) during optimisation. The result is written back in the arm's storage form.
- **It is called from `w4_recover.py`.** quant/'s CLI needs an Ignition sidecar with a calibration listing
  (quant/cli.py:149-158), which the shipped Quark files do not have.
- **The base must be the round-to-nearest shipped file.** An AdaRounded int8 base can hold -128, which
  `w4a8_emulate.conv_weights` rejects (:169-170).
- **Forms E1, E2 and U only.** No F arm, to save cost (O6, ruled).

### 3.5 GPTQ at W4

- **Layout:** W reshaped to (cout, cin·kh·kw) in ONNX order. X is the im2col of the conv's sequential quantized
  input, with its pads, strides and dilations, in the same column order.
- **The Hessian,** written G here because H names the held-out image set: G = 2XXᵀ/N in float64, over every
  output position of the 64 images, damped by 0.01 · mean(diag G).
- **A dead column** (G_ii = 0) gets G_ii = 1 and zero weights.
- **Act-order is ON** (columns in descending diag G; O9, ruled). The group scales are per output row and fixed in
  advance, so the column order does not interact with them.
- **Lazy block 128, and the Cholesky-inverse update** of Frantar et al. (2022).
- **The quantizer** is the arm's per-row group scale and [-7, 7]. It is deterministic.
- **What it generalizes:** `pipelines/yolow/3c_gptq_cv2.py`, which is 1×1 only, int8 (−128 included), per tensor,
  with no act-order and no blocks (:50-67), to k×k, the W4 grid and the forms.

### 3.6 MX: mixed W4/W8 by sensitivity (tier 2; O8, ruled)

- **The W8 set is chosen once per model and form,** before any method runs, by sensitivity:
  - the relative output error ‖y_f − y_q‖² / ‖y_f‖² of each conv's RTN-W4 weights (y_f here: float weights and
    bias on B8's input);
  - on the calibration images, with B8's own inputs (one capture).
- **The rule:** the most sensitive convs are kept at W8, most sensitive first, while the W8 share stays ≤ 5% of
  conv weight elements. For scale, E1h's seven W8 convs are 1.01% and 0.39% (MEASURED,
  `w4a8_controls_yolov8{n,s}_desktop2_20260923.log:241`).
- **MX is decision-eligible** within that budget.
- **The W8 convs keep the shipped weights exactly.** The rest go through AdaRound or GPTQ with them in place.
- **Phase-2 cost:** per-layer packet types. W8 layers save no bytes.

### 3.7 The matrix

| tier | method | forms | arms per model |
|---|---|---|---:|
| control | B8 | the shipped file | 1 |
| 1 | RTN | E1, E2, U, F | 4 |
| 1 | BC (on RTN) | E1, E2, U | 3 |
| 1 | VC (folded variance correction, on RTN) | E1, E2, U | 3 |
| 1 | AdaRound | E1, E2, U | 3 |
| 1 | GPTQ | E1, E2, U, F | 4 |
| 2 | AdaRound + BC | E1, E2, U | 3 |
| 2 | GPTQ + BC | E1, E2, U | 3 |
| 2 | MX-AdaRound | E1, E2, U | 3 |
| 2 | MX-GPTQ | E1, E2, U | 3 |

- 17 tier-1 and 12 tier-2 arms per model; with B8, 30 screen rows per model and 60 in all.
- **Not in the matrix:** VC stacked on AdaRound or GPTQ (O4, ruled). Its per-channel rescale fits the same channel
  statistics that their reconstruction already fits.
- **Tier 2 is fixed now and runs unconditionally,** as its own START REQUEST after tier 1's screen (O5, ruled).
  Nothing in it is chosen from tier 1's results.

## 4. Calibration data, and the user's ruling (B) (AMENDS v2 section 4)

**The finding (read-only, 2026-09-25):**
- v2's line "coco_calib's 300 images, which are not in val2017" is false. `data/coco_calib` is every 16th val2017
  file (`pipelines/yolov8n/2_fetch_coco.py:65-71`). All 300 names are in val2017, and the spot-check sha256s are
  byte-identical.
- `data/coco128` in this repo is a subset of coco_calib.
- No train2017 image exists anywhere on disk.
- **The shipped files were themselves calibrated inside it:**
  - B8 on 32 of these images (scripts/yolo-bench.sh:19-20; D:1065; `results/bench_s_probe.log:4, :102`);
  - `yolov8n_cut_xint8_adaround` on all 300 (yolo-bench.sh:19-20; `3b_quantize_cut.py`'s default calibration
    directory is `data/coco_calib`, :75).
  - B8's leakage is in the activation scales every arm shares, so it is inherited and equal across arms.
- A W4V method fits weights or biases to its calibration images. That is new leakage, and it favours the methods.
- **Its size, exactly** (DERIVED from 2_fetch_coco.py's stride, 5,000 // 300 = 16, over sorted names, which sort
  as the ids do):
  - the first 64 calibration images are val2017 indices 0, 16, …, 1008;
  - 32 of them (indices 0-496) are inside the screen's first 500, and those 32 are all of coco_calib's images that
    fall there;
  - all 300 are inside the decisive 5,000.

**The user's ruling, relayed by the gate (verbatim):** "O1, THE USER RULED (B). K-D per model: within 1.0 point of
B8 on all 5,000 AND on H, the 4,700 images of val2017 minus all 300 of coco_calib, each against B8 on the same
images, from the same detections. The screen prints H's 468 beside the 500, report-only. The cut stays on the 500:
leakage can only advance more arms, and H catches them at the decisive run."

**As applied:**
- **The calibration set:** the first 64 of `data/coco_calib` by name, pinned by name and sha256 (C0).
- **H is val2017 minus all 300 of coco_calib,** not only the 64 used, so it also holds none of B8's own 32
  calibration images.
  - H_5000 is 4,700 images. H_500, the screen's, is 468 (DERIVED: the first 500 minus the 32 above).
  - The id lists are fixed from `data/coco_calib`'s names and pinned by sha256 (C0, section 9).
- **There is no extra inference run.** Each H figure is pycocotools over the same row's detections JSON, with
  `params.imgIds` set to H (the same call as 5_eval_map.py:245-247).
- **Not taken:** (A), disclose only, and (C), download coco128 from train2017. No download is made.
- **The repo-wide pitfall is a DECISIONS entry that the gate lands on main** (15.1b), not on this branch. Every
  calibration-driven result evaluated on val2017 here shares it, for example:
  - quant/'s AdaRound mAPs;
  - YOLO-World's GPTQ (19 of its 64 images fall in the first 300 it was scored on).

## 5. The screen and its cut

- **What runs:** the first 500 val2017 image ids, both models, every tier-1 arm plus B8, then every tier-2 arm. One
  sitting per tier, serial, YOLOv8n then YOLOv8s.
- **The cut:**
  - An arm advances to the decisive run iff B8_500 − arm_500 ≤ 2.0, on the printed 2-decimal values. The cut
    stays on the 500 (the user's (B)).
  - F arms follow the same cut and stay report-only.
  - B8 always runs in the decisive sitting, as its anchor. RTN's 5,000-image figures come from C9's reproduction
    stage, not from the cut.
- **H's 468, report-only:** the screen prints every row's figure on H_500 and its delta against B8's H_500, beside
  the 500. They decide nothing.
  - Leakage can only make an arm look better on the 500, so it can only advance more arms (INFERRED).
  - H catches those at the decisive run.
  - Checked, report-only: the screen log flags every arm that is cut on the 500 (B8_500 − arm_500 > 2.0) but
    whose delta on H_500 (the 468) is ≤ 2.0. The flag decides nothing, because the cut stays on the 500 (the
    user's (B)). It only shows whether the inference held.
- **Every screen figure is report-only.** The screen decides nothing but the cut.
- **The cut log:** a complete screen prints SCREEN COMPLETE, then the cut (each model's advanced and cut arms) and
  the flags; an incomplete one prints PROBLEM lines and SCREEN INCOMPLETE, and no cut.
  - The engine gate, the decisive run and the verdict read the cut from exactly one committed, clean cut log per
    tier, and each model's advanced plus cut arms must be exactly its pinned arms of that tier.
- **The time, the fixed cite (AMENDS v2's "13-30 s per arm; B:2430-2443"):**
  - 13-30 s was the diagnostic's 500-image loop: 13-14 s on YOLOv8n and 30 s on YOLOv8s
    (`results/int4/diag_*_n500_desktop2_20260923.log:12`). But those ran the float models.
  - The screen runs QDQ files unoptimized. Gate D measured those loops at 242-294 s (YOLOv8n) and 548-568 s
    (YOLOv8s) over 5,000 images (MEASURED, `eval_yolov8{n,s}_*_cpu_unopt_desktop2_20260923.log:21`).
  - So about 24-29 s and 55-57 s per 500 images (DERIVED, the image loop only). Session load and pycocotools are
    not included, and are UNMEASURED.

## 6. K-D, the verdict (gate D's line, under the user's rulings)

- **Per arm, two deltas, each on the printed 2-decimal values (gate D's delta), from the same detections JSON:**
  - Δ_all = B8_5000 − arm_5000, on all 5,000;
  - Δ_H = B8_H − arm_H, on H's 4,700, B8 also scored on H.
- **An arm passes iff Δ_all ≤ 1.0 AND Δ_H ≤ 1.0.** Its K-D delta is max(Δ_all, Δ_H).
- **Per model, PASS** iff at least one decision-eligible arm passes.
- **Decision-eligible:** every arm in E1, E2 or U, of any method and either tier, MX within its budget included.
  F never is. B8 and the RTN controls are rows.
- **A form-b file must pass C8 to be decision-eligible, whatever its tag.** A FAIL, or no C8 row, and the arm is
  not. A form-a (`*_ortonly`) E2 or U arm has no engine lowering to check and stays decision-eligible.
  - An RTN arm has its 5,000-image rows from C9 whether or not the screen advanced it. A form-b RTN arm the screen
    cut has no C8 row, so it is not decision-eligible. The verdict prints it on an INFO line, "not
    decision-eligible: cut, no C8 row", which decides nothing.
- **The E1 tag needs a form-b file.**
  - An E1 arm with any form-a conv is `*_ortonly` and does not lower today.
  - Only VC can produce one, because it re-chooses pos4. AdaRound and GPTQ keep RTN's pos4, and so its k.
  - Such an arm is not decision-eligible as E1.
- **The headline names, per model, the best passing decision-eligible arm** (the smallest K-D delta; ties go to
  section 3.7's row order), with its tag, both deltas and its phase-2 cost:
  - E1 lowers today; E2 needs a shift per packet; U needs a kernel change;
  - MX adds per-layer packet types;
  - a BC or VC arm carries an int32 bias, which lowers since `2299f0b`.
- **KILLED otherwise:** "W4 killed at the tested post-training recovery set (RTN, BC, VC (folded variance
  correction), AdaRound, GPTQ, their BC stacks and MX; pow2 scales, this dialect); QAT untested."
- **NARROW:** if no decision-eligible arm passes and the best decision-eligible K-D delta is in (1.0, 2.0], the log
  adds gate D's note. It does not change the verdict. For AdaRound, see section 3.4.
- **F arms** get both deltas printed, report-only. They never enter PASS, the headline or NARROW.
- **The dets check, before any figure is used:**
  - Each row's JSON must match the sha256 its eval log printed.
  - The all-5,000 figure re-scored from it must equal the logged figure at 2 decimals.
  - Otherwise the row is missing.
- **No verdict** if any of these holds. The log ends VERDICT REFUSED (nothing scored) or VERDICT INCOMPLETE (the
  table printed, nothing decided):
  - a row is missing;
  - a row has fewer than 5,000 images, or any skipped;
  - H does not hold exactly its 4,700 pinned ids;
  - ORT or the threads differ between rows, or from the protocol;
  - a control fails or is missing;
  - the frozen triple or TOOLS_JSON differs from the prereg log's, in the tool or in any row;
  - a log it reads, or any earlier stage's log, is not committed and clean, lacks its success line or holds a
    hash block unlike the frozen values (section 9), or the cut logs break section 5's rule;
  - a form-b arm that advanced has no C8 row.
- **The pair rule** (the gate's reading of O2; the user may override it):
  - EXTEND to YOLOv8n-pose and YOLO26n iff one method passes on both models. The method is the method label; the
    forms may differ per model.
  - Otherwise the extension does not open.
- **Disclosed, not corrected:** the decisive run takes the best of up to 29 arms per model on the same 5,000
  images, and the screen's 500 are among them.

## 7. Controls (pipeline facts, PASS or FAIL; a FAIL is a STOP)

- **C0, the pins:**
  - the four model shas (section 1);
  - ORT 1.23.3.dev20260320, resnet_env17, `--ort-threads 8`;
  - the calibration listing's names and sha256s;
  - H's two id lists (4,700 and 468) and their sha256s;
  - the annotation file the eval and the verdict both score against, `data/coco/annotations/instances_val2017.json`
    (5_eval_map.py's default, :121-122, loaded at :168), with its sha256 fixed in the protocol JSON at the freeze;
  - the embedded text's sha against the accepted one. The log opens with the frozen triple and TOOLS_JSON; the
    prereg log is what freezes them (section 9).
- **C1, w8check through `w4_recover.py`'s emit path:** it rebuilds both shipped files.
  - graph_diff is empty;
  - the sha equals gate D's w8check: YOLOv8n `ec48c462…` and YOLOv8s `434ffa8b…`
    (`w4a8_controls_yolov8{n,s}_desktop2_20260923.log:76`).
- **C2, RTN rebuilds gate D's E1, E2, E1h and U files byte for byte,** by sha against gate D's prereg log
  (:43-46, :49-52).
  - The files survive in the int4-study worktree's `scratch/int4_w4a8/`.
  - E1h is emitted for this byte-identity check only; it has no screen row.
- **C3, form a == form b, bit for bit on 16 images:** RTN E1, plus one AdaRound and one GPTQ E1 arm per model
  (gate D's `compare`).
- **C4, the negative control:** `compare` tells the shipped file from RTN E1 (gate D's).
- **C5, every emitted file:**
  - graph_diff against the shipped file plus exactly the intended rewrite is empty:
    - the weights;
    - the biases for BC and VC;
    - MX's W8 convs identical to shipped;
  - the grid check;
  - float identity: stored × scale == the method's dequantized weights;
  - pos4 == MinMSE for every method except VC and F.
- **C6, quant/ int8 parity after the additive grid (section 8),** `--calib-method exact` throughout. Since
  `ebe4237` quant/ defaults to histogram mode (cli.py:58), which no parity log covers.
  - (a) Replay: `python -m quant quantize --in-model models/yolov8{n,s}_cut.onnx --cle --scales-from
    models/yolov8{n,s}_cut_xint8.onnx` gives GRAPH_DIFF_PASS on both. YOLOv8n repeats
    `prepare_probe_yolov8n_cut.log:418`; YOLOv8s is its first record.
  - (b) The recorded Quark oracle: an exact c64 CLE quantize against `models/yolov8n_cut_quark_cle_c64.onnx` (on
    disk, with its reference.json) gives INT8_EXACT True, as `diff_yolov8n_cut_ignition_cle_c64.log:9929`.
  - (c) AdaRound W8 on yolov8n c64 against `models/yolov8n_cut_quark_cle_adaround_c64.onnx` (on disk) gives
    INT8_EXACT True, as `diff_yolov8n_cut_ignition_cle_adaround_c64.log:11007-11008`.
- **C7, determinism:** GPTQ, BC, VC and AdaRound on YOLOv8n E1, each emitted twice, give the same sha.
- **C8, the engine gate** (`tools/w4a8_engine_gate.py`, mlir-aie-iron, offline, no device), on every form-b file
  that advances to the decisive run. Each file gets a `W4V_GATE <sha> PASS|FAIL` line.
  - Every layer must be exact on bus.jpg, 4 COCO images and one random input.
  - A form-b file that fails costs its arm decision-eligibility, whatever its tag (section 6). It is not a C8 FAIL.
  - The int32 coverage (O10): `W4V_CONTROL C8 int32` PASSes iff at least one int32-bias form-b file (a BC, VC,
    AdaRound + BC or GPTQ + BC arm) that advanced was checked, whatever its result, or none advanced. It FAILs only
    if one advanced and none was checked.
  - The C8 log names every other int32-bias arm and what stopped it: the cut, or a form-a conv.
  - The C8 log ends with `W4V_DONE gate`, printed only after the int32 marker PASSes. The decisive run needs both
    lines (section 9).
- **C9, B8 and RTN at 5,000 reproduce gate D exactly:** B8 27.10 / 37.21, E1 0.06 / 2.76, E2 0.02 / 2.76 and
  U 1.58 / 2.47.
  - It runs as its own stage right after the tool freeze, before any heavy stage (O11, ruled).
    - The files are byte-identical to gate D's by C2, so any drift here is the environment's.
    - Drift should surface before about 7 h of emit and screen.
  - A mismatch is a STOP.
  - Its rows also write dets and print H figures, because RTN's decisive figures come from this stage. My report
    states only whether gate D's figures match.
  - Report-only: each row's dets sha256 beside gate D's surviving file for the same row (int4-study worktree,
    `results/dets_<model>_<arm>_cpu_unopt.json`, `w4a8-eval.sh:217-219`). The mAP match decides C9; byte
    identity of the detections is printed, not required.
  - The decisive run re-runs B8 only, as its in-sitting anchor, and it must read 27.10 / 37.21 again.

## 8. The tools, and the additive quant/ grid

- **`tools/w4_recover.py`** (new, AGPL-3.0-or-later):
  - Modes: `selftest`, `controls`, `sensitivity`, `emit --model --method --form --tier`.
  - It imports `w4a8_emulate` read only (`conv_weights`, `choose_pos`, `quantize_bounded`, `groups`,
    `stem_and_head`, `emit_a`, `emit_b`, `rewire_a` and `expected_graph`). Its RTN path is gate D's own code,
    which is why C2 can demand byte identity.
  - It imports quant/ (`Graph`, `graph_diff`, `adaround.finetune`). It writes a sidecar per arm and refuses to
    overwrite:
    - gate D's fields and the method's per-layer figures;
    - the pinned models' and the arm file's shas, the frozen triple with TOOLS_JSON, and the controls' result;
    - for AdaRound, its timing, peak memory and threads (ORT's intra-op setting, torch's thread count).
    - Its paths are repo-relative, and the runner greps every sidecar for the profile name.
- **`tools/w4v_verdict.py`** (new): the prereg text, the protocol JSON, the frozen triple and TOOLS_JSON.
  - `--prereg` prints the text, the protocol JSON (with H's id-list shas), the triple and TOOLS_JSON, and the sha
    of every model, capture and tier-1 arm file. It refuses if a prereg log exists, if a model is not the pinned
    one, if the tools and every TOOLS_JSON file are not committed and clean, or if an earlier stage's log does not
    hold as section 9 says: the selftest, controls, parity, repro, sensitivity and emit1 logs, every hash block in
    them equal to its own.
  - `--pins` pins tier 2's files the same way, against the committed prereg log. It also wants the emit2 logs'
    success lines, and every hash block in them equal to its own; their commit is checked by screen2's `--ready`.
  - `--c9` checks the reproduction rows against gate D (C9). `--c8` prints C8's int32 coverage marker from the
    engine gate's lines.
  - `--pins`, `--c9`, `--screen`, `--advanced`, `--c8` and `--verdict` each print the triple and TOOLS_JSON first;
    `--prereg` prints them after the text and the protocol JSON.
  - `--screen` parses the 500-image logs and their dets into the cut list, with H_500 beside them.
  - `--advanced` lists the arms the committed cut logs advance.
  - `--verdict` parses the 5,000-image logs and their dets into K-D, on all 5,000 and on H. It reads the cut logs
    itself.
  - `--hashes`, `--pinned`, `--ready`, `--src-tree` and `--dets-sha` are the runner's checks.
  - It follows `w4a8_verdict.py`'s pattern and log-name parser.
- **`scripts/w4v-eval.sh`** (new):
  - Stages, in run order: `selftest`, `controls`, `parity`, `repro`, `emit1`, `prereg`, `screen1`, `emit2`,
    `screen2`, `gate`, `decisive` and `verdict`.
  - `refuse`, `logged` and the host-load witness are copied from `w4a8-eval.sh` (:69-90, :202-204) and extended:
    `logged` runs `no_profile` on its log, and the witness log opens with the hash block (section 9).
  - Its `scrub` extends that one. It also replaces either profile path escaped (with doubled separators), and the
    profile's name as a path component. A scrub that fails stops the stage.
  - The profile's name is read at run time, as the last path component of `USERPROFILE`, else of `HOME`, and is
    never written into a committed file. A name under 3 characters, or "user" in any case, stops the runner
    before anything runs: the scrub and its check could not be trusted.
  - `no_profile` then greps every log, sidecar and load witness (`load_w4v_*`) for that name, ignoring case. A hit
    stops the stage, and so do a missing file and a grep that cannot read the file (exit 2), which never pass
    as "no match".
  - Each stage refuses to start unless `w4v_verdict.py --ready <stage>` passes (section 9).
  - Every eval row passes `--dets` into `scratch/w4v/dets/`, refuses an existing path and logs the JSON's sha256.
  - It runs in resnet_env17, and in mlir-aie-iron for the engine gate.
  - It reaches the git-ignored `models/`, `data/` and `build/` through links to the main checkout.
    - This is how the int4-study worktree does it (`int4-study/{build,data,models}` are links).
    - They are created once, are git-ignored and are never committed.
- **The quant/ grid is additive, but it touches shared functions, hence C6:**
  - `pow2.quantize` and `pow2.sqerr`: an optional `grid=(lo, hi)`, defaulting to `qrange(dtype)`. Storage and the
    zero point stay int8.
  - `calib.choose_pow2_minmse`: optional `grid=` and `window=`, defaulting to `qrange(dtype)` and (-1, 3).
  - `adaround`: an optional per-layer weight-quantizer override (a per-row scale 2^-pos4, clamped to the grid),
    defaulting to None.
    - With None, today's path runs line for line.
    - The write-back in the arm's storage form is `w4_recover`'s.
  - Nothing else in quant/ changes: qdq emit, refine, verify and the CLI are untouched. The only other quant/ edit
    in step 5 is the dated note in DESIGN.md (15.4).
- **Frozen and untouched:**
  - `int4_bytes_gate.py`, `w4a8_emulate.py`, `w4a8_verdict.py`, `w4a8_engine_gate.py`, `w4a8-eval.sh` and
    `int4_census.py`;
  - every gate-B, gate-D and census log;
  - `pipelines/yolov8n/5_eval_map.py`, which is only called.
- **The overlap with worktree-bf16-engine:** none. It touches no file above (v2 section 6).

## 9. The frozen triple, TOOLS_JSON and the pins (as S1, S1b and F2-A)

- **PREREG_TEXT_SHA256:** this text, inside `w4v_verdict.py`.
- **PROTOCOL_JSON_SHA256:**
  - the models and their shas;
  - the arms (method × form × tier) and every method's hyperparameters;
  - the calibration listing and its shas, and the seed;
  - H's definition and its two id lists' sha256s;
  - the MX budget;
  - the screen size and cut, and K-D's line (on all 5,000 and on H) and eligibility;
  - the eval settings, ORT and the threads.
- **VERDICT_CODE_SHA256:** the whole LF source of `tools/w4v_verdict.py` and `scripts/w4v-eval.sh`, plus the
  verdict constants.
- **TOOLS_JSON,** beside it, holds the LF sha of every repo file in the import graph of the code that makes or
  judges an arm, and of `scripts/lib.sh`, which the runner sources. The graph's roots:
  - `tools/w4_recover.py`, which makes every arm;
  - `pipelines/yolov8n/5_eval_map.py`, which writes every row's detections;
  - `tools/w4a8_engine_gate.py`, which decides C8's eligibility;
  - `tools/quant_compare.py`, `quant/__main__.py` and `quant/cli.py`, which run C6.
  - The graph is read statically, and nothing is imported to read it. It takes every import statement, module-level
    or inside a function, and every `importlib.import_module` of a literal name, resolved as each root's own run
    resolves it. The eval's `--decoder` import is not literal: the runner passes no `--decoder`, and its default
    module, `npu.yolo_decode`, the eval also imports by name.
- **That is 30 files:**
  - in `tools/`: `w4_recover`, `w4a8_emulate`, `w4a8_engine_gate` and `quant_compare`;
  - `pipelines/yolov8n/5_eval_map.py`;
  - in `quant/`: `__init__`, `__main__`, `adaround`, `calib`, `cle`, `cli`, `graph`, `passes`, `pow2`, `qdq`,
    `quantize`, `refine`, `report`, `shift_cut`, `sources` and `verify`;
  - in `npu/`: `__init__`, `modnet`, `paths`, `preprocess`, `session`, `yolo` and `yolo_decode`;
  - `ignite_xdna/__init__.py`, the repo-root forwarder into `src/`;
  - `scripts/lib.sh`.
  - `w4v_verdict.py`, which the graph also reaches, is VERDICT_CODE_SHA256's. The selftest checks the list against
    the graph by set equality (section 10).
- **`src/` is not in the graph.** The gate stage records it instead, as `SRC_TREE <tree>`, the tree
  `git rev-parse HEAD:src` names (`w4v_verdict.py --src-tree`).
  - It refuses before the engine gate runs unless `src/` is clean against HEAD, with no staged, unstaged or
    untracked path. `--c8` refuses too if it is not.
  - The line records which compiler C8 judged with.
- **All three, and TOOLS_JSON, are printed in every W4V log, the load witnesses included.** `w4_recover` imports
  them. `--pins`, `--screen`, `--advanced`, `--c8` and `--verdict` refuse unless all four equal those of the one
  prereg log, committed and clean.
- **Committed means in HEAD and unchanged against it** (`git cat-file -e HEAD:<path>` and `git diff --quiet HEAD
  -- <path>`); a file only `git add`ed is not committed.
- **No stage runs until every earlier stage's logs hold.** `w4v_verdict.py --ready <stage>` checks them, and
  `w4v-eval.sh` calls it before every stage after the selftest, in section 13's order. `--prereg`, `--pins`,
  `--c9`, `--screen`, `--c8` and `--verdict` apply the same check themselves. Each earlier stage's logs need:
  - exactly one log per expected name, committed and clean;
  - its success line, printed only when that stage finished: the selftest's two `[selftest] PASS` lines; each
    model's `[controls] <model> PASS`; the PASS lines of C6a (both models), C6b and C6c; a `DETS_SHA256` line in
    every eval row; C9's PASS for both models; the sensitivity and emit logs' `W4V_DONE` lines; the prereg and
    pins logs' `PINNED` count; each cut log's `SCREEN COMPLETE`; and the gate log's `SRC_TREE` line, its C8 int32
    PASS and its `W4V_DONE gate`;
  - no `W4V_CONTROL` line that FAILs;
  - at least one hash block, and every hash block equal to the tool's own. A log of several tool calls, such as
    an emit log, holds one block per call, and a log whose blocks differ from each other refuses.
  - A log that exists but fails any of these refuses the next stage, as a missing log does. So no emit runs
    before C9's log is committed with a PASS on both models.
  - The prereg run also refuses unless the tools and every TOOLS_JSON file are committed and clean.
- **Nor does a stage run while a log of its own exists,** of any date, committed or not: a restart needs the gate's
  ruling.
  - The runner's `--ready` checks this before the stage writes anything. It covers the stage's logs above, its
    load witness and C6c's report-only env17 log.
  - The in-stage checks (`--c9`, `--screen`, `--c8`, `--prereg`, `--pins` and `--verdict`) run inside their stage,
    once its `--ready` has passed, so they leave this check to it.
- **The prereg run pins every emitted arm's sha before any mAP.**
  - The screen and the decisive run refuse a file whose sha differs.
  - Tier 2's shas are pinned by a prereg-pins log, with the same triple and TOOLS_JSON, before tier 2's screen.
  - Pins are read only from the one prereg log and at most one pins log, each committed and clean.

## 10. Blind-safety

- **Everything in sections 3-7 is fixed in the tools** before any W4V method reads a calibration image.
- **The selftest is synthetic only:**
  - GPTQ on tiny random convs: equal to an independent unblocked OBQ reference, never worse than RTN in any case,
    and at the brute-force optimum in at least 30 of 40;
  - the BC and VC identities, and MX's sensitivity on B8's captured input;
  - form a/b identity;
  - the verdict's rule, the H scoring and the dets check, on synthetic logs and JSONs;
  - the freeze, the pins, the cut logs and the commit checks, in a temporary git repository;
  - there too, the stage order: a later stage refuses on a committed log that FAILs, a log cut short of its
    success line, a missing or differing hash block, and an uncommitted log;
  - that `--advanced` and `--c8` print the triple and TOOLS_JSON first;
  - that TOOLS_JSON's list is the import graph of its roots plus `scripts/lib.sh` (section 9).
- **Controls, parity and emit print pipeline facts only:**
  - shas, counts, PASS or FAIL;
  - per-layer reconstruction loss and SQNR, report-only, as gate D's sidecars printed SQNR before its evaluation.
- **The screen logs print each arm's 500-image mAP and its H_500 figure** (their purpose) and the cut list. My
  report after a screen carries the cut list and counts, not the mAPs (O7, ruled).
- **No re-runs:**
  - a failed stage stops and comes to the gate as it stands;
  - a crashed arm is a missing row (INCOMPLETE), restarted only on the gate's ruling.
- **The verdict runs once,** on the gate's go.

## 11. Predictions (ESTIMATE; scored HIT or MISS in the verdict report; they decide nothing)

- **W4V-P2** (v2): RTN reproduces gate D exactly (C2, C9).
- **W4V-P3** (v2): YOLOv8n is harder to recover than YOLOv8s.
- **W4V-P4:** no E1 arm passes on either model.
- **W4V-P5:** in U, GPTQ and AdaRound each beat RTN by more than 10 points on the screen, on both models.
- **W4V-P1** concerns the census. It is scored in the verdict report against the census log, on its ms items only
  (the ring has no ms share).

## 12. Cost, and the START REQUESTs (ESTIMATE unless tagged)

| stage | work | cost | kind |
|---|---|---|---|
| selftest | synthetic | seconds | light (FYI and DONE) |
| controls + parity | C0-C6 | about 20 min: C6(b)'s calibration 244.5 s and C6(c)'s AdaRound 511.97 s were MEASURED on YOLOv8n | START REQUEST 1 |
| reproduction (C9) | B8 and RTN E1, E2, U at 5,000, both models: 8 rows | about 1 h: gate D's loops were 242-294 s and 548-568 s (MEASURED) | START REQUEST 1b |
| emit tier 1 | 17 arms × 2 models | about 3.5 h | START REQUEST 2 |
| prereg | pins | minutes | light |
| screen tier 1 | 18 rows × 2 models at 500 | about 35 min | START REQUEST 3 |
| emit tier 2 + pins | 12 arms × 2 models | about 3.5 h | START REQUEST 4 |
| screen tier 2 | 12 rows × 2 models | about 25 min | START REQUEST 5 |
| engine gate | form-b files that advance | minutes each | light |
| decisive | B8 per model, plus the advancing arms | about 15 min, plus 4-5 min (n) or 9-10 min (s) per advancing arm | START REQUEST 6 |
| verdict | parse, re-score all 5,000 and score H per row | minutes (a pycocotools pass per figure, UNMEASURED) | light |

**The bases of the estimates:**
- **AdaRound:** 511.97 s per YOLOv8n form at W8 (MEASURED: 157.3 s ORT + 353.2 s torch, c64 log :1012, :1107).
  YOLOv8s is UNMEASURED; taken at 3× (its conv weights are 3.5×, per the census).
- **GPTQ, BC and VC captures:** at most 63 sub-model runs × 64 images per pass, each under one full forward
  (42.89 / 104.92 ms per image, MEASURED, gate D's B8 unopt logs). That is ≤ 3 min (n) and ≤ 7 min (s) per pass.
- **The float-side statistics** (y_f) are captured once per model and shared by every arm.
- **Memory:**
  - AdaRound peaked at 5.74 GB on YOLOv8n at 64 images (MEASURED).
  - YOLOv8s is UNMEASURED; about 10-15 GB is an ESTIMATE, against MODNet's 22.3 GB at 64 (MEASURED, the repo's
    largest).
  - BFP16's needs go in each START REQUEST.
- **Disk:** one detections JSON per row, 60 screen rows plus the 5,000-image rows, in `scratch/w4v/dets/`.
  - Gate D's surviving files, in the int4-study worktree's `results/`, are MEASURED at 21,275,566 B (YOLOv8s U) to
    67,668,653 B (YOLOv8n B8) per 5,000-image row, and 6,424,894 B (n) and 5,536,967 B (s) per 500-image B8 row.
  - So the screen needs under 0.5 GB (ESTIMATE, 60 rows at up to about 7 MB). Each 5,000-image row needs up to about
    70 MB (ESTIMATE). Every row prints its file's size.

## 13. Files and order

1. v1 to the gate (done); its rulings on O1-O11 (done); v2 to v2.4; the gate's acceptance (LF).
2. The diff (`w4_recover.py`, `w4v_verdict.py`, `w4v-eval.sh` and the quant/ grid), with the selftest's output,
   for the gate's go.
3. **The step-5 commit(s)**, then HEAD reported:
   - the frozen tools with the selftest log, which the controls stage needs committed;
   - the DECISIONS entry (15.1a); the pitfall entry is the gate's, on main (15.1b);
   - the BENCHMARKS gate-B line (15.2);
   - the results/README census row (15.3);
   - the quant/DESIGN.md dated note (15.4).
4. START REQUEST 1: controls and parity.
   - The controls logs are committed before parity runs, then the parity logs; each commit holds logs alone, with
     a pipeline-facts report.
   - Then START REQUEST 1b: C9's reproduction. Its logs are committed. Gate D's figures are known, so the report
     states whether they match.
5. START REQUEST 2: emit tier 1; its sensitivity and emit logs are committed. Then the prereg run (it pins every
   tier-1 arm), committed before any mAP.
6. START REQUEST 3: screen tier 1, with B8 at 500. The log is committed, and the report is the cut list.
7. START REQUEST 4: emit tier 2; its emit logs and its pins log are committed. Then START REQUEST 5: screen tier 2.
8. The engine gate on every advancing form-b file (light). Its log is committed before the decisive run.
9. START REQUEST 6: the decisive run. The logs are committed.
10. The verdict, once, on the gate's go.

Logs are `results/int4/w4v_*`, and the load witnesses `results/int4/load_w4v_*`: UTF-8, the profile path scrubbed,
never overwritten, each committed alone. Arm files, captures and detections JSONs stay in the git-ignored
`scratch/w4v/`.

## 14. The rulings on v1's open points, as applied

| point | ruling | where it lands |
|---|---|---|
| O1 | THE USER RULED (B): K-D on all 5,000 AND on H (4,700); the screen prints H's 468, report-only; the cut stays on the 500 | sections 4, 5, 6, 7 (C0), 9; 15.1a |
| O2 | the gate's reading: ONE METHOD passing on BOTH models opens pose and YOLO26n; the user may override | section 6 |
| O3 | folded, accepted, named "VC (folded variance correction)", never bare "BN re-estimation" | sections 1, 3, 3.3, 3.7, 6, 7, 10, 12; 15.1a |
| O4 | no VC stacks | section 3.7 |
| O5 | tier 2 unconditional | section 3.7 |
| O6 | 1,000 iterations; a NARROW AdaRound means a new pre-registration at 10k, never a re-run | section 3.4 |
| O7 | the cut list only in reports | section 10 |
| O8 | ≤ 5%, decision-eligible | section 3.6 |
| O9 | act-order ON | section 3.5 |
| O10 | int32 for BC and VC; C8 includes ≥ 1 int32-bias form-b file or states why none advanced | sections 3, 7 (C8) |
| O11 | C9 as its own stage (the `repro` stage) | sections 7 (C9), 8, 12, 13 |
| docs | 15.1: accepted, with the (B) line and the VC name; the pitfall TAKEN, with the adaround fact, then moved to main by the gate; 15.2 and 15.3 accepted; the DESIGN.md note | 15.1a, 15.1b, 15.4 |

## 15. Text for the step-5 commit (the gate's rulings on steps 4 and 5)

### 15.1a docs/DECISIONS.md: after :975, a sibling of gate D's bullet under "Rejected: int4 weights in the graph engine"

```
  - **The user paused the LLM work and turned int4 to the vision engine's models (2026-09-25).**
    "pause the LLM work and focus on INT4 as a broader function, so i think we should use the models
    we already know this chip is good at." (the user's direction, relayed by the gate)
    - Criterion (b): accuracy is the gate. int4 is a capability: a W4 model within 1.0 point of its
      B8, measured the gate-D way, in a scale form the engine can lower, with the form named. Speed
      and energy are report-only; bytes decide nothing.
    - YOLOv8n and YOLOv8s first; YOLOv8n-pose and YOLO26n only if a method works on that pair.
    - The engine lowering, which collides with the bf16 branch's files, is decided after the
      accuracy screen.
    - Gate B and gate D stand. This is gate D's reopen attempt: post-training recovery at W4, with
      bias correction; VC (folded variance correction), a per-channel output-variance correction
      that re-estimates no BatchNorm statistics; AdaRound; GPTQ; and mixed W4/W8. It is
      pre-registered in `tools/w4v_verdict.py` before any method reads an image. QAT is out unless
      every post-training method fails and the user rules for it.
    - The methods calibrate on the first 64 images of `data/coco_calib`, which is inside val2017
      (see the coco_calib pitfall entry). The user ruled (B) on that leakage: K-D passes a model
      only within 1.0 point of B8 both on all 5,000 images and on H, the 4,700 images of val2017
      outside `data/coco_calib`, each against B8 on the same images, from the same detections.
    - A byte census ran first and decides nothing
      (`results/int4/w4v_census_desktop2_20260925.log`).
    - Plan W4V v2 (git-ignored, LF `88b440fa`); branch `worktree-int4-vision`.
```

### 15.1b docs/DECISIONS.md: the coco_calib pitfall, NOT on this branch

- The gate lands it on main, in a docs commit that also covers main's AdaRound and GPTQ claims.
- This branch adds nothing there. 15.1a names the entry in plain text until main is merged in.
- My offered text, and the facts I verified for it, went to the gate with v1 and v2:
  - `2_fetch_coco.py:65-71`;
  - `yolo-bench.sh:19-20`;
  - `3b_quantize_cut.py:75`.

### 15.2 docs/BENCHMARKS.md: after :2332 (gate B's reopen bullet), before "**Gate C**" at :2334 (accepted as v1)

```
- **Correction (2026-09-25): two denominators above were not the smallest measured dispatch.**
  SESR-M7 also measured 4.178 ms and YOLOv8n-pose 7.527 ms (the energy sitting,
  [the balanced default](#the-amd-comparisons-re-measured-in-the-balanced-default-2026-09-16-desktop-2)),
  against the 4.208 and 7.548 ms used. On them, SESR-M7's int4 bound moves from 1.365% to 1.375% and
  its ceiling from 1.674% to 1.686%; YOLOv8n-pose's from 2.762% to 2.770% and from 4.159% to 4.170%
  (DERIVED, recomputed from the log's bytes and task counts, which the log prints to one decimal as
  1.4 / 1.7 / 2.8 / 4.2%). Every move is under 0.02 points, the verdict stands, and the log is
  unchanged.
```

### 15.3 results/README.md: the `int4/` row (:108), a sentence appended before the link cell (accepted as v1)

```
Then W4V (2026-09-25): `w4v_census_desktop2_20260925.log`, the int4 byte census of nine engine rows
(eight models, ten containers) and six AMD-stack models from `tools/int4_census.py` (no device; report-only, it decides
nothing; every file read is listed with its sha256, and every trust check passes).
```

### 15.4 quant/DESIGN.md: a dated one-line note after :714, as its own paragraph before "**`quant/refine.py`**" (:716)

```
Note (2026-09-25): yolov8n-cut's 0 patterns is MEASURED, no longer an expectation: Quark and quant/ both count 0 on the prepared graph (`results/quant/prepare_probe_yolov8n_cut.log:269, :296`).
```
"""

PREREG_V25 = r"""# W4V pre-registration v2.5: the amendment after the screen1 STOP (DRAFT, 2026-09-26)

**Status:** accepted by the user on 2026-09-26 after the gate's review. It is the reviewed draft (LF `dfea7430…`) unchanged except for this Status block; the title keeps the draft's wording. It is embedded in `tools/w4v_verdict.py` as PREREG's v2.5 part.
- Nothing runs under v2.5 before its prereg stage freezes the triple.
- v2.5 is v2.4 with the sections below added. Section references are to v2.4's text (`tools/w4v_verdict.py` PREREG,
  LF `7ff9c2c4…`).
- No frozen tool changes: `5_eval_map.py`, `w4_recover.py`, `w4a8_*`, `quant/` and `npu/` stay as they are.
- What changes:
  - the text;
  - the protocol JSON, by one added key, `v2_5`;
  - the verdict code: `w4v_verdict.py` and `w4v-eval.sh`;
  - TOOLS_JSON, by one added root, `tools/w4v_bias_audit.py`;
  - new evidence files.
- Figures carry MEASURED, DERIVED or ESTIMATE. None of them is an mAP or H figure.

---

## A. Preamble: what happened, and what is on the record

**The STOP (MEASURED, `scratch/w4v/monitor/ram_screen1.log` and the runner's output).**
- screen1 ran on the gate's GO from 07:27:40 to 07:42:12Z, 872 s, and ended with rc 1.
- Row 24 of 36, `w4v_screen_yolov8s_bc_e1_n500`, printed `0 detections over 500 images` (5_eval_map.py:233). It wrote
  its detections file (:239-240), then raised `SystemExit("no detections at all -- something is broken upstream")`
  (:243). The runner stopped the stage, as §10 requires.
  - Its file was the pinned one (`2700ece0…`). Its setup was normal.
- 23 rows were complete and clean: all 18 YOLOv8n rows, plus YOLOv8s B8 and rtn_e1, rtn_e2, rtn_u, rtn_f.
- 12 YOLOv8s rows and the cut never ran. Under §10 the screen is INCOMPLETE.
- **The held files, never committed and not read since:**
  - the 24 row logs;
  - the load witness;
  - the 24 detections files in `scratch/w4v/dets/`.

**No metric line was read (INT4's statement, for the record).**
- INT4 has not read the metric line of any completed screen1 row.
- Method, the session transcript: no tool result since 2026-09-26T07:27:00Z carries an mAP-like figure (`mAP…`, `AP`,
  `H_500` or `H (n images)` followed by a 2-decimal number). The last metric line in any tool result of that session
  is dated 2026-09-25T20:14Z and carries no W4V stem.
- The runner's console output holds 23 metric lines. It was never opened; it was only count-grepped.
- The row logs, the witness and the detections were never opened. INT4's per-file check prints only `DETS_SHA256`,
  `W4V_CONTROL`, `W4V_REPORT` and `Traceback` lines.
- `0 detections over 500 images` is the eval's detection-count line (:233), not a metric line. The failing row never
  reached the metric print (:250).

**The audit (the gate's ruling of 2026-09-26; criteria fixed before any run).**
- **The criteria.** An arm is a DEFECT if any of these holds:
  - (a) in any Conv, the emitted int32 bias, dequantized at its stored scale, differs from an independent
    recomputation of §3.2 by more than 1 LSB (of s_x·s_w for that channel) in any channel;
  - (b) any Conv's int32 bias scale is not its s_x·s_w as read from the QDQ graph;
  - (c) any bias channel is clamped or saturated short of `make_bias`'s overflow refusal;
  - (d) any int32 bias is wired to the wrong Conv, has the wrong length, or carries a nonzero zero point or a scale
    other than (b)'s.
- **The script:** `w4v_bc_audit.py`, sha256 `dbc4b7bb4d2d0195f4becf479b4711732f62d0a9f3952fcbfc3c06e34741bd89`.
  - It is new code written from §3.2 and never imports `w4_recover.py` or `quant/`.
  - The gate read it and accepted it before any (a) result existed.
  - It has two walks:
    - SEQ: an independent sequential walk from the pinned RTN file;
    - ANCHORED: the fixed-point test on the arm file's own sequential inputs.
  - Its self-checks are bit-identity checks.
- **Results (MEASURED; logs in `scratch/w4v/audit/`, cited under B.6):**

  | arm | (a) SEQ | (a) ANCHORED | (b) | (c) | (d) | outcome |
  |---|---|---|---|---|---|---|
  | yolov8n bc_e1 (the control) | 0/63 Conv beyond 1 LSB, max 0.500 | 0/63, max 0.500 | False | False | False | GENUINE |
  | yolov8s bc_e1 | 0/63, max 0.500 | 0/63, max 0.500 | False | False | False | GENUINE |
  | yolov8s bc_e2, bc_u, vc_e1, vc_e2, vc_u | not run (static only) | not run | False | False (static) | False | — |

  - The emitted int32 equals SEQ's on 63/63 Conv for yolov8s bc_e1.
  - No channel is clamped, and no SEQ value overflows.
  - Every self-check is True. No import of `w4_recover.py` or `quant/`.
  - Every file equals its prereg PIN, and the calibration listing equals its pin.
- **Tensor facts (MEASURED, 16 C3 images, letterboxed as the eval does):**
  - (i) class-logit outputs at the output quantizer's minimum code: 0/10,752,000 for rtn_e1, and the same for bc_e1;
  - (ii) no Conv has bc_e1's per-channel mean error above rtn_e1's in more than half its channels.
- **The mechanism of the zero detections is UNEXPLAINED.** v2.5 states no mechanism.
  - The audit rules out a bias-emission defect on the pre-stated criterion.
  - (i) shows that no class logit of either arm sits at the output quantizer's floor.
  - (ii) shows that no Conv has BC's mean error exceeding RTN's in most of its channels.
  - Neither fact explains the row.

**The machine (the gate's dating from run timings; the user replaced the part).**
- Desktop 2's CPU fan failed between about 07:42 and 08:08Z on 2026-09-26. It was replaced before the 12:25:44Z
  reboot, and is the likely cause of the two unclean stops:
  - after the 08:08Z audit run's last write at 08:24:43Z;
  - at about 11:27Z, during the 11:11Z run.
  - 08:23:48Z is event 6008's last-alive stamp, not the stop.
- Every W4V log up to 07:42Z predates it: controls, parity, repro, emit1, prereg and the held screen1 rows.
- The audit runs at 08:08, 10:58 and 11:11Z ran throttled, about 7-8x slow. Their results do not depend on speed.
- **Partial and superseded, never deleted:**
  - `recompute_yolov8n_desktop2_20260926.log` (CONV 0-40 of 63), superseded by `recompute_yolov8n_rerun_…`;
  - `recompute_yolov8s_rerun_desktop2_20260926.log` (CONV 0-19), superseded by `recompute_yolov8s_rerun2_…`.
- From v2.5 on, any W4V run longer than 15 min holds:
  - the per-run keep-awake (`SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`, released at the runner's
    exit; no power setting changes);
  - HWiNFO CSV logging (`Documents\HWiNFO-logs`);
  - the RAM monitor.
  - Known limit: the RAM monitor's psutil tree does not reach a python started under Git Bash, because the msys fork
    stub exits. Its available-RAM samples are right, but its stop does not reach the python. That python's pid is
    stopped by hand on two samples under 2 GB.

**C5, corrected.**
- §7 lists C5's bias item as part of "graph_diff against the shipped file plus exactly the intended rewrite". It is
  more than that, and less than an independent check.
- `check_arm` (w4_recover.py:344-358) checks each int32 bias numerically: its scale must equal the method's own
  `float32(s_x) · w_scale(row)`, and `i32 · scale` must lie within half a step of the method's own float b′. It
  printed 63/63 for every BC and VC arm in emit1.
- The check is self-referential: it compares b′ with itself. It verifies the rounding and the scale of the method's
  output, never b′ against §3.2.
- The graph_diff part is value-blind for biases, because the expected graph takes its bias values from the emitted
  file (:290-294).
- C5b (section C) is the independent check.

**Conv order, INFO.**
- §3 says GPTQ, BC and VC "walk the XINT8 graph's Conv order". BC walked the emitted files' Conv node order: the emit1
  `[seq]` lines equal the arm file's order.
- That order differs from the shipped XINT8 file's node order from position 38, where the model.18 and model.21 cv1
  convs trade places with the model.22 cv2.x.1 and cv3.x.1 convs.
- All three orders (shipped, arm file, walked) are dependency-consistent in the shipped graph, with 0 Conv pairs out
  of order (MEASURED). A sequential walk therefore gives the same result in each.
- So it is a textual mismatch, with no numeric consequence. v2.5 reads §3's phrase as "a dependency-consistent order
  of the XINT8 graph's Convs".

---

## B. Zero-detection rows (new §5.1; applies to every eval row W4V scores)

**B.1 Definition.** A *zero-detection row* is an eval row whose `5_eval_map.py` run did all of these:
- printed `0 detections over <N> images in <s>s` (:233), with N equal to the row's image count (500 or 5,000);
- printed `wrote <its --dets path>` (:240);
- exited with `no detections at all -- something is broken upstream` (:243), rc 1.

It also has these properties:
- its detections file exists and parses as an empty JSON list: count 0; `[]`, 2 bytes, for the held row;
- its log has no `DETS_SHA256` line, since the runner prints that only after the eval succeeds;
- its `model sha256` line (:186) equals the arm's pinned sha;
- it has no `skip unreadable` line.

**B.2 The figure is defined by rule.**
- pycocotools 2.0.11 cannot score an empty result list: `COCO.loadRes` indexes `anns[0]` (coco.py:330) after its two
  asserts.
- Zero detections give zero recall at every IoU threshold. v2.5 **defines** a zero-detection row's figures as **AP
  0.00 on its N images and 0.00 on H**, from the same (empty) detections file. Nothing is rescored.
- The definition agrees with COCOeval's own code on these image sets (source read in resnet_env17; nothing scored):
  - `evaluateImg` returns None only for an image with neither ground truth nor detections (cocoeval.py:247-248);
  - so for every category with non-ignored ground truth, zero detections give recall 0 (:392), and precision stays 0
    through the `except: pass` path (:403-409);
  - the -1 "absent" sentinel (:334) survives only for categories with no ground truth (:360-361, :373-374), and
    `summarize` leaves those out (:452-453).
  - The 500 and H_500 both hold ground truth, so COCOeval, if it could load the empty list, would give 0.00, not -1.

**B.3 When a zero-detection row is recorded.**
- **Arms with int32 biases** (BC, VC, AdaRound + BC, GPTQ + BC):
  - the row is recorded as AP 0.00 iff a committed audit log for **that exact arm file sha** ends `HELD … (a) False
    (ANCHORED agrees: True) (b) False (c) False (d) False -> GENUINE`;
  - that log must come from the accepted script (dbc4b7bb) or from C5b's tool (section C), and must print `IMPORTS …
    none`;
  - otherwise the stage stops. The arm is audited on (a)-(d) before anything else, and the result goes to the gate.
- **A DEFECT:**
  - the stage stops; the arm gets no row;
  - the fix in `w4_recover.py`, its validation by (a)-(d) on the affected arm and on one GENUINE control arm, the
    re-emit of every arm the fixed code path emits, and their re-pin form a **v2.6** amendment, which goes to the gate
    and to the user;
  - "detections appeared" never validates a fix.
- **Arms with no int32 bias** (RTN, AdaRound, GPTQ, MX): (a)-(d) are vacuous there, since their biases are the
  shipped int8 ones.
  - Recommended: a zero-detection row of such an arm is a STOP, as in v2.4. It goes to the gate, with no AP 0.00 by
    rule. (Decision D2.)
  - Why: their emitted weights are checked only by C5's self-referential float identity, and no pre-registered audit
    covers them. Recording 0.00 would enter an unexplained, unchecked result as a method result.
- **B8, and the RTN rows of C9:** a zero-detection row there is always a STOP. B8 is the shipped file, and C9 is a
  control.

**B.4 The completion record (no frozen tool edited, the row log untouched).**
- A new command, `w4v_verdict.py --zerodet --log <row log> --dets <dets file> --audit <audit log>`, writes
  `results/int4/w4v_zerodet_<model>_<arm>_n<N>_<machine>_<date>.log`.
- **What it prints:**
  - the hash block;
  - the row log's name and LF sha256;
  - the dets path, its sha256 and the count 0;
  - the audit log's name, its LF sha, its HELD line and the arm file sha it covers;
  - `ZERO_DET_ROW <row> AP_N 0.00 AP_H 0.00 (rule B.2)`.
- **It refuses unless** B.1 holds line by line, and the audit log is committed and clean and satisfies B.3.
  - The arm file's sha is taken from the audit log's `PIN_CHECK` line for the arm file, never from a log name, and
    must equal the arm's current pin (`pins()`). So an arm re-pinned after a v2.6 can never cite a stale audit.
- **The row log stays byte-identical.** The record is the only added file.
- **In the runner** (`w4v-eval.sh` `eval_row`), when `row_body` fails:
  - it checks for B.1's exact exit;
  - if B.1 holds, it looks up the arm's audit log in the protocol's `v2_5.audits` table and calls `--zerodet`;
  - a refusal stops the stage exactly as in v2.4;
  - a record lets the stage go on to the next row.
- **In `ready()`, `stage_logs()` and `--screen`:**
  - a screen or decisive row counts as present iff its log has `DETS_SHA256` (v2.4) or a committed, clean
    `w4v_zerodet_*` record names it and re-verifies (B.1 and the dets sha);
  - `parse_eval_log` reads such a row's image count from :233 and its ORT, thread and model-sha lines from :184-186;
  - `dets_check` is bypassed for it, and its figures are 0.00 and 0.00 by rule.
  - The cut, the flag and K-D then apply unchanged. B8_500 − 0.00 > 2.0, so the arm is cut. It is not flagged, since
    B8_H − 0.00 > 2.0. It can never pass K-D.

**B.5 Scope.**
- B applies to every eval row W4V scores: both screens and the decisive run.
- **emit2 arms:**
  - the tier-2 +BC arms (AdaRound + BC, GPTQ + BC) fall under B.3's int32-bias branch. Their C5b log (section C) is
    their audit, so a zero-detection row with a C5b PASS is recorded directly;
  - MX-AdaRound and MX-GPTQ carry no int32 bias. They fall under the no-bias branch: STOP, if D2 is taken as
    recommended.
- A decisive row cannot be a zero-detection row of an advanced arm unless its 5,000-image run differs from its screen
  in kind. The rule still covers it.

**B.6 The audit's evidence is committed before it is cited.**
- **The audit logs** move from `scratch/w4v/audit/` to a new `results/int4/w4v_audit/`: names unchanged,
  byte-identical, with a `results/README.md` row.
  - They are CRLF, so each is labelled "working copy X; its LF blob Y".
  - Every one first passes the gate's detection pass (D.2).

  | log | what | LF sha256 |
  |---|---|---|
  | static_yolov8n_desktop2_20260926.log | (b)-(d), yolov8n bc_e1 | `0f04d645ff9d6d42480d36d2d70ffae126ab3a1968983ac7279876dcedbc831e` |
  | static_yolov8s_desktop2_20260926.log | (b)-(d), 6 yolov8s arms | `2225f7179fe1959325d384635d931c73e7af213068c37daf214ecc98ba89d571` |
  | recompute_yolov8n_rerun_desktop2_20260926.log | (a)-(d), yolov8n bc_e1: GENUINE | `6cb4d247199a46e0dc5ae0843003c73b9dde043791a7e0508e8aa6d5c63950f7` |
  | recompute_yolov8s_rerun2_desktop2_20260926.log | (a)-(d), yolov8s bc_e1: GENUINE | `8ac21b7e3b00ecf91467095c367c039d14c0c1dfc395da7b0d4781f835e0c3fd` |
  | tensors_yolov8s_rerun2_desktop2_20260926.log | (i), (ii) | `d8f2e5dab3c3cd116f5cfc00cb9704208077715d79bf021314ff61b4bc9f4f39` |
  | recompute_yolov8n_desktop2_20260926.log | PARTIAL (CONV 0-40), superseded | `b52aa5378198397b8e9b60fd6b0f89b5c56956f8e36dc3a14ba72c5a3392801b` |
  | recompute_yolov8s_rerun_desktop2_20260926.log | PARTIAL (CONV 0-19), superseded | `6217213499e82dcfe72fe82d6e4b632a308774dbf8cef95a6d7b0193d17a0a47` |

- **The script** is committed byte-identical (sha `dbc4b7bb…`) beside the logs as evidence of what ran, not as a tool:
  `results/int4/w4v_audit/w4v_bc_audit.py`.
  - Its docstring still says "uncommitted". That is stale, and the line is kept so that the sha holds.
  - Where it lives is decision D4.
- **Not committed:** the keep-awake helper and its logs, the sequence scripts and the RAM monitor logs. They are
  scratch helpers, and the gate's rule keeps the helper uncommitted. They hold profile paths.

---

## C. C5b, the independent bias check (new control; the gate's preference)

**The tool:** `tools/w4v_bias_audit.py`, a new TOOLS_JSON root (AGPL header).
- It is `w4v_bc_audit.py` (dbc4b7bb) with the arm and its base as parameters. Its static (b)-(d), SEQ, ANCHORED and
  self-check code stay line for line, so the gate can diff it against dbc4b7bb.
- It reads its pins through `w4v_verdict.pins()`, not a hard-coded prereg log name.
- It drops the tensors mode, which was a diagnostic.
- It prints dbc4b7bb's `IMPORTS w4_recover or quant loaded: none` line.
- Its new sha goes to the gate before any use.
- **Its TOOLS_JSON addition**, as the import-graph code (`import_graph`, :242) computes it:
  - it reaches `w4v_verdict.py`, which is VERDICT_CODE's and excepted from TOOLS_JSON (:223-225, and the selftest's
    set check at :1564). `w4_recover.py` already reaches that file today;
  - it reaches `npu/yolo.py`, which is already in;
  - so the expected addition is exactly `tools/w4v_bias_audit.py`. F.2 confirms it by running that computation
    (light, no inference) before the text is frozen.
- **Validation before any use (the gate's precondition).** Run on yolov8n bc_e1 and yolov8s bc_e1, it must reproduce
  the committed GENUINE logs (B.6):
  - every CONV line, line for line with the timings stripped;
  - the (b), (c), (d), SELFCHECK and HELD lines.
  - The gate checks it with its own parser. A diff against dbc4b7bb alone is not enough.

**Per method:**

| method | SEQ base (weights; starting biases) | BASE check |
|---|---|---|
| BC (tier 1) | the pinned `rtn_<form>` file; its shipped biases | the arm's non-bias initializers == the base's |
| AdaRound + BC, GPTQ + BC (tier 2) | the pinned tier-1 `ar_<form>` / `gptq_<form>` file (w4_recover.py:843-844); its shipped biases | the same |
| VC (tier 1) | the arm's own weights, the shipped biases as the start (§3.3 step 3: BC on the rescaled layer) | none: the weights are rescaled by design; C5 checks them, self-referentially |

On a VC arm, C5b checks only the BC step on the rescaled weights. It does not check the rescale (a_c) or the
re-quantization; C5 covers those, self-referentially.

**PASS:** `HELD … (a) False (ANCHORED agrees: True) (b) False (c) False (d) False -> GENUINE` for the pinned arm sha.
Anything else is a C5b FAIL, which is a DEFECT and falls under B.3 (STOP, v2.6).

**When.** C5b runs as its own stage, before the screen of the arms it covers:
- **`bias2`**, after emit2 and its pins, before screen2: all 12 +BC arms (6 per model). This is the gate's
  preference, and it is recommended.
  - ESTIMATE about 56 min: 6 × 370 s plus 6 × about 190 s.
    - 370 s is the yolov8s SEQ, MEASURED in rerun2, unthrottled.
    - About 190 s for yolov8n (ESTIMATE): about 2.9 s per Conv × 63, plus about 9 s. This is taken from the
      unthrottled CONV 0-14 of the 10:58Z rerun. That run throttled after CONV 14, so no single factor applies to its
      total.
    - VC arms may differ.
- **`bias1`** (decision D3; beyond the gate's ask), after the v2.5 prereg, before the screen1 re-run: the 5 tier-1
  int32-bias arms per model not yet audited on (a) (bc_e2, bc_u, vc_e1, vc_e2, vc_u).
  - ESTIMATE about 47 min: 5 × 370 s plus 5 × about 190 s. The VC arms may differ.
  - Why offered: the screen1 cut uses these arms, and their biases carry only C5's self-referential check.
  - A second reason: bias1 runs before the roughly 30-min screen1 re-run, so it is the first run over 20 min on the
    new fan, under HWiNFO logging. The fan is proven so far for 6.3 min (rerun2, 12:32:01-12:38:19Z). bias1 closes
    that gap before screen1.
  - A bias1 PASS is also their B.3 audit.
- Both stages exceed 15 min and are heavy runs: START REQUEST, keep-awake, HWiNFO and the RAM monitor.
- Their logs are `w4v_bias{1,2}_<model>_<machine>_<date>.log`, with `W4V_CONTROL C5b <model> <arm> PASS|FAIL` lines
  and a `W4V_DONE bias<tier> <model>` success line.
- `STAGES` gains `bias1` (if D3 is yes) and `bias2`.

---

## D. screen1 under v2.5: re-run, or complete the held run (decision D1)

- Section B's record defines how a zero-detection row is recorded. It applies to whichever bc_e1 row is scored; in
  option A, that is the re-run's row.
- The choice between A and B is the user's. The gate's position is A.

**Option A: re-run from row 1 (recommended).**
- **The held v2.4 files are archived, not deleted.** This is their first commit, so it is not a rewrite.
  - The 24 row logs and the load witness go to `results/int4/w4v_v24_screen1_stop/`, after D.2's detection pass and
    the witness scrub. The directory gets a `results/README.md` row.
  - The bc_e1 row log stays byte-identical there.
  - The 24 detections files move to `scratch/w4v/dets_v24_screen1_stop/`, git-ignored. Their sha256s go into the
    archive commit's body.
- **Why the move is needed:**
  - The dets path has no date (`$DETS/w4v_screen_<model>_<arm>_n500.json`, w4v-eval.sh:220), and the runner refuses
    an existing target (:94), so a re-run collides whatever its date.
  - `ready()`, `own_logs()` and the runner's `ls` globs are all non-recursive (w4v_verdict.py:714, :738;
    w4v-eval.sh:478, :563-566), so the subdirectory is invisible to them. No code has to learn about it.
- **In the re-run:**
  - all 36 rows and the cut run under the v2.5 triple, in one sitting: B8 and the 17 tier-1 arms per model;
  - yolov8s bc_e1 will again be a zero-detection row. Its record cites
    `w4v_audit/recompute_yolov8s_rerun2_desktop2_20260926.log` (GENUINE for `2700ece0…`).
- **Report-only**, in the repro pattern (`GATE_D_DETS_SHA256`, w4v-eval.sh:197-203): each re-run row prints
  `V24_DETS_SHA256 <archived sha> (byte-identical True|False)` against the archived v2.4 detections of the same row.
  Printed, not required.
- **Cost:**
  - MEASURED: the v2.4 sitting ran 24 rows in 872 s.
  - ESTIMATE: the full 36 rows and the cut take about 30 min, so the run needs the keep-awake and HWiNFO.
  - One triple per stage. No resume logic.

**Option B: complete the held run.**
- The 23 complete v2.4 rows and the bc_e1 row (with its record) are carried inside a v2.5 stage.
- The runner gains resume logic: `stage_screen1` skips the rows it carries.
- `log_problems` accepts v2.4 hash blocks for exactly those 24 names.
- There are two witnesses for one stage.
- Only the 12 remaining yolov8s rows and the cut run. ESTIMATE about 15 min.
- **Against it:**
  - a stage with mixed triples;
  - new resume code in the one stage that failed;
  - a saving of about 15 min.

**Recommendation: A.** It keeps one triple per stage, needs no resume path, and matches the gate's first ruling. It
also gives a free determinism check of 24 rows.

**D.2 The detection pass and the scrub (before any commit of screen1 files or audit logs; the gate's terms).**
- **Read-only, over the load witness and the 24 row logs, and over the audit logs of B.6.** It looks for:
  - the gate's tool-name list: (ii)'s six names plus five more, case-insensitive. The list is operational: it stays
    in the pass's script and out of v2.5's committed text, which may hold no tool names;
  - `Users[\\/]<name>`;
  - the machine's hostname;
  - the job-tmp and worktree paths.
- Every hit is reported under (iii) with its line numbers. No metric line is printed.
- **Then the witness fix, rule (ii), narrowed (A):**
  - in `HOST_LOAD_TOP` lines only, the image-name token before `:<pid>` becomes `<tool>` where it matches (ii)'s six
    names: 29 tokens;
  - every other hit goes to the gate first;
  - the commit body gives the pre- and post-fix sha256s and the counts;
  - no metric line is touched.
- If `w4v_witness_fix.py` changes, its new sha goes to the gate first.

---

## E. The carry-across rule (new §9.1)

**The rule, stated generally.** A committed v2.4 log is accepted by v2.5's `ready()` and `log_problems()` in place of
its v2.5 re-run iff all of these hold:
1. **Listed:** its name and HEAD LF sha256 are in `v2_5.carried_v24` (below), and the committed file matches.
2. **Its blocks:** every hash block in it equals `V24_FROZEN`, embedded in v2.5:
   - PREREG_TEXT_SHA256 `7ff9c2c46e9280e5002aafb6cb573646d0c920b6257b6aba4e2ff47a08caee49`;
   - PROTOCOL_JSON_SHA256 `46297af64e23a10df62c363f766bacc3f78faf121b5194ea8e6077b364710872`;
   - VERDICT_CODE_SHA256 `2fc815d93451a3be1f8403d768cddb5737218d9e58abe282cc25e77fb7eb7c64`;
   - the 30-file TOOLS_JSON the v2.4 prereg log prints.
3. **Tools unchanged:** every file in v2.4's TOOLS_JSON has the same LF sha in v2.5's. v2.5 may add files.
   - Its expected addition is exactly `tools/w4v_bias_audit.py`, as the import-graph code computes it (section C). A
     run in F.2 confirms it.
   - Any change or removal refuses every carried log, and those stages then re-run.
4. **Inputs unchanged:** v2.5's protocol JSON minus its one added key, `v2_5`, equals v2.4's protocol JSON exactly. The
   v2.4 JSON is embedded and checked against `46297af6…`. That covers the models and their shas, the arms, every
   hyperparameter, the calibration listing, H, the eval settings, ORT and the threads.
5. **As before:** committed and clean, its success lines, and no `W4V_CONTROL … FAIL`.

Any log that fails 1-5 is not carried: its stage re-runs, on the gate's ruling.

**Per stage:**

| stage | v2.4 log(s) (HEAD LF sha256) | v2.5 | why |
|---|---|---|---|
| selftest | `w4v_selftest_desktop2_20260925.log` `55f34e43e42fd736c36e2d23f6a679fd73130fe26bdc6140644697fa25f3c5dd` | RE-RUN; the v2.4 log is SUPERSEDED | it tests the verdict code, which v2.5 changes. §10 requires it to cover the new behaviour synthetically: the zerodet record and its refusals, carried-log acceptance and refusal (each of 1-5 broken once), the superseded exclusion, and C5b's PASS/FAIL parsing |
| controls | `w4v_controls_yolov8n_…20260925.log` `3230ae56d3275ece32933480ce0a8f80b8ecfaeac2f00bca7c45a7def5f6ec98`; `w4v_controls_yolov8s_…20260925.log` `851fa188434c0f9e0b2a09adb9386d205480bb0b973a6bccd2a563a0fcee7a54` | CARRY | `w4_recover` and its imports are unchanged (rule 3) |
| parity | `w4v_parity_c6a_yolov8n_…` `d5b83902878341423e84245def7daae8396e1bc6dee599dd776f640a737053b6`; `…c6a_yolov8s_…` `df44ee761f19f40b9c92d639ff1986d159d55eccf99f4b57df87b69a2c35234b`; `…c6b_yolov8n_…` `1b0b49ebd013004a649edfaadba60fa614d1c2c2dbe9042a3f192a618536fe39`; `…c6c_yolov8n_…` `f7a14d7c739443783ca5672f648677fc552aa7120b67c55470c21993af5861da`; report-only `…c6c_env17_yolov8n_…` `6bc22cbc430f740f4d49b33ca2dfaa7de5cfa1ddd826a1e12989ef16331a6d2e` | CARRY | `quant/` is unchanged |
| repro | `w4v_repro_<model>_<arm>_n5000_desktop2_20260925.log`: n b8 `cbdb9ae5b861b4de177f20edbeb459683bbe19e296edff8ba33a2c97c3f85a21`, n rtn_e1 `4e2eca4165cbf578f8cc19e430712a0e5f081d4a90beb5f9f25ac99cec2dfb7b`, n rtn_e2 `4f3b3d649fd49b766b40cea72cc9a82f17079732a1d90a00b37620edc8ddee42`, n rtn_u `0e70b22215c876f98c3364ee74cc252599676c1af4bc8c92ae750afb362d7a2a`, s b8 `74471da4518272671612dfa80beabc1fb075e77efcc464db8d0b733470c7383e`, s rtn_e1 `d0268253cf11f9c70344032a8539e24548b5fc435cd1639763f86279d2129493`, s rtn_e2 `bfd776faa6695ab7b2fe0b451bcd79b61e3079ee6fd0197cf0327105ed022363`, s rtn_u `50f7f0e0829a64773e6d418cbea2ce1de2dbd13474bff3d535969d243054ef3e`; `w4v_repro_c9_…` `e174c2c7096d02a1747ea8cd5469a3d899d2efcc2d5d6cb203fbe835804dc348`; witness `load_w4v_repro_…` `1a665b29cbd7c30a5994a9e36ecbe42c894363b7726e08cf2224999f86accba9` | CARRY; C9 re-derived | the eval is unchanged. The C9 PASS line was printed by v2.4's verdict code, so v2.5's `--verdict` re-derives C9 from the carried rows (the same comparison `--c9` made) and refuses on a mismatch |
| emit1 | `w4v_sensitivity_yolov8n_…20260926.log` `4a6e64be7adc4644d2cd2639f646da8cbf4fc8489c6e3deed89c4f1851fc45e8`; `w4v_emit1_yolov8n_…` `8a70e4bc1f6e1e037f0c11570923a5c08f88dba592eb0703c4db89da6f4c9d1f`; `w4v_sensitivity_yolov8s_…` `beff7fbb90c264ed46a84eaf91eb8bc463e325bfa0b42d0930d5bbec1eb478a1`; `w4v_emit1_yolov8s_…` `605379a310388e509c0606b9081090ff2405e04260b4516e444a8093d5443a5a` (the 28 sidecars go with them) | CARRY | `w4_recover` is unchanged; the arm files keep their pins |
| (census, not a stage) | `w4v_census_desktop2_20260925.log` `8185c5ee75278918a00fbc401fa3d36bcb38ccb57031fcec73d1ce50fa895197` | unchanged | report-only; only P1's scoring reads it |
| prereg | `w4v_prereg_desktop2_20260926.log` `35c758eceafd53d4c99c62fadc0fe541188f694dbcd27741f620fb49567acc6e` | RE-RUN; the v2.4 log is SUPERSEDED | the prereg log is what freezes the triple. The v2.5 run must pin exactly the v2.4 log's 42 files with equal shas, or refuse |
| screen1 | the held 24 row logs and the witness, never committed | ARCHIVED (option A), or carried (option B) | section D |

**What SUPERSEDED means** (`v2_5.superseded_v24`: name and LF sha, for exactly the selftest and prereg logs above):
- The file stays committed and untouched.
- Every glob that would otherwise count it skips exactly that name at exactly that sha:
  - `frozen_state()` (:548);
  - `pins()` (:567), which never reads a superseded prereg log's PIN lines;
  - `stage_logs()` for selftest and prereg (:659, :673), and so `ready()` (:738) and `own_logs()` (:714);
  - `cmd_prereg`'s "a prereg log exists already" check (:981).
- A superseded name whose sha differs is a refusal, so nothing can hide behind the list.

**Names.**
- v2.5 logs keep `<machine>_<date>` (TAG `desktop2_<date>`), because the runner's screen2 and verdict globs use
  `_${MACHINE}_*` (w4v-eval.sh:478, :563-566).
- A v2.5 log may never take a v2.4 log's name: the runner refuses an existing path (:94).
- The runner's date is local (`date +%Y%m%d`, :65), which is EDT on Desktop 2. So the v2.5 prereg can run from
  2026-09-27 00:00 EDT, which is 04:00Z, at the earliest. That is a stated cost.
- The v2.5 selftest may run on 2026-09-26, since the v2.4 selftest is dated 20260925.

**No merge of main before the verdict.**
- main's `8b54cb6` (`quant/cli.py`) and `37d5950` (`scripts/lib.sh`) are not in worktree-int4-vision, and both files
  are in the 30-file TOOLS_JSON.
- A merge would change them, and rule 3 would then refuse every carried log.

---

## F. Order after approval (every run on the gate's GO; commits local, via commit.sh)

1. **Evidence, no run.**
   1. The detection pass (D.2) over the audit logs, then commit `results/int4/w4v_audit/` (B.6) and its README row.
      - The script there is the first .py under results/. `tools/license_header_audit.py` (report-only, never
        `--apply`) lists only tracked files that name AMD or Xilinx, and the script names neither: 0 matches for
        amd|xilinx|apache today.
      - The report-only run confirms it once the file is tracked. pytest does not collect it.
   2. The detection pass and the witness fix over the held screen1 files, then commit
      `results/int4/w4v_v24_screen1_stop/` and its README row. Move the dets (option A).
      - These logs carry v2.4 metric lines. The gate reviews this commit by sha, line counts and the hygiene grep
        only; no diff of those logs is sent.
2. **The code.** The v2.5 text, `protocol()["v2_5"]`, `--zerodet`, the carried and superseded rules, the zero-row
   parse, the `eval_row` branch, the `V24_DETS_SHA256` line, `bias1`/`bias2`, `tools/w4v_bias_audit.py` and the new
   selftest cases.
   - Each diff and sha goes to the gate before its commit.
   - The frozen tools stay untouched.
   - Before the text is frozen, run `import_graph()` over TOOL_ROOTS plus the new root (light, no inference), to
     confirm TOOLS_JSON's addition set is exactly `tools/w4v_bias_audit.py`.
   - C5b's validation (section C) runs on yolov8n bc_e1 and yolov8s bc_e1 and reproduces the committed GENUINE logs.
     The gate checks it, before any bias1 or bias2 use. ESTIMATE about 9 min, light.
3. **Re-freeze.**
   - The selftest stage (light); commit.
   - The prereg stage (light; not before 2026-09-27 00:00 EDT = 04:00Z, by the naming rule; the same 42 pins);
     commit.
   - v2.5's triple is now frozen.
4. **bias1** (only if D3 is yes): START REQUEST; heavy, ~47 min ESTIMATE, keep-awake and HWiNFO; commit.
5. **screen1** (option A): START REQUEST, ~30 min ESTIMATE, keep-awake and HWiNFO; commit the rows, the zerodet
   record, the witness and the cut.
   - The report carries the cut list and counts per §10 (O7), no mAP.
6. **The rest.**
   - emit2 (START REQUEST), then bias2 (START REQUEST, ~56 min ESTIMATE), then pins, then screen2, then the gate, then
     the decisive run, then the verdict.
   - After the verdict: the README clause that `quant/cli.py:232` was fixed on main in `8b54cb6`.

---

## G. Decisions for the gate, then the user

| # | decision | recommended | the gate | why |
|---|---|---|---|---|
| D1 | screen1: re-run from row 1 (A) or complete the held run (B) | **A** | A | one triple per stage, no resume code; ~15 min dearer (ESTIMATE) |
| D2 | a zero-detection row of an arm with no int32 bias (RTN, AR, GPTQ, MX) | **STOP**, as v2.4 | STOP | no pre-registered audit covers those arms; 0.00 by rule would record an unchecked result |
| D3 | bias1: C5b on the 10 tier-1 int32-bias arms not yet audited on (a) | **yes** | yes | the screen1 cut uses them, and their biases carry only a self-referential check. It is also the first run over 20 min on the new fan (proven for 6.3 min), under HWiNFO, before the ~30 min screen1. ~47 min ESTIMATE. It is beyond the gate's ask |
| D4 | where the audit script is committed | `results/int4/w4v_audit/` beside its logs | accept | evidence of what ran, byte-identical; C5b's tool is the one under `tools/` |
| D5 | the report-only `V24_DETS_SHA256` line in the re-run | **yes** | yes | a free determinism check of 24 rows; prints byte identity only |
"""

PREREG = PREREG_V24 + "\n" + PREREG_V25



if __name__ == "__main__":
    sys.exit(main())
