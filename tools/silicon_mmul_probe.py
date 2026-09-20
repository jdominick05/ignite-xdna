"""Compile and disassemble every mixed int16/int8 shape in the installed AIE2 API."""
import hashlib
import platform
import re
import subprocess
from pathlib import Path

from aie.utils import config


def main():
    root = Path(__file__).resolve().parents[1]
    build = root / "build/silicon_mmul_probe"
    build.mkdir(parents=True, exist_ok=True)
    inc = Path(config.cxx_header_path())
    header = inc / "aie_api/detail/aie2/mmul_16_8.hpp"
    body = header.read_text()
    shapes = [tuple(map(int, s)) for s in re.findall(r"struct mmul_16_8<(\d+), (\d+), (\d+), TypeA, TypeB, 32>", body)]
    print("machine:", platform.node(), flush=True)
    print("API_HEADER_SHA256:", hashlib.sha256(header.read_bytes()).hexdigest(), flush=True)
    compiler = config.peano_cxx_path()
    print(subprocess.check_output([compiler, "--version"], text=True), flush=True)
    failures = []
    # The two K=16 specializations take sparse weights; passing a vector by
    # reference exercises code generation without inventing a sparse encoding.
    for m, k, n in shapes + [(4, 8, 8)]:
        sparse = k == 16
        label = f"{m}x{k}x{n}_{'sparse' if sparse else 'dense'}"
        src = build / (label + ".cc")
        obj = src.with_suffix(".o")
        src.write_text(f'''#include <aie_api/aie.hpp>
extern "C" void probe(const aie::vector<int16,{m*k}>& a,
 const aie::{'sparse_vector' if sparse else 'vector'}<int8,{k*n}>& b, int32* out) {{
 aie::mmul<{m},{k},{n},int16,int8,acc32> c;
 c.mul(a,b);
 c.mac(a,b);
 aie::store_v(out,c.to_vector<int32>());
}}
''', encoding="utf-8")
        cmd = [compiler, str(src), "-c", "-o", str(obj), "-I" + str(inc), "-std=c++20",
               "-O2", "-Wno-attributes", "-Wno-macro-redefined", "-D__AIE_API_AIE_ADF_HPP__",
               "--target=aie2-none-unknown-elf"]
        print("COMMAND:", subprocess.list2cmdline(cmd), flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)
        expected = (m, k, n) in shapes
        print(f"COMPILE {label} exit={result.returncode} expected={'accept' if expected else 'reject'}", flush=True)
        if bool(result.returncode == 0) != expected:
            failures.append(label)
        if result.returncode:
            print(result.stderr, flush=True)
        else:
            dis = subprocess.check_output([str(Path(compiler).with_name("llvm-objdump.exe")), "-d", str(obj)], text=True)
            print("OBJECT_SHA256:", hashlib.sha256(obj.read_bytes()).hexdigest(), flush=True)
            print(dis, flush=True)
            if "vmac" not in dis and "vmul" not in dis:
                failures.append(label + "_missing_vector_multiply")
    print("VERDICT:", "PASS" if not failures else "FAIL " + repr(failures), flush=True)
    print("Scope: measured compiler acceptance and emitted AIE2 instructions, not silicon throughput or accuracy.")
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
