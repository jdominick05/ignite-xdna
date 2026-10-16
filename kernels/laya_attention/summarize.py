#!/usr/bin/env python3
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline summary of an attention suite; refuses to qualify incomplete timings."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def summarize(path):
    text=path.read_text(encoding="utf-8")
    arms={}
    arm=None
    suite=None
    process=None
    paired=[]
    paired_checks=[]
    for line in text.splitlines():
        if line.startswith("SUITE_BEGIN "):
            arm=line.split()[1]
            arms[arm]=dict(checks=[],timing=None,resident=None)
        elif line.startswith("SUITE_RESULT "):
            suite=json.loads(line.split(" ",1)[1])
        elif line.startswith("RESEARCH_PROCESS_RESULT "):
            process=json.loads(line.split(" ",1)[1])
        elif line.startswith("HOST_PAIR_RESULT "):
            paired.append(json.loads(line.split(" ",1)[1]))
        elif line.startswith("PAIR_CHECK "):
            paired_checks.append(json.loads(line.split(" ",1)[1]))
        elif arm and line.startswith("{"):
            try: row=json.loads(line)
            except json.JSONDecodeError: continue
            if "seed" in row: arms[arm]["checks"].append(row)
            if "cpu_ort_8threads_ms" in row: arms[arm]["timing"]=row
            if "slope_ms" in row: arms[arm]["resident"]=row
    eligible=bool(suite and suite["verdict"]=="PASS" and process and process["timing_eligible"])
    print(json.dumps(dict(source=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          suite=suite,process=process,qualified=eligible)))
    if eligible:
        for row in paired:
            samples=row['samples']; n=len(row['paired_deltas_ms'])
            pack_deltas=[mean(x['packing_ms'] for x in samples['base'][2*i:2*i+2])
                         -mean(x['packing_ms'] for x in samples['host'][2*i:2*i+2]) for i in range(n)]
            import math
            from statistics import stdev
            pm=mean(pack_deltas); margin=1.96*stdev(pack_deltas)/math.sqrt(n)
            checks=[r for r in paired_checks if r['mode']==row['mode']]
            print(json.dumps(dict(paired_host=row['mode'],output_checks=len(checks),
                                  checks_pass=bool(checks) and all(r['verdict']=='PASS' for r in checks),
                                  baseline_end_to_end_ms=row['base_mean_ms'],host_end_to_end_ms=row['host_mean_ms'],
                                  end_to_end_delta_ms=row['paired_delta_mean_ms'],
                                  end_to_end_delta_ci95=row['paired_delta_ci95_normal_approx'],
                                  baseline_pack_ms=row['base_pack_mean_ms'],host_pack_ms=row['host_pack_mean_ms'],
                                  packing_delta_ms=pm,packing_delta_ci95_normal_approx=[pm-margin,pm+margin],
                                  cpu_mean_ms=mean(row['cpu_ms']))))
    for name,data in arms.items():
        checks=data["checks"]
        result=dict(arm=name,output_checks=len(checks),
                    checks_pass=all(r["verdict"]=="PASS" for r in checks) if checks else None,
                    max_abs=max((r["max_abs"] for r in checks),default=None),
                    max_relative_l2=max((r["relative_l2"] for r in checks),default=None))
        if eligible and data["timing"]:
            row=data["timing"]
            result.update(dispatch_mean_ms=mean(row["dispatch_ms"]),
                          dispatch_min_ms=min(row["dispatch_ms"]),dispatch_max_ms=max(row["dispatch_ms"]),
                          cpu_mean_ms=mean(row["cpu_ort_8threads_ms"]),
                          packing_mean_ms=mean(r["packing_ms"] for r in checks),
                          write_mean_ms=mean(r["write_sync_ms"] for r in checks),
                          read_mean_ms=mean(r["read_sync_ms"] for r in checks))
            if row.get("mapped_end_to_end_ms"):
                result.update(end_to_end_mean_ms=mean(row["mapped_end_to_end_ms"]),
                              end_to_end_min_ms=min(row["mapped_end_to_end_ms"]),
                              end_to_end_max_ms=max(row["mapped_end_to_end_ms"]))
        if eligible and data["resident"]: result["resident"]=data["resident"]
        print(json.dumps(result))
    if eligible:
        for mode in ("global","local"):
            baseline=arms.get(f"base_{mode}_h16_v512",{}).get("timing")
            if not baseline: continue
            bm=mean(baseline["dispatch_ms"]); be=mean(baseline["mapped_end_to_end_ms"])
            for variant in ("mask","bank","unroll","host","combined"):
                candidate=arms.get(f"{variant}_{mode}_h16_v512",{}).get("timing")
                if not candidate: continue
                dm=mean(candidate["dispatch_ms"]); em=mean(candidate["mapped_end_to_end_ms"])
                cpu=mean(candidate["cpu_ort_8threads_ms"])
                print(json.dumps(dict(variant_comparison=variant,mode=mode,baseline_dispatch_ms=bm,
                                      candidate_dispatch_ms=dm,dispatch_reduction_percent=100*(1-dm/bm),
                                      baseline_end_to_end_ms=be,candidate_end_to_end_ms=em,
                                      end_to_end_reduction_percent=100*(1-em/be),cpu_ms=cpu,
                                      end_to_end_over_cpu=em/cpu)))
        for mode in ("global","local"):
            vector=arms.get(f"{mode}_h16_v512",{}).get("timing")
            scalar=arms.get(f"scalar_{mode}_h16_v512",{}).get("timing")
            if vector and scalar:
                vm=mean(vector["dispatch_ms"]); sm=mean(scalar["dispatch_ms"])
                cm=mean(vector["cpu_ort_8threads_ms"])
                print(json.dumps(dict(comparison=mode,heads=16,vector_ms=vm,scalar_ms=sm,
                                      scalar_over_vector=sm/vm,cpu_ms=cm,vector_over_cpu=vm/cm)))
            control=arms.get(f"vector_{mode}_h16_v512",{}).get("timing")
            if vector and control and vector.get("mapped_end_to_end_ms"):
                vm=mean(vector["dispatch_ms"]); cm=mean(vector["cpu_ort_8threads_ms"])
                end=mean(vector["mapped_end_to_end_ms"])
                print(json.dumps(dict(comparison=mode,heads=16,shared_dispatch_ms=vm,
                                      mapped_end_to_end_ms=end,cpu_ms=cm,
                                      vector_control_ms=mean(control["dispatch_ms"]),
                                      dispatch_over_cpu=vm/cm,end_to_end_over_cpu=end/cm)))
    return 0 if eligible else 1


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log",type=Path)
    raise SystemExit(summarize(ap.parse_args().log))
