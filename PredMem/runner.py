"""Run the benchmark suite and export verified results."""
import argparse
import fcntl
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from PredMem.clustering import check_foldseek
from PredMem.config import ROOT, digest, load, path, resolved
from PredMem.io import fasta, save, sha
from PredMem.reporting import write_benchmark_summary


MODELS = ("dplm2", "dplm21", "esm3")
MAIN_TASKS = ("joint", "motif")
SINGLE_TASKS = ("structure_prediction", "inverse_folding")
MAIN_ARMS = ("vanilla", "pm", "cfg", "cfg_pm")
SINGLE_ARMS = ("default", "vanilla", "pm")
DATASETS = {"cameo2022": 183, "pdb_date": 449}


def read(file):
    return json.loads(file.read_text())


def recipes(main_only):
    for model in MODELS:
        for task in MAIN_TASKS:
            for arm in MAIN_ARMS:
                yield model, task, "", arm
        if main_only:
            continue
        for task in SINGLE_TASKS:
            for dataset in DATASETS:
                for arm in SINGLE_ARMS:
                    if (model, task, dataset, arm) != ("esm3", "structure_prediction", "cameo2022", "pm"):
                        yield model, task, dataset, arm


def plan(out, main_only, single_run):
    jobs = []
    for model, task, dataset, arm in recipes(main_only):
        group = Path(model, task, *([dataset] if dataset else []), arm)
        recipe = Path("configs/models") / group.with_suffix(".yaml")
        config = resolved(ROOT / recipe)
        config["output"] = str(out / "runs" / group)
        config["evaluation"]["after_generation"] = False
        if task == "joint":
            expected = len(config["sampling"]["lengths"]) * config["sampling"]["per_length"]
        elif task == "motif":
            variants = config["dataset"]["motif_variants"] or load(
                path(config["dataset"]["motif_spans"]))["variant_reference"]
            expected = len(variants) * config["sampling"]["per_motif"]
        else:
            expected = DATASETS[dataset]
        sampling = config["sampling"]
        seeds = sampling["seeds"][:1] if single_run else sampling["seeds"]
        for seed in seeds:
            seed_config = {**config, "sampling": {**sampling, "seeds": [seed]}}
            jobs.append({"id": str(group / f"seed-{seed}"), "model": model, "task": task,
                         "dataset": dataset, "arm": arm, "seed": seed, "recipe": str(recipe),
                         "expected": expected, "config": seed_config})
    return jobs


def comparable_plan(jobs):
    # A move of the recipe directory does not change the resolved job.
    return [{**job, "recipe": Path(job["recipe"]).parts[2:]} for job in jobs]


def executable(name):
    target = shutil.which(name)
    if target is None:
        raise ValueError(f"Executable unavailable: {name}")
    # Keep a virtual environment's interpreter symlink intact.
    return os.path.abspath(target)


def preflight(jobs):
    interpreters = {"dplm2": executable(os.environ["PM_DPLM_PYTHON"])}
    interpreters["dplm21"] = interpreters["dplm2"]
    if any(job["model"] == "esm3" for job in jobs):
        interpreters["esm3"] = executable(os.environ["PM_ESM3_PYTHON"])
    checked = set()
    for job in jobs:
        if job["recipe"] in checked:
            continue
        checked.add(job["recipe"])
        config = job["config"]
        data = config["dataset"]
        evaluation = config["evaluation"]
        executable(evaluation["python"])
        if job["task"] in MAIN_TASKS:
            check_foldseek(executable(evaluation["foldseek"]), evaluation)
        if job["task"] == "motif":
            required = [path(data["motif_dir"])]
            if job["model"] != "esm3":
                required += [path(data["motif_aa"]), path(data["motif_structure"])]
            for entry in required:
                if not entry.exists():
                    raise ValueError(f"Motif input unavailable: {entry}")
        elif job["dataset"]:
            targets = fasta(path(data["fasta"]))
            if len(targets) != job["expected"] or not path(data["reference_dir"]).is_dir():
                raise ValueError(f"Benchmark inputs differ from {job['recipe']}")
    return interpreters


def seed_dir(job):
    return Path(job["config"]["output"]) / f"seed-{job['seed']}"


def verify_generation(job):
    directory = seed_dir(job)
    receipt = read(directory / "GENERATION-COMPLETE.json")
    if (not receipt.get("passed") or receipt["seed"] != job["seed"]
            or receipt["config_sha256"] != digest(job["config"])):
        raise ValueError(f"Generation receipt does not match {job['id']}")
    if len(read(directory / "samples.json")) != job["expected"]:
        raise ValueError(f"Generation sample count does not match {job['id']}")


def completed(job):
    verify_generation(job)
    directory = seed_dir(job)
    evaluation = read(directory / "EVALUATION-COMPLETE.json")
    if (not evaluation.get("passed") or evaluation["outputs"] != job["expected"]
            or evaluation["metrics_sha256"] != sha(directory / "metrics.csv")
            or digest(read(directory / "evaluation-resolved-config.json")) != digest(job["config"])):
        raise ValueError(f"Evaluation receipt does not match {job['id']}")
    return {k: job[k] for k in ("model", "task", "dataset", "arm", "seed")} | {
        "metrics": evaluation["metrics"]}


def execute(out, job, gpu, python):
    directory = seed_dir(job)
    if (directory / "EVALUATION-COMPLETE.json").exists():
        return completed(job)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    record = out / "attempts" / job["id"] / stamp
    record.mkdir(parents=True)
    config_file = record / "resolved-config.json"
    save(config_file, job["config"])
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    log = record / "run.log"
    print(f"[GPU {gpu}] {job['id']}  {log}", flush=True)
    with log.open("w") as stream:
        if not (directory / "GENERATION-COMPLETE.json").exists():
            if directory.exists():
                directory.rename(record / "partial-generation")
            subprocess.run([python, "-u", "-m", "PredMem", "run", "--config", str(config_file)],
                           cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        verify_generation(job)
        for name in ("clusters", "refold.log"):
            partial = directory / name
            if partial.exists():
                partial.rename(record / name)
        subprocess.run([job["config"]["evaluation"]["python"], "-u", "-m", "PredMem",
                        "evaluate", "--config", str(config_file)], cwd=ROOT, env=env,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    return completed(job)


def run_jobs(out, jobs, devices, interpreters):
    pending = queue.Queue()
    for index, job in enumerate(jobs):
        pending.put((index, job))
    stop = threading.Event()

    def worker(gpu):
        results, failures = [], []
        while not stop.is_set():
            try:
                index, job = pending.get_nowait()
            except queue.Empty:
                break
            try:
                results.append((index, execute(out, job, gpu, interpreters[job["model"]])))
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
                failures.append((job["id"], str(exc)))
                print(f"Failed {job['id']}: {exc}", file=sys.stderr, flush=True)
        return results, failures

    results, failures = [], []
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        workers = [pool.submit(worker, device) for device in devices]
        try:
            for future in workers:
                done, errors = future.result()
                results.extend(done)
                failures.extend(errors)
        except KeyboardInterrupt:
            stop.set()
            raise
    if failures:
        raise ValueError(f"{len(failures)} seed jobs failed. See {out / 'attempts'} and rerun with the same options.")
    return [record for _, record in sorted(results)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", help="Comma-separated CUDA devices. Defaults to CUDA_VISIBLE_DEVICES or 0.")
    parser.add_argument("--output", type=Path, help="Directory for model outputs and benchmark summaries.")
    parser.add_argument("--main-only", action="store_true", help="Run only the joint and motif benchmarks.")
    parser.add_argument("--single-run", "--single-seed", action="store_true", help=argparse.SUPPRESS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="List recipes and run counts without running jobs.")
    mode.add_argument("--report-only", action="store_true", help="Export metrics from an existing completed run.")
    args = parser.parse_args()

    suffix = ("-main" if args.main_only else "") + ("-single-seed" if args.single_run else "")
    out = (args.output or ROOT / "outputs" / ("paper" + suffix)).expanduser().resolve()
    try:
        if args.dry_run:
            jobs = plan(out, args.main_only, args.single_run)
            counts = {}
            for job in jobs:
                key = (job["recipe"], job["expected"])
                counts[key] = counts.get(key, 0) + 1
            for (recipe, expected), repeats in counts.items():
                unit = "run" if repeats == 1 else "runs"
                print(f"{recipe}  {repeats} {unit}, {expected} samples per run")
            print(f"{len(counts)} recipes, {len(jobs)} runs, output {out}")
            return

        out.mkdir(parents=True, exist_ok=True)
        with (out / ".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError(f"Another runner is using {out}") from exc
            manifest = out / "RUN-PLAN.json"
            if args.report_only:
                records = [completed(job) for job in read(manifest)]
            else:
                devices = (args.gpus if args.gpus is not None else os.environ.get(
                    "CUDA_VISIBLE_DEVICES", "0")).split(",")
                devices = [device.strip() for device in devices]
                if not all(devices) or len(set(devices)) != len(devices):
                    raise ValueError("Specify one or more distinct CUDA devices with --gpus.")
                saved_jobs = read(manifest) if manifest.exists() else None
                jobs = plan(out, args.main_only, args.single_run)
                if saved_jobs is not None and comparable_plan(saved_jobs) != comparable_plan(jobs):
                    raise ValueError(f"{manifest} contains a different plan. Choose another --output directory.")
                interpreters = preflight(jobs)
                if not manifest.exists():
                    if (out / "runs").exists():
                        raise ValueError(f"{out / 'runs'} exists without a run plan. Choose another --output directory.")
                    save(manifest, jobs)
                records = run_jobs(out, jobs, devices, interpreters)
            summary = write_benchmark_summary(out, records)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"PredMem runner: {exc}\n")
    print(f"Benchmark summary: {summary}", flush=True)


if __name__ == "__main__":
    main()
