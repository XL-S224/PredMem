"""Export benchmark summaries as Markdown, LaTeX, and CSV."""
import csv
import math
import statistics
from pathlib import Path


MODELS = {"dplm2": "DPLM-2", "dplm21": "DPLM-2.1", "esm3": "ESM3"}
ARMS = {"default": "Default", "vanilla": "Vanilla", "pm": "Memory",
        "cfg": "CFG", "cfg_pm": "CFG+Memory"}
BASELINES = {"pm": "vanilla", "cfg_pm": "cfg"}
SUCCESS = ("success_percent", "Success (%)", 2)
PLDDT = ("plddt", "pLDDT", 2)
SCTM = ("tm_score", "scTM", 3)
SCRMSD = ("rmsd", "scRMSD (Å)", 2)
CLUSTERS = ("successful_clusters", "Success clusters", 1)
SOLVED = ("solved", "Solved / 24", 1)
METRICS = {
    "joint": (SUCCESS, PLDDT, SCTM, SCRMSD, CLUSTERS),
    "motif": (SUCCESS, SCTM, SCRMSD, SOLVED, CLUSTERS),
    "structure_prediction": (("rmsd", "RMSD (Å)", 2), ("tm_score", "TM-score", 3)),
    "inverse_folding": (SCTM, ("aar", "AAR (%)", 2)),
}
SINGLE_COLUMNS = (
    ("structure_prediction", "cameo2022", "CAMEO2022 SP"),
    ("structure_prediction", "pdb_date", "PDB Date Split SP"),
    ("inverse_folding", "cameo2022", "CAMEO2022 IF"),
    ("inverse_folding", "pdb_date", "PDB Date Split IF"),
)
RECIPE_NOTES = {
    ("esm3", "motif"): "ESM3 motif uses the number of unfixed residues as the step count.",
}


def aggregate(records):
    groups = {}
    for record in records:
        key = tuple(record[k] for k in ("model", "task", "dataset", "arm"))
        group = groups.setdefault(key, {})
        seed = record["seed"]
        if seed in group:
            raise ValueError(f"Duplicate seed for {key}: {seed}")
        group[seed] = record["metrics"]
    summaries = {}
    for key, by_seed in groups.items():
        metrics = {}
        for name, _, _ in METRICS[key[1]]:
            values = [
                result["diversity"]["successful"]["clusters"] if name == "successful_clusters" else result[name]
                for result in by_seed.values()
            ]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite {name} for {key}")
            metrics[name] = (statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None)
        summaries[key] = (tuple(sorted(by_seed)), metrics)
    return summaries


def markdown(title, headers, rows):
    return "\n".join([f"## {title}", "", "| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(row) + " |" for row in rows), ""])


def latex(title, headers, rows):
    def escape(value):
        return value.replace("%", r"\%").replace("Å", r"\AA{}").replace("±", r"\(\pm\)").replace("—", "---")

    def line(row):
        return " & ".join(map(escape, row)) + r" \\"

    body = [r"\begin{tabular}{" + "ll" + "r" * (len(headers) - 2) + "}",
            r"\hline", line(headers), r"\hline", *map(line, rows),
            r"\hline", r"\end{tabular}"]
    if len(headers) > 8:
        body = [r"\resizebox{\linewidth}{!}{%", *body, "}"]
    return "\n".join([r"\begin{table*}[t]", r"\centering", r"\small",
                      r"\caption{" + escape(title) + "}", *body, r"\end{table*}", ""])


def main_table(summaries, task):
    rows = []
    for model in MODELS:
        for arm in ("vanilla", "pm", "cfg", "cfg_pm"):
            seeds, metrics = summaries[model, task, "", arm]
            baseline = BASELINES.get(arm)
            if baseline:
                base_seeds, base_metrics = summaries[model, task, "", baseline]
                if seeds != base_seeds:
                    raise ValueError(f"Seed cohorts differ for {model}/{task}/{arm}")
            method = ARMS[arm]
            if task == "motif" and (model == "esm3" or (model == "dplm21" and baseline)):
                method += " (archived)"
            cells = [MODELS[model], method]
            for key, _, precision in METRICS[task]:
                mean = metrics[key][0]
                value = f"{mean:.{precision}f}"
                if baseline:
                    delta = mean - base_metrics[key][0]
                    if abs(delta) < 0.5 * 10 ** -precision:
                        delta = 0
                    value += f" ({delta:+.{precision}f})"
                cells.append(value)
            rows.append(cells)
    headers = ["Model", "Method", *(label for _, label, _ in METRICS[task])]
    return headers, rows


def single_table(summaries):
    headers = ["Model", "Method", *(f"{prefix} {label}" for task, _, prefix in SINGLE_COLUMNS
                                   for _, label, _ in METRICS[task])]
    rows = []
    for model in MODELS:
        for arm in ("default", "vanilla", "pm"):
            cells = [MODELS[model], ARMS[arm]]
            for task, dataset, _ in SINGLE_COLUMNS:
                result = summaries.get((model, task, dataset, arm))
                if result is None:
                    if (model, task, dataset, arm) != ("esm3", "structure_prediction", "cameo2022", "pm"):
                        raise ValueError(f"Missing benchmark result: {model}/{task}/{dataset}/{arm}")
                    cells.extend(("—", "—"))
                    continue
                _, metrics = result
                for key, _, precision in METRICS[task]:
                    mean, sd = metrics[key]
                    value = f"{mean:.{precision}f}"
                    if sd is not None:
                        value += f" ± {sd:.{precision}f}"
                    cells.append(value)
            rows.append(cells)
    return headers, rows


def write_benchmark_summary(out: Path, records) -> Path:
    summaries = aggregate(records)
    tables = [
        ("Unconditional co-generation", *main_table(summaries, "joint")),
        ("Motif scaffolding", *main_table(summaries, "motif")),
    ]
    if any(key[1] in ("structure_prediction", "inverse_folding") for key in summaries):
        tables.append(("Structure prediction and inverse folding", *single_table(summaries)))
    note = ("Values average the selected seeds equally. Parentheses give the change from Vanilla for Memory "
            "and from CFG for CFG+Memory. Success changes are in percentage points. "
            "The CSV includes sample standard deviations across seeds and the seed IDs.")
    notes = ["# Benchmark summary", "", f"{len(records)} completed seed jobs.", "", note, "",
             *(f"- {text}" for text in RECIPE_NOTES.values()), ""]
    summary = out / "benchmark_summary.md"
    summary.write_text("\n".join(notes + [markdown(*table) for table in tables]))
    comments = "\n".join(f"% {line}" for line in (note, *RECIPE_NOTES.values()))
    (out / "benchmark_summary.tex").write_text(
        "% The wide single-modality table uses graphicx.\n" + comments + "\n\n"
        + "\n".join(latex(*table) for table in tables))
    with (out / "benchmark_metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("model", "task", "dataset", "method", "metric", "mean", "seed_sd", "n_seeds",
                         "seeds", "delta_vs_baseline", "protocol_note"))
        for (model, task, dataset, arm), (seeds, metrics) in summaries.items():
            baseline = summaries.get((model, task, dataset, BASELINES.get(arm))) if task in ("joint", "motif") else None
            protocol = RECIPE_NOTES.get((model, task), "") if model == "esm3" or arm in BASELINES else ""
            for key, (mean, sd) in metrics.items():
                delta = mean - baseline[1][key][0] if baseline else None
                writer.writerow((model, task, dataset, ARMS[arm], key, mean, sd, len(seeds),
                                 " ".join(map(str, seeds)), delta, protocol))
    return summary
