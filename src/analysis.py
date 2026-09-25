import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from . import config, utils

GA_COLOUR = "#1f4e79"
RS_COLOUR = "#c1666b"


def load_searches():
    runs = {"ga": [], "random": []}
    for path in sorted(config.LOG_DIR.glob("search_*.json")):
        summary = utils.load_json(path)
        if summary["budget"] == config.EVALUATION_BUDGET:
            runs[summary["method"]].append(summary)
    return runs


# RQ1 figure
def figure_search_comparison(runs):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, colour, label in [("ga", GA_COLOUR, "Genetic algorithm"),
                                  ("random", RS_COLOUR, "Random search")]:
        curves = np.array([r["best_so_far_curve"] for r in runs[method]])
        x = np.arange(1, curves.shape[1] + 1)

        # Show the individual runs too
        for curve in curves:
            ax.plot(x, curve, color=colour, alpha=0.25, linewidth=0.9)
        ax.plot(x, curves.mean(axis=0), color=colour, linewidth=2.2,
                label=f"{label} (n={len(curves)})")
        ax.fill_between(x, curves.min(axis=0), curves.max(axis=0), color=colour, alpha=0.12)

    ax.set_xlabel("Models trained")
    ax.set_ylabel("Best validation macro F1 so far")
    ax.set_title("Search progress under an identical budget")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "search_comparison.png", bbox_inches="tight")
    plt.close(fig)


# How the GA population moves and which the best-so far
def figure_ga_generations(runs):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run in runs["ga"]:
        log = run["generation_log"]
        gens = [g["generation"] for g in log]
        ax.plot(gens, [g["best_fitness"] for g in log], color=GA_COLOUR,
                linewidth=1.8, alpha=0.8)
        ax.plot(gens, [g["mean_fitness"] for g in log], color=GA_COLOUR,
                linewidth=1.2, alpha=0.5, linestyle="--")

    ax.plot([], [], color=GA_COLOUR, linewidth=1.8, label="Best in population")
    ax.plot([], [], color=GA_COLOUR, linewidth=1.2, linestyle="--", label="Population mean")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Validation macro F1")
    ax.set_title(f"Best and mean population fitness across {len(runs['ga'])} GA runs")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "ga_generations.png", bbox_inches="tight")
    plt.close(fig)


# RQ2 figure
def figure_proxy_fidelity():
    summary = utils.load_json(config.LOG_DIR / "proxy_fidelity.json")
    results = summary["results"]
    proxy = np.array([r["proxy_macro_f1"] for r in results])
    full = np.array([r["full_val_macro_f1"] for r in results])
    params = np.array([r["params"] for r in results]) / 1e6

    fig, ax = plt.subplots(figsize=(6.2, 5))
    scatter = ax.scatter(proxy, full, c=params, cmap="viridis", s=90,
                         edgecolor="white", linewidth=1.2, zorder=3)
    fig.colorbar(scatter, ax=ax, label="Parameters (millions)")


    fit = np.polyfit(proxy, full, 1)
    xs = np.linspace(proxy.min(), proxy.max(), 50)
    ax.plot(xs, np.polyval(fit, xs), color="#888", linewidth=1.2, linestyle="--", zorder=2)

    ax.set_xlabel(f"Proxy macro F1 ({summary['proxy_epochs']} epochs)")
    ax.set_ylabel("Fully-trained macro F1")
    ax.set_title(f"Proxy fidelity: Spearman $\\rho$ = {summary['spearman_rho']:.3f} "
                 f"(n = {summary['n']})")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "proxy_fidelity.png", bbox_inches="tight")
    plt.close(fig)


# full score vs a linear fit on the proxy
def figure_proxy_bias():
    results = utils.load_json(config.LOG_DIR / "proxy_fidelity.json")["results"]
    proxy = np.array([r["proxy_macro_f1"] for r in results])
    full = np.array([r["full_val_macro_f1"] for r in results])
    residual = full - np.polyval(np.polyfit(proxy, full, 1), proxy)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, key, label in [(axes[0], "dropout", "Dropout rate"),
                           (axes[1], "params", "Parameters (millions)")]:
        values = np.array([r[key] for r in results], dtype=float)
        if key == "params":
            values /= 1e6
        fit = np.polyfit(values, residual, 1)
        xs = np.linspace(values.min(), values.max(), 40)
        corr = np.corrcoef(values, residual)[0, 1]

        ax.axhline(0, color="#999", linewidth=1, linestyle="--")
        ax.scatter(values, residual, s=70, color=GA_COLOUR, edgecolor="white", zorder=3)
        ax.plot(xs, np.polyval(fit, xs), color=RS_COLOUR, linewidth=1.6)
        ax.set_title(f"{label}   (r = {corr:+.2f})", fontsize=10)
        ax.set_xlabel(label)
    axes[0].set_ylabel("Residual: full score minus proxy prediction")
    fig.suptitle("Proxy residuals", y=1.01)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "proxy_bias.png", bbox_inches="tight")
    plt.close(fig)


# E5 figure
def figure_surrogate():
    summary = utils.load_json(config.LOG_DIR / "surrogate.json")
    sim = summary["simulation"]
    ga = np.array(sim["ga_best_values"])
    rs = np.array(sim["rs_best_values"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})

    bins = np.linspace(min(ga.min(), rs.min()), max(ga.max(), rs.max()), 34)
    axes[0].hist(rs, bins=bins, color=RS_COLOUR, alpha=0.55, label="Random search")
    axes[0].hist(ga, bins=bins, color=GA_COLOUR, alpha=0.55, label="Genetic algorithm")
    axes[0].axvline(rs.mean(), color=RS_COLOUR, linewidth=2)
    axes[0].axvline(ga.mean(), color=GA_COLOUR, linewidth=2)
    axes[0].set_xlabel("Best surrogate score reached")
    axes[0].set_ylabel(f"Replications (n = {sim['replications']})")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].axvline(0, color="#999", linewidth=1, linestyle="--")
    axes[1].hist(ga - rs, bins=30, color="#6b7a8f")
    axes[1].set_xlabel("GA minus random search, seed-matched replications")
    axes[1].set_title(f"GA ahead in {sim['ga_win_rate']:.1%} of replications", fontsize=10)

    fig.suptitle(f"Surrogate simulation (5-fold CV $\\rho$ = "
                 f"{summary['validation']['cv_spearman_mean']:.2f})", y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "surrogate_comparison.png", bbox_inches="tight")
    plt.close(fig)


# Per class test F1 of final models
def figure_per_class_f1():
    summaries = [utils.load_json(p) for p in sorted(config.LOG_DIR.glob("summary_*.json"))]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    width = 0.8 / len(summaries)
    x = np.arange(config.NUM_CLASSES)
    for i, summary in enumerate(summaries):
        scores = np.array([[run["per_class_f1"][e] for e in config.EMOTIONS]
                           for run in summary["runs"]])
        ax.bar(x + i * width, scores.mean(axis=0), width,
               yerr=scores.std(axis=0), capsize=2, label=summary["tag"])

    ax.set_xticks(x + width * (len(summaries) - 1) / 2)
    ax.set_xticklabels(config.EMOTIONS, rotation=20, ha="right")
    ax.set_ylabel("Test F1")
    ax.set_title("Per-class F1 (mean over seeds)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "per_class_f1.png", bbox_inches="tight")
    plt.close(fig)

def write_tables(runs):
    ga = np.array([r["best_fitness"] for r in runs["ga"]])
    rs = np.array([r["best_fitness"] for r in runs["random"]])
    gap = ga.mean() - rs.mean()
    pooled = np.sqrt((ga.var(ddof=1) + rs.var(ddof=1)) / 2)

    lines = ["# Results tables", "",
             "## Search comparison (RQ1)", "",
             "| Method | Run | Best macro F1 | Models trained | Fitness calls | Cache hits | Minutes |",
             "|---|---|---|---|---|---|---|"]
    for method, label in [("ga", "GA"), ("random", "Random")]:
        for run in runs[method]:
            lines.append(f"| {label} | {run['run_seed']} | {run['best_fitness']:.4f} | "
                         f"{run['evaluations_spent']} | {run['fitness_calls']} | "
                         f"{run['cache_hits']} | {run['seconds'] / 60:.1f} |")
    lines += ["",
              f"- GA: macro F1 {ga.mean():.4f} ± {ga.std(ddof=1):.4f} over {len(ga)} runs",
              f"- Random search: macro F1 {rs.mean():.4f} ± {rs.std(ddof=1):.4f} over {len(rs)} runs",
              f"- Difference in means: {gap:+.4f}, Cohen's d: {gap / pooled:.2f}",
              ""]

    fidelity = utils.load_json(config.LOG_DIR / "proxy_fidelity.json")
    lines += ["## Proxy fidelity (RQ2)", "",
              f"Spearman rho = {fidelity['spearman_rho']:.3f} (p = {fidelity['spearman_p']:.4f}, "
              f"n = {fidelity['n']})", "",
              "| Proxy macro F1 | Full macro F1 | Params | Configuration |", "|---|---|---|---|"]
    for r in sorted(fidelity["results"], key=lambda r: r["proxy_macro_f1"], reverse=True):
        lines.append(f"| {r['proxy_macro_f1']:.4f} | {r['full_val_macro_f1']:.4f} | "
                     f"{r['params'] / 1e6:.2f}M | `{r['genome_key']}` |")
    lines.append("")

    surrogate = utils.load_json(config.LOG_DIR / "surrogate.json")
    v, sim = surrogate["validation"], surrogate["simulation"]
    lines += ["## Surrogate (E5)", "",
              f"Random forest on {v['n_training_points']} evaluations, CV Spearman "
              f"{v['cv_spearman_mean']:.3f} ± {v['cv_spearman_std']:.3f}, "
              f"MAE {v['cv_mean_absolute_error']:.4f}", "",
              "| Gene | Importance |", "|---|---|"]
    for name, importance in sorted(v["feature_importance"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {name} | {importance:.3f} |")
    lines += ["",
              f"{sim['replications']} replications per method, budget {sim['budget']}:", "",
              f"- GA: {sim['ga_mean']:.4f} ± {sim['ga_std']:.4f}",
              f"- Random search: {sim['rs_mean']:.4f} ± {sim['rs_std']:.4f}",
              f"- Difference {sim['mean_difference']:+.4f}, Cohen's d {sim['cohens_d']:.2f}, "
              f"GA ahead in {sim['ga_win_rate']:.1%}, Mann-Whitney p = {sim['mannwhitney_p']:.2e}",
              ""]

    for path in sorted(config.LOG_DIR.glob("summary_*.json")):
        summary = utils.load_json(path)
        lines += [f"## {summary['tag']} (final training)", "",
                  f"Test macro F1 {summary['test_macro_f1_mean']:.4f} "
                  f"± {summary['test_macro_f1_std']:.4f}, accuracy "
                  f"{summary['test_accuracy_mean']:.4f} ± {summary['test_accuracy_std']:.4f}, "
                  f"seeds {summary['seeds']}", "",
                  "| Class | F1 (mean ± std) |", "|---|---|"]
        for emotion in config.EMOTIONS:
            values = np.array([run["per_class_f1"][emotion] for run in summary["runs"]])
            lines.append(f"| {emotion} | {values.mean():.3f} ± {values.std(ddof=1):.3f} |")
        lines.append("")

    (config.RESULTS_DIR / "tables.md").write_text("\n".join(lines), encoding="utf-8")


def run():
    matplotlib.use("Agg")
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })

    runs = load_searches()
    figure_search_comparison(runs)
    figure_ga_generations(runs)
    figure_proxy_fidelity()
    figure_proxy_bias()
    figure_surrogate()
    figure_per_class_f1()
    write_tables(runs)
    print(f"Figures saved to {config.FIGURE_DIR}, tables to {config.RESULTS_DIR / 'tables.md'}")
