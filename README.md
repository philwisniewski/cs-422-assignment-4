# CS 422 Assignment 4 — Maximum Concurrent Flow and Network Design

Part 1: best topology for the uniform traffic matrix (bound + construction + LP check).
Part 2: Gurobi MILP that picks the best topology for any hose traffic matrix (n = 8, d = 4).
The write-up is `report/report.pdf` (source: `report/report.md`).

## Setup

```bash
python3 -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`pip install gurobipy` ships a size-limited license (2000 variables / constraints).
That is enough for Part 2 (505 variables). For bigger Part 1 instances the LP falls
back to SciPy/HiGHS automatically. To use your free academic license, request it at
gurobi.com (Academia) and put the `gurobi.lic` file in your home directory (or point
the `GRB_LICENSE_FILE` environment variable at it) — no code changes needed.

## Run

```bash
# Part 1: sweep over (n, d), prints bound vs. achieved theta, saves results/part1_results.json
python src/part1_uniform.py
# Part 1: one instance + a figure in results/
python src/part1_uniform.py --n 13 --d 4

# Part 2: experiment suite (uniform, permutation, random, clustered hose TMs)
python src/part2_milp.py --show-topologies
python src/part2_milp.py --simple            # at most one link per ordered pair
python src/part2_milp.py --tm examples/example_tm.csv   # your own 8x8 TM (CSV)
python src/part2_milp.py --robust 3          # one static topology for 3 TMs (~2 min)
python src/part2_milp.py --verbose           # show the Gurobi log
```

Expected highlights: Part 1 `(8,4)` and `(13,4)` reach the bound (0.700, 0.600);
Part 2 uniform TM gives θ = 0.700 (matches Part 1), permutation TM θ = 1.0.

Rebuild the PDF (needs pandoc + xelatex):
```bash
cd report && pandoc report.md -o report.pdf --pdf-engine=xelatex --resource-path=.
```

## Files

| file | what |
|---|---|
| `src/common.py` | Moore bound, BFS distances, max-concurrent-flow LP (Gurobi or SciPy) |
| `src/part1_uniform.py` | Part 1 constructions (complete / cyclotomic Cayley / Kautz + search), selection, figures |
| `src/part2_milp.py` | **Part 2 MILP** (`best_topology_milp`), robust variant, experiment driver |
| `src/hose_tms.py` | hose traffic-matrix generators |
| `results/` | JSON outputs and topology figures |


