"""
test_sanity.py
--------------
Lightweight integration smoke test for the full GA pipeline.

This is not a unit test suite — it is a quick end-to-end check that
all components wire together correctly before committing to a long run.

What it checks
--------------
- Triangle: construction, mutation, serialisation
- FitnessFunction: RMSE and SSIM evaluate without crashing
- utils: render() produces the correct array shape and dtype
- Individual: random creation, lazy fitness evaluation, copy_with
- Population: random init, evaluate, stats, replace
- Operators: one forward pass of selection -> crossover -> mutation
- GeneticAlgorithm: 3-generation mini-run completes without errors

Run with:
    python test_sanity.py

Expected output: a series of PASS lines and no exceptions.
All checks run in a few seconds (no full GA run, tiny population).
"""

import sys
import traceback
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"

def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    suffix = f" -- {detail}" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return condition

def section(title: str) -> None:
    print(f"\n--- {title} ---")

all_passed = True

def run_check(label: str, condition: bool, detail: str = "") -> None:
    global all_passed
    if not check(label, condition, detail):
        all_passed = False


# ---------------------------------------------------------------------------
# Synthetic target image (avoids needing the actual painting on disk)
# ---------------------------------------------------------------------------

IMG_W, IMG_H = 300, 400
rng = np.random.default_rng(42)

# Synthetic target: random noise image, dtype uint8, shape (400, 300, 3)
target_array = rng.integers(0, 256, size=(IMG_H, IMG_W, 3), dtype=np.uint8)
target_pil = Image.fromarray(target_array, mode="RGB")

# ---------------------------------------------------------------------------
# 1. Triangle
# ---------------------------------------------------------------------------
section("Triangle")

from triangle import Triangle

try:
    tri = Triangle.random(IMG_W, IMG_H, rng)
    run_check("Triangle.random() constructs without error", True)
    run_check("vertices has 3 elements", len(tri.vertices) == 3)
    run_check("color has 4 channels", len(tri.color) == 4)
    run_check("color channels in [0,255]", all(0 <= c <= 255 for c in tri.color))

    mutated_v = tri.mutate_vertices(IMG_W, IMG_H, rng, sigma=10.0)
    run_check("mutate_vertices returns new Triangle", mutated_v is not tri)
    run_check("mutated vertices still 3", len(mutated_v.vertices) == 3)

    mutated_c = tri.mutate_color(rng, sigma=10.0)
    run_check("mutate_color returns new Triangle", mutated_c is not tri)

    d = tri.to_dict()
    restored = Triangle.from_dict(d)
    run_check("to_dict/from_dict roundtrip", restored.vertices == tri.vertices and restored.color == tri.color)

    run_check("area() >= 0", tri.area() >= 0)
    run_check("bounding_box() returns 4 ints", len(tri.bounding_box()) == 4)

except Exception:
    traceback.print_exc()
    run_check("Triangle section completed without exception", False)

# ---------------------------------------------------------------------------
# 2. Fitness functions
# ---------------------------------------------------------------------------
section("Fitness functions")

from fitness import build_fitness, RMSEFitness, SSIMFitness, CIEDEFitness

try:
    rmse_fn   = build_fitness("rmse",      target_pil)
    ssim_fn   = build_fitness("ssim",      target_pil)
    ciede_fn  = build_fitness("ciede2000", target_pil)

    run_check("build_fitness('rmse') constructs",      isinstance(rmse_fn, RMSEFitness))
    run_check("build_fitness('ssim') constructs",      isinstance(ssim_fn, SSIMFitness))
    run_check("build_fitness('ciede2000') constructs", isinstance(ciede_fn, CIEDEFitness))

    candidate = rng.integers(0, 256, size=(IMG_H, IMG_W, 3), dtype=np.uint8)

    rmse_score  = rmse_fn.evaluate(candidate)
    ssim_score  = ssim_fn.evaluate(candidate)
    ciede_score = ciede_fn.evaluate(candidate)

    run_check("RMSE score is float >= 0",        isinstance(rmse_score, float) and rmse_score >= 0)
    run_check("SSIM score is float in [0, 2]",   isinstance(ssim_score, float) and 0 <= ssim_score <= 2)
    run_check("CIEDE2000 score is float >= 0",   isinstance(ciede_score, float) and ciede_score >= 0)

    # Perfect match -> RMSE == 0, SSIM fitness == 0
    perfect_rmse  = rmse_fn.evaluate(target_array)
    perfect_ssim  = ssim_fn.evaluate(target_array)
    run_check("RMSE of target against itself == 0", perfect_rmse == 0.0)
    run_check("SSIM fitness of target against itself == 0", abs(perfect_ssim) < 1e-5)

except Exception:
    traceback.print_exc()
    run_check("Fitness section completed without exception", False)

# ---------------------------------------------------------------------------
# 3. Rendering (utils)
# ---------------------------------------------------------------------------
section("Rendering (utils)")

from utils import render, IMG_WIDTH, IMG_HEIGHT

try:
    triangles = [Triangle.random(IMG_WIDTH, IMG_HEIGHT, rng) for _ in range(100)]
    arr = render(triangles)

    run_check("render() returns ndarray",           isinstance(arr, np.ndarray))
    run_check("render() shape is (H, W, 3)",        arr.shape == (IMG_HEIGHT, IMG_WIDTH, 3))
    run_check("render() dtype is uint8",            arr.dtype == np.uint8)
    run_check("render() values in [0, 255]",        arr.min() >= 0 and arr.max() <= 255)

except Exception:
    traceback.print_exc()
    run_check("Rendering section completed without exception", False)

# ---------------------------------------------------------------------------
# 4. Individual
# ---------------------------------------------------------------------------
section("Individual")

from individual import Individual, NUM_TRIANGLES

try:
    ind = Individual.random(rmse_fn, rng)

    run_check("Individual.random() creates unevaluated individual",   not ind.is_evaluated())
    run_check("Individual has correct number of triangles",           len(ind.triangles) == NUM_TRIANGLES)

    fitness_val = ind.fitness   # triggers evaluation
    run_check("fitness property triggers evaluation",                  ind.is_evaluated())
    run_check("fitness is a non-negative float",                       isinstance(fitness_val, float) and fitness_val >= 0)

    fitness_val_2 = ind.fitness
    run_check("fitness is cached (same value on second access)",       fitness_val == fitness_val_2)

    copy_ind = ind.copy_with()
    run_check("copy_with() returns new object",                        copy_ind is not ind)
    run_check("copy_with() has same triangles",                        copy_ind.triangles == ind.triangles)
    run_check("copy_with() fitness cache is reset",                    not copy_ind.is_evaluated())

    new_tri = Triangle.random(IMG_WIDTH, IMG_HEIGHT, rng)
    replaced = ind.with_triangle(0, new_tri)
    run_check("with_triangle() returns new Individual",                replaced is not ind)
    run_check("with_triangle() replaces correct index",                replaced.triangles[0] is new_tri)

    d = ind.to_dict()
    restored_ind = Individual.from_dict(d, rmse_fn)
    run_check("to_dict/from_dict restores triangles",                  len(restored_ind.triangles) == NUM_TRIANGLES)
    run_check("to_dict/from_dict restores fitness cache",              restored_ind.is_evaluated())

    run_check("Individual __lt__ uses fitness",                        (ind < ind) == False)

except Exception:
    traceback.print_exc()
    run_check("Individual section completed without exception", False)

# ---------------------------------------------------------------------------
# 5. Population
# ---------------------------------------------------------------------------
section("Population")

from population import Population

POP_SIZE = 6  # tiny population for speed

try:
    pop = Population.random(POP_SIZE, rmse_fn, rng)

    run_check("Population.random() creates correct size",  len(pop) == POP_SIZE)
    run_check("Population repr shows unevaluated",         "evaluated=0" in repr(pop))

    pop.evaluate(n_workers=1)

    run_check("All individuals evaluated after evaluate()", all(ind.is_evaluated() for ind in pop))
    run_check("best is Individual",                         isinstance(pop.best, Individual))
    run_check("best fitness <= mean fitness",               pop.best_fitness() <= pop.mean_fitness())

    stats = pop.stats()
    run_check("stats() has required keys",
              all(k in stats for k in ("best", "mean", "std", "worst")))
    run_check("stats best <= worst",                        stats["best"] <= stats["worst"])

    diversity = pop.diversity()
    run_check("diversity() in [0, 1]",                     0.0 <= diversity <= 1.0)

    # Test replace with elitism
    offspring = [Individual.random(rmse_fn, rng) for _ in range(POP_SIZE - 1)]
    next_pop = pop.replace(offspring, n_elites=1)
    run_check("replace() returns Population of same size",  len(next_pop) == POP_SIZE)
    run_check("elite survives into next generation",
              pop.best in next_pop.individuals)

except Exception:
    traceback.print_exc()
    run_check("Population section completed without exception", False)

# ---------------------------------------------------------------------------
# 6. Operators
# ---------------------------------------------------------------------------
section("Operators")

from ga_operators.selection import TournamentSelection, RankSelection, RouletteSelection
from ga_operators.crossover import SinglePointCrossover, UniformCrossover, KPointCrossover
from ga_operators.mutation import GaussianMutation, ResetMutation, SwapMutation, CompositeMutation

try:
    pop.evaluate()  # ensure evaluated

    # Selection
    for cls, kwargs in [
        (TournamentSelection, {"tournament_size": 3}),
        (RankSelection,       {"selection_pressure": 1.5}),
        (RouletteSelection,   {}),
    ]:
        sel = cls(**kwargs)
        selected = sel.select(pop.individuals, n_parents=4, rng=rng)
        run_check(
            f"{cls.__name__} returns correct number of parents",
            len(selected) == 4,
        )
        run_check(
            f"{cls.__name__} returns Individual objects",
            all(isinstance(s, Individual) for s in selected),
        )

    # Crossover
    pa, pb = pop.individuals[0], pop.individuals[1]
    for cls, kwargs in [
        (SinglePointCrossover, {}),
        (UniformCrossover,     {"swap_prob": 0.5}),
        (KPointCrossover,      {"k": 2}),
    ]:
        cx = cls(**kwargs)
        c1, c2 = cx.cross(pa, pb, rng)
        run_check(f"{cls.__name__} produces two offspring",  c1 is not None and c2 is not None)
        run_check(f"{cls.__name__} offspring are unevaluated", not c1.is_evaluated())
        run_check(
            f"{cls.__name__} offspring have correct triangle count",
            len(c1.triangles) == NUM_TRIANGLES and len(c2.triangles) == NUM_TRIANGLES,
        )

    # Mutation
    ind_to_mutate = pop.individuals[0]
    for cls, kwargs in [
        (GaussianMutation, {"mutation_rate": 0.5}),
        (ResetMutation,    {"mutation_rate": 0.5}),
        (SwapMutation,     {"mutation_rate": 1.0, "n_swaps": 3}),
    ]:
        mut = cls(**kwargs)
        mutated = mut.mutate(ind_to_mutate, rng)
        run_check(f"{cls.__name__} returns new Individual",       mutated is not ind_to_mutate)
        run_check(f"{cls.__name__} result is unevaluated",        not mutated.is_evaluated())
        run_check(
            f"{cls.__name__} result has correct triangle count",
            len(mutated.triangles) == NUM_TRIANGLES,
        )

    composite = CompositeMutation([
        GaussianMutation(mutation_rate=0.3),
        ResetMutation(mutation_rate=0.1),
        SwapMutation(mutation_rate=0.5),
    ])
    cm_result = composite.mutate(ind_to_mutate, rng)
    run_check("CompositeMutation returns new Individual",       cm_result is not ind_to_mutate)
    run_check("CompositeMutation result is unevaluated",        not cm_result.is_evaluated())

except Exception:
    traceback.print_exc()
    run_check("Operators section completed without exception", False)

# ---------------------------------------------------------------------------
# 7. GA engine -- mini run (3 generations, population=4)
# ---------------------------------------------------------------------------
section("GA engine (mini run: 3 generations, population=4)")

from ga import GeneticAlgorithm, GAConfig, EarlyStopping
from runner import RunConfig

try:
    config = GAConfig(
        population_size=4,
        n_generations=3,
        crossover_rate=0.8,
        n_elites=1,
        n_workers=1,
        early_stopping=EarlyStopping(patience=0),  # disabled
        checkpoint_interval=0,                      # disabled
        seed=42,
    )

    ga = GeneticAlgorithm(
        fitness_fn=rmse_fn,
        selection=TournamentSelection(tournament_size=2),
        crossover=SinglePointCrossover(),
        mutation=GaussianMutation(mutation_rate=0.1),
        config=config,
    )

    callback_calls = []
    def on_generation(gen, best, stats):
        callback_calls.append(gen)

    best = ga.run(callback=on_generation)

    run_check("run() returns an Individual",               isinstance(best, Individual))
    run_check("best individual is evaluated",              best.is_evaluated())
    run_check("best fitness is non-negative float",        isinstance(best.fitness, float) and best.fitness >= 0)
    run_check("generation log has 4 entries (0-3)",        len(ga.generation_log) == 4)
    run_check("callback fired for each generation",        len(callback_calls) == 4)
    run_check("log entries have required keys",
              all("best" in e and "mean" in e for e in ga.generation_log))

    # Verify global best matches min logged best within rounding tolerance.
    # Log entries are rounded to 6 decimal places so we allow 1e-5 slack.
    logged_bests = [e["best"] for e in ga.generation_log]
    run_check(
        "global best_individual fitness matches min of all logged bests",
        abs(best.fitness - min(logged_bests)) < 1e-5,
    )

except Exception:
    traceback.print_exc()
    run_check("GA engine section completed without exception", False)

# ---------------------------------------------------------------------------
# 8. RunConfig fields
# ---------------------------------------------------------------------------
section("RunConfig fields")

try:
    def test_run_config_has_init_strategy_field():
        """RunConfig must expose init_strategy with default 'random'."""
        cfg = RunConfig(
            name="test",
            phase=0,
            description="test",
            selection="tournament_k3",
            crossover="uniform",
            mutation="gaussian_fixed",
            n_elites=1,
        )
        assert cfg.init_strategy == "random"

    def test_run_config_has_image_ratio_field():
        """RunConfig must expose image_ratio with default 0.5."""
        cfg = RunConfig(
            name="test",
            phase=0,
            description="test",
            selection="tournament_k3",
            crossover="uniform",
            mutation="gaussian_fixed",
            n_elites=1,
        )
        assert cfg.image_ratio == 0.5

    test_run_config_has_init_strategy_field()
    run_check("RunConfig.init_strategy exists with default 'random'", True)

    test_run_config_has_image_ratio_field()
    run_check("RunConfig.image_ratio exists with default 0.5", True)

except Exception:
    traceback.print_exc()
    run_check("RunConfig fields section completed without exception", False)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
if all_passed:
    print("All checks passed.")
else:
    print("Some checks FAILED -- review output above.")
    sys.exit(1)