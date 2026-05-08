# CIFO_GroupConvergence
CIFO Project 2025/2026


project/
├── src/
│   ├── triangle.py          # Triangle representation
│   ├── individual.py        # Individual (chromosome)
│   ├── population.py        # Population management
│   ├── fitness.py           # Fitness functions
│   ├── operators/
│   │   ├── selection.py     # Selection strategies
│   │   ├── crossover.py     # Crossover strategies
│   │   └── mutation.py      # Mutation strategies
│   ├── ga.py                # GA engine / runner
│   └── utils.py             # Image rendering, helpers
├── notebooks/
│   ├── experiments.ipynb    # Run GA, tune params
│   └── results.ipynb        # Plots, comparisons, visuals
├── data/
│   └── girl_pearl.png
└── outputs/                 # Saved results, images



Triangle          <- done
fitness.py        <- done
    |
utils.py          <- next (rendering logic)
    |
Individual        <- depends on Triangle + utils.py
    |
Operators         <- depend on Individual
    |
Population        <- depends on Individual + Operators
    |
GA engine         <- depends on everything