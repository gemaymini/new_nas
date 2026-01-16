# Future Tasks

- [ ] **Consolidate Plotting Logic**: Merge `src/apply/plot_algorithm_comparison.py` into `src/utils/plotting.py`. It currently shares ConvexHull and scatter logic with `plot_pareto_comparison` but adds inset zooming.
- [ ] **Unit Tests**: Add unit tests for `src/utils/checkpoint.py` to verify safe loading and error handling.
- [ ] **Dependency Management**: Ensure all plotting scripts rely on `src/utils/plotting.py` to maintain consistent styling.
