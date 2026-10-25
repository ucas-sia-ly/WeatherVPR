# P0 descriptive query strata

Fixed brightness edges: 0, 0.1, 0.25, 0.5, 1. Spatial groups: floor(UTM/200 m).

Groups describe correlated development queries; they do not establish independent routes or significance.

| Comparison | TopK | Brightness bin | Queries | Repaired | Regressed | Net hits |
|---|---|---|---:|---:|---:|---:|
| seed42_B_vs_A | R@1 | [0,0.1) | 0 | 0 | 0 | +0 |
| seed42_B_vs_A | R@1 | [0.1,0.25) | 226 | 1 | 0 | +1 |
| seed42_B_vs_A | R@1 | [0.25,0.5) | 476 | 6 | 2 | +4 |
| seed42_B_vs_A | R@1 | [0.5,1] | 0 | 0 | 0 | +0 |
| seed42_B_vs_A | R@5 | [0,0.1) | 0 | 0 | 0 | +0 |
| seed42_B_vs_A | R@5 | [0.1,0.25) | 226 | 0 | 0 | +0 |
| seed42_B_vs_A | R@5 | [0.25,0.5) | 476 | 1 | 2 | -1 |
| seed42_B_vs_A | R@5 | [0.5,1] | 0 | 0 | 0 | +0 |
| seed42_B_vs_A | R@10 | [0,0.1) | 0 | 0 | 0 | +0 |
| seed42_B_vs_A | R@10 | [0.1,0.25) | 226 | 1 | 0 | +1 |
| seed42_B_vs_A | R@10 | [0.25,0.5) | 476 | 1 | 3 | -2 |
| seed42_B_vs_A | R@10 | [0.5,1] | 0 | 0 | 0 | +0 |
| seed42_B_vs_C | R@1 | [0,0.1) | 0 | 0 | 0 | +0 |
| seed42_B_vs_C | R@1 | [0.1,0.25) | 226 | 1 | 0 | +1 |
| seed42_B_vs_C | R@1 | [0.25,0.5) | 476 | 6 | 1 | +5 |
| seed42_B_vs_C | R@1 | [0.5,1] | 0 | 0 | 0 | +0 |
| seed42_B_vs_C | R@5 | [0,0.1) | 0 | 0 | 0 | +0 |
| seed42_B_vs_C | R@5 | [0.1,0.25) | 226 | 0 | 0 | +0 |
| seed42_B_vs_C | R@5 | [0.25,0.5) | 476 | 1 | 2 | -1 |
| seed42_B_vs_C | R@5 | [0.5,1] | 0 | 0 | 0 | +0 |
| seed42_B_vs_C | R@10 | [0,0.1) | 0 | 0 | 0 | +0 |
| seed42_B_vs_C | R@10 | [0.1,0.25) | 226 | 1 | 0 | +1 |
| seed42_B_vs_C | R@10 | [0.25,0.5) | 476 | 1 | 2 | -1 |
| seed42_B_vs_C | R@10 | [0.5,1] | 0 | 0 | 0 | +0 |
| seed43_B_vs_A | R@1 | [0,0.1) | 0 | 0 | 0 | +0 |
| seed43_B_vs_A | R@1 | [0.1,0.25) | 226 | 1 | 1 | +0 |
| seed43_B_vs_A | R@1 | [0.25,0.5) | 476 | 7 | 0 | +7 |
| seed43_B_vs_A | R@1 | [0.5,1] | 0 | 0 | 0 | +0 |
| seed43_B_vs_A | R@5 | [0,0.1) | 0 | 0 | 0 | +0 |
| seed43_B_vs_A | R@5 | [0.1,0.25) | 226 | 0 | 1 | -1 |
| seed43_B_vs_A | R@5 | [0.25,0.5) | 476 | 1 | 2 | -1 |
| seed43_B_vs_A | R@5 | [0.5,1] | 0 | 0 | 0 | +0 |
| seed43_B_vs_A | R@10 | [0,0.1) | 0 | 0 | 0 | +0 |
| seed43_B_vs_A | R@10 | [0.1,0.25) | 226 | 1 | 0 | +1 |
| seed43_B_vs_A | R@10 | [0.25,0.5) | 476 | 1 | 2 | -1 |
| seed43_B_vs_A | R@10 | [0.5,1] | 0 | 0 | 0 | +0 |
| seed43_B_vs_C | R@1 | [0,0.1) | 0 | 0 | 0 | +0 |
| seed43_B_vs_C | R@1 | [0.1,0.25) | 226 | 1 | 1 | +0 |
| seed43_B_vs_C | R@1 | [0.25,0.5) | 476 | 8 | 0 | +8 |
| seed43_B_vs_C | R@1 | [0.5,1] | 0 | 0 | 0 | +0 |
| seed43_B_vs_C | R@5 | [0,0.1) | 0 | 0 | 0 | +0 |
| seed43_B_vs_C | R@5 | [0.1,0.25) | 226 | 0 | 1 | -1 |
| seed43_B_vs_C | R@5 | [0.25,0.5) | 476 | 2 | 2 | +0 |
| seed43_B_vs_C | R@5 | [0.5,1] | 0 | 0 | 0 | +0 |
| seed43_B_vs_C | R@10 | [0,0.1) | 0 | 0 | 0 | +0 |
| seed43_B_vs_C | R@10 | [0.1,0.25) | 226 | 0 | 0 | +0 |
| seed43_B_vs_C | R@10 | [0.25,0.5) | 476 | 1 | 2 | -1 |
| seed43_B_vs_C | R@10 | [0.5,1] | 0 | 0 | 0 | +0 |
| seed44_B_vs_A | R@1 | [0,0.1) | 0 | 0 | 0 | +0 |
| seed44_B_vs_A | R@1 | [0.1,0.25) | 226 | 1 | 1 | +0 |
| seed44_B_vs_A | R@1 | [0.25,0.5) | 476 | 4 | 2 | +2 |
| seed44_B_vs_A | R@1 | [0.5,1] | 0 | 0 | 0 | +0 |
| seed44_B_vs_A | R@5 | [0,0.1) | 0 | 0 | 0 | +0 |
| seed44_B_vs_A | R@5 | [0.1,0.25) | 226 | 0 | 0 | +0 |
| seed44_B_vs_A | R@5 | [0.25,0.5) | 476 | 0 | 2 | -2 |
| seed44_B_vs_A | R@5 | [0.5,1] | 0 | 0 | 0 | +0 |
| seed44_B_vs_A | R@10 | [0,0.1) | 0 | 0 | 0 | +0 |
| seed44_B_vs_A | R@10 | [0.1,0.25) | 226 | 1 | 0 | +1 |
| seed44_B_vs_A | R@10 | [0.25,0.5) | 476 | 0 | 3 | -3 |
| seed44_B_vs_A | R@10 | [0.5,1] | 0 | 0 | 0 | +0 |
| seed44_B_vs_C | R@1 | [0,0.1) | 0 | 0 | 0 | +0 |
| seed44_B_vs_C | R@1 | [0.1,0.25) | 226 | 1 | 1 | +0 |
| seed44_B_vs_C | R@1 | [0.25,0.5) | 476 | 4 | 1 | +3 |
| seed44_B_vs_C | R@1 | [0.5,1] | 0 | 0 | 0 | +0 |
| seed44_B_vs_C | R@5 | [0,0.1) | 0 | 0 | 0 | +0 |
| seed44_B_vs_C | R@5 | [0.1,0.25) | 226 | 0 | 1 | -1 |
| seed44_B_vs_C | R@5 | [0.25,0.5) | 476 | 0 | 2 | -2 |
| seed44_B_vs_C | R@5 | [0.5,1] | 0 | 0 | 0 | +0 |
| seed44_B_vs_C | R@10 | [0,0.1) | 0 | 0 | 0 | +0 |
| seed44_B_vs_C | R@10 | [0.1,0.25) | 226 | 1 | 0 | +1 |
| seed44_B_vs_C | R@10 | [0.25,0.5) | 476 | 1 | 2 | -1 |
| seed44_B_vs_C | R@10 | [0.5,1] | 0 | 0 | 0 | +0 |

Full per-query changes in each 200 m spatial cell and brightness bin are in the adjacent JSON artifact.

Brightness can correlate with content, camera and place.
UTM grid cells are descriptive groups; adjacent cells and correlated frames are not independent routes.
No bootstrap confidence intervals, p-values, model selection or new retrieval evaluation.
