# External night comparison figures

Figures show every fixed model on the four frozen protocols. The helper requires all 52 results and verifies their file hashes before plotting.

- Panel (a): B minus A separately for seeds 42, 43, and 44; black ticks show their mean.
- Panel (b): A/B/C minus the original Day baseline, mean ± sample SD over three training seeds. Faint dots show the individual seeds. These are not confidence intervals.
- Panel (c): G2 minus G1 within A and B, seed 42 only. Labels show net correct-query differences.

Positive differences mean higher Top-1 recall. One correct query corresponds to 100/N percentage points; query counts differ substantially across protocols. Results do not establish statistical significance or independence across geographically correlated queries.

H1 uses same-main-batch Source Night and the Qwen bank. H3 uses an independent Source stream and a separate Source-only bank; cross-axis differences do not isolate only pool size.

SVOX/Tokyo/RobotCar protocols have been inspected historically; these are not newly blind tests. RobotCar uses public train poses as evaluation only, not the official benchmark. Tokyo local v3 query images have not been independently byte-verified against v2.

SF-XL remains untouched; Aachen/official RobotCar tests lack local public retrieval ground truth

Input summary SHA256: `b5fbc23d140d42f77a970557d8b7b28302d93c8f96e4d10a4ed19646af8fb3ee`. Exact plotted values and figure hashes are in `plot_data.json`.
