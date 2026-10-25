# Fixed model artifact audit

13 model artifacts audited; all checks pass. Actual weights hashed: 4.19 GiB. No model execution or image scans.

All models use the original Day checkpoint, native224 preprocessing, full late4 backbone Students (blocks8–11), 1816 sequential microbatches, 908 optimizer updates, one full epoch, main7263 / Source1816 views, and the same fixed Qwen Day bank. A has no auxiliary; B adds1816 real views; C adds1816 synthetic views. No initial adapter or partial step limit is recorded; token style and query fusion are inactive.

| Model | Axis | Seed | Auxiliary | Source regime | Updates | Weights SHA256 |
|---|---|---:|---|---|---:|---|
| seed42_A | H1 | 42 | none | same main batch / Qwen bank | 908 | `2d9de08d9c79984edcde9723d9d867b115f3e6e27334d14227769f567883a262` |
| seed42_B | H1 | 42 | real | same main batch / Qwen bank | 908 | `a9a804c85470b93a7ef3de79a9a74c7b6b7706cc9700bac645511ec8ee2673f9` |
| seed42_C | H1 | 42 | synthetic | same main batch / Qwen bank | 908 | `cfd0fc2b7f6b3594c942f0bafc7a7786f023c7329e937680ec1ce087dc1296a5` |
| seed43_A | H1 | 43 | none | same main batch / Qwen bank | 908 | `46fd4b3648e3dd389874896406be88ef3a501b348fa5c8ea4b75ef7430058870` |
| seed43_B | H1 | 43 | real | same main batch / Qwen bank | 908 | `2e4711fea5812357d4e33ad523bb104a0852c802e9147fe462ab1fcaea6ea7b1` |
| seed43_C | H1 | 43 | synthetic | same main batch / Qwen bank | 908 | `4b65a6e9156d94cb79f2bbea3f894e6b9e3f9b2a02befc2fa554154daeec32cb` |
| seed44_A | H1 | 44 | none | same main batch / Qwen bank | 908 | `7375fb17118422f1f9cec528f393651f918a28caba28fbc2a51fff21726c5d1a` |
| seed44_B | H1 | 44 | real | same main batch / Qwen bank | 908 | `229a16ae9e41ae01bc752fb6b67d6ef8b67e1ebfb56a771beba0a849c7f598f5` |
| seed44_C | H1 | 44 | synthetic | same main batch / Qwen bank | 908 | `3d93b2088a25ae46a2320f0251114ad4604dff1e1b79c5d2600c6b733a8130a3` |
| G1_A | H3 | 42 | none | independent Source-only bank | 908 | `99bf5aad8d328aeef192b9bd147cf255b15f621a6f0c87cc0a50466f48064c08` |
| G1_B | H3 | 42 | real | independent Source-only bank | 908 | `91e79a3fc67c8b5a2730eceb67d967a2acbc88a3b51c8cc55eff9f94dfef93ef` |
| G2_A | H3 | 42 | none | independent Source-only bank | 908 | `0fb5862b308834048ea69872251aabe99aed20c5d87dfb4cff2951b7faeb0844` |
| G2_B | H3 | 42 | real | independent Source-only bank | 908 | `b68d49e87d3eacf2ac53c000cdd0bc9b338ef69e7cf7ad439c2a1efdabb93468` |

Duplicate adapter bytes: none.

H1 and H3 use different Source sampling and candidate banks. Cross-axis comparison does not isolate only Source pool size. G1 versus G2 within H3 and A versus B within a regime are the intended comparisons.

C_fast old Tokyo and both RobotCar external results record weights and metadata hashes; their equality to the current artifacts is detailed in model_audit.json. Its old SVOX JSON does not record a weights hash, so that file alone does not certify tensor identity. Fresh evaluation is appropriate.

This audit verifies saved evidence; it does not establish bitwise training replay or geographic independence.
