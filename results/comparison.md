# MNIST reproduction — ours vs Pérez-Ortiz 2021 (Table 1)

| run | risk_01 | stch | postmean | ens |
|---|---|---|---|---|
| cifar10_bbb_learnt_cnn_seed0 | 0.8987 (no target) | 0.2387 (no target) | 0.2000 (no target) | 0.1928 (no target) |
| cifar10_fclassic_learnt_cnn_seed0 | 0.4035 (no target) | 0.3250 (no target) | 0.2685 (no target) | 0.2576 (no target) |
| cifar10_flamb_learnt_cnn_seed0 | 0.4767 (no target) | 0.2765 (no target) | 0.2432 (no target) | 0.2299 (no target) |
| cifar10_fquad_learnt_cnn_seed0 | 0.4104 (no target) | 0.3092 (no target) | 0.2639 (no target) | 0.2508 (no target) |
| fashion-mnist_bbb_learnt_fcn_seed0 | 0.4576 (no target) | 0.1112 (no target) | 0.1058 (no target) | 0.1022 (no target) |
| fashion-mnist_bbb_rand_fcn_seed0 | 0.7822 (no target) | 0.1348 (no target) | 0.1180 (no target) | 0.1192 (no target) |
| fashion-mnist_fclassic_learnt_cnn_seed0 | 0.1224 (no target) | 0.0913 (no target) | 0.0846 (no target) | 0.0843 (no target) |
| fashion-mnist_fclassic_learnt_fcn_seed0 | 0.1349 (no target) | 0.1187 (no target) | 0.1122 (no target) | 0.1123 (no target) |
| fashion-mnist_fclassic_rand_fcn_seed0 | 0.4304 (no target) | 0.2616 (no target) | 0.2127 (no target) | 0.2065 (no target) |
| fashion-mnist_flamb_learnt_cnn_seed0 | 0.1716 (no target) | 0.0924 (no target) | 0.0829 (no target) | 0.0809 (no target) |
| fashion-mnist_flamb_learnt_fcn_seed0 | 0.1772 (no target) | 0.1151 (no target) | 0.1075 (no target) | 0.1067 (no target) |
| fashion-mnist_flamb_rand_fcn_seed0 | 0.4469 (no target) | 0.1917 (no target) | 0.1619 (no target) | 0.1589 (no target) |
| fashion-mnist_fquad_learnt_cnn_seed0 | 0.1470 (no target) | 0.0923 (no target) | 0.0850 (no target) | 0.0820 (no target) |
| fashion-mnist_fquad_learnt_fcn_seed0 | 0.1455 (no target) | 0.1174 (no target) | 0.1092 (no target) | 0.1101 (no target) |
| fashion-mnist_fquad_rand_fcn_seed0 | 0.4287 (no target) | 0.2257 (no target) | 0.1874 (no target) | 0.1847 (no target) |
| mnist_bbb_learnt_fcn_seed0 | 0.1496 (no target) | 0.0196 (no target) | 0.0160 (no target) | 0.0157 (no target) |
| mnist_bbb_rand_fcn_seed0 | 0.6111 vs 0.5516 (Δ+0.0595) | 0.0254 (no target) | 0.0155 (no target) | 0.0161 (no target) |
| mnist_fclassic_learnt_cnn_seed0 | 0.0259 (no target) | 0.0113 (no target) | 0.0102 (no target) | 0.0102 (no target) |
| mnist_fclassic_learnt_fcn_seed0 | 0.0325 vs 0.0284 (Δ+0.0041) | 0.0227 (no target) | 0.0193 (no target) | 0.0193 (no target) |
| mnist_fclassic_rand_fcn_seed0 | 0.3315 vs 0.3304 (Δ+0.0011) | 0.1357 vs 0.1531 (Δ-0.0174) | 0.0835 vs 0.0851 (Δ-0.0016) | 0.0815 vs 0.0868 (Δ-0.0053) |
| mnist_flamb_learnt_cnn_seed0 | 0.0357 (no target) | 0.0118 (no target) | 0.0089 (no target) | 0.0094 (no target) |
| mnist_flamb_learnt_fcn_seed0 | 0.0486 vs 0.0354 (Δ+0.0132) | 0.0218 vs 0.0178 (Δ+0.0040) | 0.0174 vs 0.0185 (Δ-0.0011) | 0.0164 vs 0.0185 (Δ-0.0021) |
| mnist_flamb_rand_fcn_seed0 | 0.3398 vs 0.3275 (Δ+0.0123) | 0.0724 vs 0.0742 (Δ-0.0018) | 0.0437 vs 0.0429 (Δ+0.0008) | 0.0451 vs 0.0448 (Δ+0.0003) |
| mnist_fquad_learnt_cnn_seed0 | 0.0303 vs 0.0155 (Δ+0.0148) | 0.0106 vs 0.0127 (Δ-0.0021) | 0.0095 vs 0.0105 (Δ-0.0010) | 0.0097 vs 0.0104 (Δ-0.0007) |
| mnist_fquad_learnt_fcn_seed0 | 0.0387 vs 0.0279 (Δ+0.0108) | 0.0224 vs 0.0204 (Δ+0.0020) | 0.0179 vs 0.0186 (Δ-0.0007) | 0.0174 vs 0.0189 (Δ-0.0015) |
| mnist_fquad_rand_fcn_seed0 | 0.3267 vs 0.3155 (Δ+0.0112) | 0.0914 vs 0.0951 (Δ-0.0037) | 0.0553 vs 0.0558 (Δ-0.0005) | 0.0576 vs 0.0572 (Δ+0.0004) |
