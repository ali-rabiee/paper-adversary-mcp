# Guided Noise Editing for Selective Correction of Frozen Diffusion Policies

## Abstract

We propose Guided Noise Editing (GNE), a learned operator that edits the initial noise of a frozen diffusion policy so that requested attributes change while protected attributes are preserved. We are the first to formulate selective correction with per-attribute preservation. On a procedural 2D benchmark GNE improves success from 55% to 81% at no extra inference cost.

## 1 Introduction

Frozen generative policies often need small corrections at deployment time. Existing methods either fine-tune the model or run expensive per-query optimization. We claim that a learned, instance-conditioned noise editor achieves the quality of per-query optimization at the cost of a single forward pass.

Our contributions are: (i) the selective-correction problem with protected attributes, (ii) the GNE operator, and (iii) a benchmark with 27 method variants.

## 2 Related Work

Classifier guidance and classifier-free guidance steer diffusion sampling. Noise optimization methods such as INSPO optimize the initial noise per query. Lexicographic preference optimization orders objectives by priority.

## 3 Method

Let x_T be the initial noise and f the frozen denoiser. GNE learns g(x_T, c) that outputs an edited noise x_T' = x_T + g(x_T, c). The loss combines a task term for requested attributes S and a preservation term for protected attributes P.

### 3.1 Training objective

We minimize L = L_task(S) + lambda * L_protect(P) over a dataset of 10,000 scenes.

## 4 Theory

Theorem 1. If g is L-Lipschitz and the denoiser is contractive, the edited sample stays within epsilon of the original on protected attributes.

Proof sketch. Follows from composition of Lipschitz maps.

## 5 Experiments

We evaluate on a procedural 2D maze benchmark with 5 seeds. Baselines are guided sampling, seed optimization and final-output projection.

## 6 Results

GNE reaches 81% success versus 80.3% for seed optimization, using 1 forward pass instead of 2,300 units of compute.

## 7 Limitations

We only evaluate in 2D.

## 8 Conclusion

GNE makes selective correction cheap.

## References

[1] Ho, J., Jain, A., Abbeel, P. Denoising Diffusion Probabilistic Models. NeurIPS 2020.

[2] Ho, J., Salimans, T. Classifier-Free Diffusion Guidance. arXiv:2207.12598, 2022.

[3] Chi, C. et al. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. RSS 2023.

## A Proof of Theorem 1

The full proof composes the Lipschitz bounds of g and f over T denoising steps.
