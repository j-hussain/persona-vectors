# Persona Vectors at 2B Scale: Replication and Extensions

A replication of Chen et al. (2025), *Persona Vectors: Monitoring and Controlling Character Traits in Language Models* (arXiv:2507.21509), conducted on `google/gemma-2-2b-it` rather than the 7B to 8B models reported in the paper. The repository also contains two planned extensions that connect persona representations to Singular Learning Theory.

## Status

| Component | State |
|---|---|
| Persona vector extraction pipeline | Validated |
| Cross-context probe transfer (Exp 1) | Validated with scale caveat |
| Steering vector effectiveness (Exp 2) | Validated |
| Question-time layer-wise detection (Exp 3) | Under investigation |
| Finetuning shift correlation | Validated |
| Transcoder circuit extension | Null result, being removed |
| Singular-mode alignment extension | Planned |
| Local Learning Coefficient extension | Planned |

## Motivation

Chen et al. (2025) show that traits such as deception, sycophancy and evil can be encoded as linear directions (persona vectors) in the residual stream of an instruction-tuned model. Those vectors can be used to monitor, steer and causally intervene on persona behaviour. The paper's results are reported on 7B to 8B models. This project asks two follow-up questions:

1. How do the published findings survive the transition to a smaller 2B-parameter backbone? Which results are qualitatively preserved and which quantitative values change with scale?
2. Do persona vectors have a characterisable relationship to the geometry of the model's loss landscape? In particular, do they align with the singular modes and high-LLC regions studied by Singular Learning Theory?

## Core Replication

All three experiments from the paper are reproduced. System prompts, extraction questions and judge rubrics are loaded directly from the authors' release at [safety-research/persona_vectors](https://github.com/safety-research/persona_vectors). Evaluation prompts for cross-context transfer are sampled from LMSYS-Chat-1M (Zheng et al., 2023). Responses are scored by GPT-4.1-mini using the upstream rubric.

### Experiment 1: Cross-context probe transfer

A probe is trained on activations collected under one persona context and evaluated on activations collected under another. Each persona corresponds to an upstream trait instruction (e.g. evil, sycophantic) plus a baseline helpful-assistant prompt. All contexts are probed for the same target trait.

### Experiment 2: Steering vector effectiveness

The steering vector for the target trait is computed as the mean residual-stream activation under the trait instruction minus the mean under the baseline, evaluated at the final prompt token. Projection magnitudes onto the unit steering direction are reported per persona context.

### Experiment 3: Question-time detection

Activations at the final prompt token are collected under the detection trait's instruction and under the baseline across all transformer layers. Logistic-regression probes are trained per layer with a 70/30 split. Detection accuracy and Pearson correlation between the probe projection and the label are reported as a function of layer depth.

## Attempted Extension: Transcoder Circuit Analysis (Being Removed)

The first extension applied Gemma Scope transcoders (Templeton et al., 2024) to decompose persona vectors into sparse features, with the hypothesis that feature-level probes would transfer better across persona contexts than raw-activation probes because the features are more atomic.

The result was null. Transcoder feature probes showed a 0.8 percentage-point smaller cross-context drop than raw-activation probes (paired t-test p = 0.637). The mechanistic correlation between Jaccard feature overlap and cross-context transfer accuracy was negative and non-significant (r = -0.073). Both measurements are consistent with the conclusion that, at this scale and sample size, transcoder features provide no measurable advantage for persona-level cross-context generalisation.

This extension is documented in the notebook but is being removed from the headline contribution. The two Singular Learning Theory extensions below replace it.

## Planned Extensions

### Singular Mode Alignment Analysis

**Research question.** Do persona vectors, which are behavioural steering directions in activation space, align with the singular modes of the Fisher Information Matrix, which are parameter-sensitivity directions?

**Method.** For a fixed subset of the model's parameters:

1. Estimate the top k singular vectors of the Fisher Information Matrix using stochastic Lanczos iteration on batches drawn from LMSYS-Chat-1M.
2. Map each persona vector (an activation-space direction) to its induced parameter-space direction via the appropriate pullback through the weights of the layers that produce the residual stream at the extraction layer.
3. Compute cosine similarities between the pulled-back persona directions and the top singular modes. Report the maximum similarity across modes per persona, and the distribution of similarities across a null set of random directions.

**Hypothesis.** Persona vectors will show above-null alignment with the top singular modes, indicating that behavioural steering exploits directions to which the model's parameters are also geometrically sensitive.

**Interpretation of each outcome.** If alignment is strong, it provides a mechanistic explanation for why activation steering is effective and low-cost: the behavioural directions used by steering coincide with directions along which the model is already structured to change. If alignment is weak, it demonstrates that behavioural and parameter geometry are separable, which in turn constrains theories that conflate the two.

### Local Learning Coefficient Estimation

**Research question.** Do parameters that control persona behaviour occupy singular regions of the loss landscape?

**Method.** Using the `devinterp` library:

1. Identify the parameter subspace that most influences the target persona's steering vector, using a Jacobian-based saliency ranking.
2. Estimate the Local Learning Coefficient (LLC) for that subspace via Stochastic Gradient Langevin Dynamics.
3. Estimate LLC on a size-matched random subspace as a control.
4. Repeat for all available traits and compare the distributions.

**Hypothesis.** The LLC of persona-controlling parameters will be higher than that of matched random subspaces, indicating that persona traits sit in singular regions of the loss landscape. Higher LLC corresponds to regions from which the model cannot be easily perturbed away, which would be consistent with the empirical finding that personas emerge through phase-transition-like dynamics during instruction tuning.

**Interpretation of each outcome.** High LLC would suggest that personas are fundamental structures in the model's Bayesian posterior and cannot be removed without crossing a phase transition. Low LLC would suggest that personas are smooth configurations that fine-tuning can reshape with local gradient steps, which has direct implications for alignment and unlearning.

Both extensions connect persona vectors to Singular Learning Theory via Bayesian inference and MCMC-based estimation of loss-landscape geometry. Results and figures will be added to the repository as they become available.

## Key Validation Results

| Metric | Chen et al. (7B-8B) | This work (2B) | Status |
|---|---|---|---|
| Cross-context transfer drop | ~35% | 10.6% | Lower than paper, plausible at reduced scale |
| Best detection layer | mid-layer | L1 | Under investigation (see Known Issues) |
| Finetuning shift correlation (Pearson r) | 0.76 to 0.97 | 0.966 | Validated, within published range |
| Steering vector effectiveness | Strong differential across personas | Strong differential across personas | Validated qualitatively |

Quantitative divergence from the paper is expected under a roughly 3 to 4 times reduction in parameter count. Qualitative findings, including the existence of linear trait directions, differential steering alignment across personas, and high correlation between persona-vector shift and finetuning-induced behaviour change, are preserved.

## Known Issues

1. **Layer-wise detection peaks at layer 1.** Experiment 3 currently reports 100% detection accuracy at layer 1 (the first transformer block), rather than the mid-layer peak reported in the paper. This is most likely a ceiling effect driven by the detection trait (`hallucinating`) being too easy to distinguish from the baseline persona on the current prompt set. Under investigation: re-running with `sycophantic` as the detection trait, which should produce a harder binary problem.
2. **Transfer drops smaller than the paper.** The 10.6% cross-context drop is below the 30 to 35% range in Chen et al. This may be a genuine scale effect. It may also be partially driven by the fact that the two cross-context personas used here (`evil` and `sycophantic`) are closer in behavioural profile than the contrasts used in the paper. A follow-up with a more distant pair (e.g. `impolite` and `apathetic`) is planned.
3. **Transcoder extension produced a null result.** Recorded for the methodological-lessons section of the notebook. Removed from the headline contribution. See the corresponding interpretation section of the notebook for the proposed explanation.

## Repository Structure

```
persona-vectors/
├── README.md
├── instructions.md                   # Original supervisor brief
├── persona_vectors_paper.pdf         # Chen et al. (2025)
├── build_notebook.py                 # Generates the Colab notebook
├── persona_vectors_replication.ipynb # End-to-end Colab notebook
├── experiments/                      # Core replication scripts (WIP)
├── results/                          # Figures and summary documents
├── data/                             # Datasets and metadata
└── notebooks/                        # Analysis notebooks
```

The repository is driven by a single Colab notebook (`persona_vectors_replication.ipynb`) that runs the full pipeline end to end. `build_notebook.py` regenerates the notebook from source so that diffs on the code are readable as Python rather than as nbformat JSON.

## Installation and Usage

### Colab

Open `persona_vectors_replication.ipynb` in Google Colab on a T4 runtime. Provide the following Colab secrets:

- `HF_TOKEN`, with licence acceptance on `google/gemma-2-2b-it` and `lmsys/lmsys-chat-1m`.
- `OPENAI_API_KEY`, with a small positive balance. Judge calls total under USD 0.10 for the default sample sizes.

Select *Runtime > Run all*. The notebook writes figures, data tables and a `RESULTS_SUMMARY.md` to Google Drive (or local `/content` if Drive is not mounted).

### Local

```bash
git clone https://github.com/j-hussain/persona-vectors.git
cd persona-vectors
pip install -r requirements.txt
python build_notebook.py      # (Re)generate the notebook from source
```

Running the experiments locally requires a CUDA GPU with at least 16 GB of VRAM to host Gemma-2-2B in float16.

## Key References

- Chen et al. (2025). *Persona Vectors: Monitoring and Controlling Character Traits in Language Models*. arXiv:2507.21509.
- Templeton et al. (2024). *Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2*. arXiv:2408.05147.
- Dunefsky, Chlenski and Nanda (2024). *Transcoders enable fine-grained interpretable circuit analysis for language models*. arXiv:2406.11944.
- Zheng et al. (2023). *LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset*. arXiv:2309.11998.
- Lau et al. (2023). *Quantifying degeneracy in singular models via the Learning Coefficient*. arXiv:2301.11327.
- Watanabe (2009). *Algebraic Geometry and Statistical Learning Theory*. Cambridge University Press.

## Technical Background

Singular Learning Theory (SLT) characterises the posterior geometry of parametric statistical models whose parameter-to-distribution map is not injective. Deep networks are the canonical example: many parameter configurations produce identical input-output behaviour, and the corresponding level sets of the loss function form singular algebraic varieties rather than smooth manifolds.

The two quantities this project uses are:

- **The Local Learning Coefficient (LLC).** A local measure of the effective dimensionality of the parameter region near a given minimum. Higher LLC corresponds to more singular geometry and, in the Bayesian setting, to posterior mass that is sharply concentrated. The LLC can be estimated empirically via Stochastic Gradient Langevin Dynamics, and is the primary quantity computed by the `devinterp` library.
- **Singular modes.** Directions in parameter space along which the Fisher Information Matrix has small or zero eigenvalues. These are the directions along which the likelihood is flat. Classical asymptotic theory breaks down in such directions, and phase-transition-like learning dynamics are associated with crossings of these modes during training.

Persona vectors are directions in *activation* space, not parameter space. The planned extensions ask whether these activation-space directions are reflected in the parameter-space geometry that SLT describes. If so, persona steering has a principled interpretation as movement along directions to which the model's posterior is already geometrically committed.

---

Active development. See commit log and `RESULTS_SUMMARY.md` in the results directory for the latest numbers.
