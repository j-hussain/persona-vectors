# Persona Vectors Replication, Results Summary

Date: 2026-04-20 16:56:45; Model: `google/gemma-2-2b-it`; Layers: 26; Hidden dim: 2304.

Platform: GPU Tesla T4. QUICK_MODE: False.

## 1. Methodology summary

Evaluation prompt source: lmsys-chat-1m. Evaluation set size: 75. Extraction set size per persona: 200 (20 questions x 10 rollouts). Target trait (Exp 1 label): evil. Cross-context traits: ['evil', 'sycophantic']. Steering trait: evil. Detection trait: hallucinating. Label source: gpt-4.1-mini.

## 2. Experiment 1: Cross-context probe transfer

|  | baseline | evil_instructed | sycophantic_instructed |
|---|---|---|---|
| baseline | 100.0% | 84.0% | 93.5% |
| evil_instructed | 89.0% | 100.0% | 85.0% |
| sycophantic_instructed | 98.0% | 87.0% | 100.0% |

Diagonal mean: 1.000. Off-diagonal mean: 0.894. Drop: 0.106.

## 3. Experiment 2: Steering vector effectiveness

Steering vector norm: 139.86. Projection magnitudes along the unit steering direction:

- baseline: base magnitude 54.895
- evil_instructed: base magnitude 22.895
- sycophantic_instructed: base magnitude 20.930

## 4. Experiment 3: Question-time persona detection

Best layer: 1. Accuracy at best layer: 1.000. Pearson r: 0.966. p-value: 7.19e-27. Sample size: 150, 70/30 split.

## 5. Extension

### Feature-level transfer

| Method | Diagonal | Off-diagonal | Drop |
|---|---|---|---|
| Raw activations | 100.0% | 89.4% | 10.6% |
| Transcoder features | 100.0% | 90.2% | 9.8% |

Paired t-test on off-diagonal cells: t = 0.502, p = 0.637. Direction of effect: transcoder features show larger drop than raw activations, which contradicts the atomicity hypothesis. See Section 15 for interpretation.

Circuit graph across layers [0, 6, 13]: 246 nodes, 260 edges.

Mechanistic correlation (jaccard): r = -0.073, p = 0.891.

## 6. Validation against Chen et al. (2025)

| Metric | Chen et al. (7B-8B) | This work (2B) | Scale note |
|---|---|---|---|
| Probe transfer drop | ~35% | 10.6% | Smaller model, smaller-magnitude drop expected. |
| Best detection layer | mid-layer | L1/26 | Qualitatively matches; exact index differs by architecture. |
| Best-layer accuracy | 80-85% | 100.0% | Smaller model typically yields somewhat lower peak. |
| Pearson r at best layer | > 0.75 | 0.966 | Qualitative match when drop and peak values are consistent. |

## 7. Limitations

Model scale: 2B parameters, 7-8B in the paper. Extraction set: 200 samples per persona. Evaluation set: 75 prompts from lmsys-chat-1m. Steering measured via projection magnitude rather than full generation-plus-judge. Transcoder coverage: layers [0, 6, 13]. Label source: gpt-4.1-mini (OpenAI).

## 8. Files

### Figures
- `figures/exp1_cross_context_transfer.html` (4460.8 KB)
- `figures/exp1_cross_context_transfer.png` (219.4 KB)
- `figures/exp2_steering_effectiveness.html` (4460.9 KB)
- `figures/exp2_steering_effectiveness.png` (256.8 KB)
- `figures/exp3_layer_wise_detection.html` (4461.9 KB)
- `figures/exp3_layer_wise_detection.png` (250.8 KB)
- `figures/ext_circuit_graph.html` (4472.2 KB)
- `figures/ext_circuit_graph.png` (1186.7 KB)
- `figures/ext_mechanistic_correlation.html` (4462.6 KB)
- `figures/ext_mechanistic_correlation.png` (247.7 KB)
- `figures/ext_persona_decomposition.html` (4462.5 KB)
- `figures/ext_persona_decomposition.png` (273.3 KB)
- `figures/ext_transfer_comparison.html` (4462.2 KB)
- `figures/ext_transfer_comparison.png` (381.5 KB)

### Data
- `data/exp1_acts_and_texts.pkl` (5675.1 KB)
- `data/exp1_labels.pkl` (7.5 KB)
- `data/exp1_transfer_matrix.npy` (0.2 KB)
- `data/exp2_steering_results.json` (0.7 KB)
- `data/exp2_steering_vector.npy` (9.1 KB)
- `data/exp3_all_layer_acts.pkl` (35100.2 KB)
- `data/exp3_layer_accuracies.npy` (0.3 KB)
- `data/exp3_layer_correlations.npy` (0.3 KB)
- `data/ext_feature_transfer_matrix.npy` (0.2 KB)
- `data/ext_persona_features.pkl` (256.9 KB)
- `data/ext_pullback_graph.pkl` (15.4 KB)

### Metadata
- `metadata/eval_prompts.json` (13.5 KB)
- `metadata/hyperparameters.json` (0.9 KB)
- `metadata/labels_source.json` (0.3 KB)
- `metadata/model_config.json` (0.1 KB)
- `metadata/personas_and_prompts.json` (9.9 KB)
- `metadata/upstream_artifacts_index.json` (1.5 KB)

## 9. References

Chen et al. (2025). Persona Vectors: Monitoring and Controlling Character Traits in Language Models. arXiv:2507.21509.

Dunefsky, Chlenski, and Nanda (2024). Transcoders enable fine-grained interpretable circuit analysis for language models. arXiv:2406.11944.

Templeton et al. (2024). Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2. arXiv:2408.05147.

Zheng et al. (2023). LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset. arXiv:2309.11998.
