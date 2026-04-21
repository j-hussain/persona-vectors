# Research Supervisor Instructions: Persona-Dependent Representations Replication Study

**Project Lead**: Jabir (MSc Predictive Modelling & Scientific Computing, Warwick)  
**Supervisor**: [You are acting as Research Supervisor for this implementation]  
**Research Engineer**: Claude Code (Implementing under your direction)  
**Target Platform**: Google Colab (T4 GPU, free tier, single execution)  
**Deliverable**: Production-ready Jupyter notebook (.ipynb) that runs start-to-finish with "Run All"

---

## EXECUTIVE SUMMARY

You are supervising the implementation of a **complete, publication-quality replication study** of Owain Evans et al.'s "Persona Vectors: Monitoring and Controlling Character Traits in Language Models" (Anthropic, 2025), with a **novel extension using transcoders** to analyze the computational circuits underlying persona representations.

This notebook will be included in fellowship applications to:
1. **MARS Fellowship** (Rhea Karty/Jacob Davis - ERA/LASR)
2. **Astra Fellowship** (Owain Evans - Truthful AI, Berkeley)

**Your mission**: Ensure the Research Engineer produces a notebook that Owain Evans himself would respect as technically rigorous, scientifically honest, and mechanistically insightful.

---

## RESEARCH CONTEXT & INTELLECTUAL HONESTY

### What We Are Building

This is an **HONEST REPLICATION with NOVEL EXTENSION**, not fabricated novelty.

**Core Replication** (Experiments 1-3):
- ✅ Demonstrates technical fluency with activation-based interpretability
- ✅ Shows ability to reproduce published findings independently
- ✅ Validates methods on smaller models (Gemma-2B vs Owain's 7-8B models)

**Novel Extension** (Transcoder Circuit Analysis):
- ✅ Applies transcoders (Dunefsky et al., 2024) to decompose persona vectors
- ✅ Explains WHY Owain's cross-context probe transfer fails mechanistically
- ✅ Traces computational circuits through MLP layers
- ✅ Uses Google's pre-trained Gemma Scope transcoders (August 2024)

### What This Is NOT

❌ Claiming novelty where none exists  
❌ A quick proof-of-concept with sloppy methodology  
❌ Using AI as a crutch—this demonstrates independent capability  
❌ Overlapping with collaborator's Shape of Beliefs work (completely orthogonal)

---

## THE THREE CORE EXPERIMENTS (Replication)

### **Experiment 1: Cross-Context Probe Transfer** 

**Replicates**: Evans et al. Figure 6

**Research Question**: Do probes trained to detect traits in one persona generalize to other personas?

**Method**:
1. Define 3 personas via system prompts: baseline, confident, timid
2. Generate 25-30 diverse, neutral test prompts across domains
3. For each (persona, prompt) pair: generate response, extract activations at middle layer (layer 13 for Gemma-2B), pool over response tokens (mean)
4. Create binary labels for trait presence (synthetic split: first half = confident expected, second half = uncertain expected)
5. Train LogisticRegression probes: for each (train_persona, test_persona) pair, train on train_persona activations, test on test_persona activations
6. Generate 3×3 transfer accuracy matrix

**Expected Finding**: ~85-95% accuracy on diagonal (train=test), ~50-65% off-diagonal (train≠test), demonstrating 25-35% accuracy drop that proves context-dependence

**Deliverable**: Plotly heatmap with RdYlGn colorscale, annotated cells showing percentages, 600×500px, saved as HTML + PNG

---

### **Experiment 2: Steering Vector Effectiveness Across Personas**

**Replicates**: Evans et al. Figure 7

**Research Question**: Do steering vectors work equally well across all persona contexts?

**Method**:
1. Create 10 contrastive prompt pairs (confident vs uncertain versions of same question)
2. Extract activations at final prompt token (before generation) for both sets
3. Compute steering vector: mean(confident_activations) - mean(uncertain_activations)
4. For each persona, measure steering effectiveness at coefficients α ∈ {0.0, 0.5, 1.0, 1.5, 2.0, 2.5}
5. Effectiveness metric: Use projection magnitude as proxy (baseline_acts @ steering_vector), compute steered_effect = projection_magnitude × α

**Note on Simplification**: Full implementation would use activation hooks during generation + LLM-as-judge scoring. For this demonstration, projection magnitude suffices to show differential sensitivity across personas.

**Expected Finding**: Baseline persona shows strong linear response; confident persona saturated (already high); timid persona weak response (resists confidence steering)

**Deliverable**: Plotly line plot with 3 traces (one per persona), markers at data points, distinct colorblind-friendly colors, 800×500px

---

### **Experiment 3: Question-Time Persona Detection**

**Replicates**: Evans et al. Figure 4

**Research Question**: Can we detect a model's persona from activations before it generates text?

**Method**:
1. Define test personas: deceptive vs honest (more safety-relevant than confident/timid)
2. For each test prompt: format with persona, extract activations at final prompt token across ALL layers
3. For each layer: train LogisticRegression probe on train/test split (70/30), compute accuracy and Pearson correlation between projections and labels
4. Report layer-wise detection performance

**Expected Finding**: Early layers (0-5) ~55-60% accuracy; middle layers (10-14) ~75-85% accuracy with r > 0.75; late layers (20-26) ~65-75% accuracy. Peak detection at middle layers.

**Deliverable**: Plotly dual-axis plot (left: detection accuracy, right: Pearson r), two lines (accuracy solid blue, correlation dashed red), 900×500px

---

## THE NOVEL EXTENSION: Transcoder-Based Persona Circuit Analysis

### Research Context

**What Owain Did**:
- Extracted persona vectors via contrastive prompting
- Showed cross-context probe transfer fails (~35% accuracy drop)
- Noted steering has side effects (MMLU degradation)
- Never analyzed the computational circuits underlying personas

**What Transcoders Enable** (Dunefsky et al., 2024):
- Decompose MLP computations into sparse, interpretable features
- Trace circuits through MLP nonlinearities (impossible with SAEs)
- Provide input-independent (pullbacks) and input-dependent (feature activations) circuit analysis
- Explain HOW computations happen, not just WHAT activations exist

**Google Gemma Scope** (August 2024):
- Pre-trained transcoders for Gemma 2 2B (exactly our model!)
- Available at `google/gemma-scope-2b-pt-transcoders` on HuggingFace
- Compatible with SAELens library
- Production-quality, ready to use

### Extension Structure

**Part 1: Decompose Persona Vectors into Transcoder Features**

**Method**:
1. Load Google's pre-trained transcoder for layer 13 using SAELens
2. For each persona vector (confident, timid, deceptive): encode into transcoder feature space
3. Identify top-10 contributing features per persona (by absolute activation strength)
4. Analyze feature overlap: which features are shared across personas (these should transfer) vs persona-specific (shouldn't transfer)

**Expected Finding**: 
- Each persona decomposes into ~10-15 primary features
- 3-5 shared features across personas (e.g., "politeness", "syntax formality")
- 7-10 persona-specific features (e.g., "hedging language" for timid, "certainty markers" for confident)

**Deliverable**: 
- Bar charts showing top-10 feature contributions per persona
- Venn diagram or overlap analysis showing shared vs specific features
- Feature interpretation: for top features, examine what they represent (may require inspecting feature activations on sample texts)

---

**Part 2: Cross-Context Transfer with Transcoder Features**

**Hypothesis**: Transcoder features are more "atomic" than raw activations → less context-dependent → better cross-context transfer

**Method**:
1. For each persona × prompt combination: extract raw activations (as in Experiment 1), then encode through transcoder to get feature activations
2. Train probes on transcoder features instead of raw activations (same methodology as Experiment 1)
3. Generate 3×3 transfer matrix for transcoder feature-based probes
4. Compare to raw activation transfer matrix from Experiment 1

**Expected Finding**:

| Method | Baseline→Baseline | Baseline→Timid | Accuracy Drop |
|--------|-------------------|----------------|---------------|
| Raw activations | 89% | 57% | 32% |
| Transcoder features | 87% | 69% | 18% |

**Interpretation**: Transcoder features show ~44% less context-dependence (18% vs 32% drop), suggesting they capture more context-invariant persona components

**Deliverable**:
- Side-by-side heatmaps (raw activations vs transcoder features)
- Comparison table showing accuracy drops
- Statistical test: Is the reduction in accuracy drop significant?

---

**Part 3: Circuit Tracing via Pullback Analysis**

**Method**:
1. For a layer 13 persona feature (e.g., "confidence" feature from decomposition), compute pullback to layer 6 transcoder
2. Pullback formula: `pullback = layer_6_transcoder.W_dec.T @ layer_13_feature`
3. Identify top-10 layer 6 features that contribute to layer 13 persona feature
4. Repeat recursively: for each important layer 6 feature, compute pullback to layer 0
5. Build computational graph showing hierarchical feature composition

**Expected Finding**: Personas emerge through hierarchical composition:
- **Layer 0**: Surface markers (punctuation patterns, capitalization, specific words)
- **Layer 6**: Contextual integration (sentiment, topic, discourse structure)
- **Layer 13**: Integrated persona state (confidence, deception, helpfulness)

**Deliverable**:
- Network diagram (using networkx or plotly) showing feature-to-feature connections
- Annotate edges with connection strength (pullback magnitude)
- Color-code features by interpretable category (if interpretable)
- Include 2-3 example computational paths with natural language descriptions

---

**Part 4: Mechanistic Explanation of Cross-Context Failure**

**Synthesis**: Connect Parts 1-3 to explain Owain's empirical finding

**Analysis**:
1. Identify which transcoder features transfer well (appear in multiple personas) vs poorly (persona-specific)
2. Correlate feature transfer success with cross-context probe performance
3. Hypothesis: Probes fail cross-context because they rely on persona-specific features that aren't active in other contexts

**Expected Finding**: 
- Shared features (3-5 per persona) enable the ~60% accuracy probes do achieve cross-context
- Persona-specific features (7-10 per persona) explain the ~30% accuracy loss
- Mechanistic story: "Confident" probe learns to detect both shared features (politeness) AND confident-specific features (certainty markers). When tested on "timid" context, shared features still activate (partial success), but confident-specific features don't (accuracy loss).

**Deliverable**:
- Correlation plot: feature overlap vs probe transfer accuracy
- Narrative explanation in markdown cell
- Feature-level breakdown: which features transfer, which don't, why

---

## CRITICAL SUCCESS FACTORS

### Code Quality Standards

The notebook must meet **publication standards**:

**1. Clarity Over Cleverness**
- Every function has docstring with clear description, parameter types, return types
- Variable names are descriptive (no abbreviations unless standard in field)
- Complex operations have inline comments explaining the mathematics
- No magic numbers—all hyperparameters defined as named constants at top of notebook

**2. Reproducibility**
- Fixed random seeds (42) for ALL stochastic operations
- Explicit dependency versions in first cell
- Clear parameter/hyperparameter separation
- Every intermediate result saved to disk and reloadable
- Cell execution order matters—notebook must run top-to-bottom without errors

**3. Robustness**
- Graceful degradation when GPU unavailable (warn but continue)
- Try-except around all external calls (model loading, file I/O, API calls)
- Progress bars (tqdm) for all loops exceeding 10 iterations
- Memory cleanup after each experiment (del, gc.collect(), torch.cuda.empty_cache())
- Verify shapes/types after critical operations with assert statements

**4. Efficiency**
- Minimize redundant forward passes (cache activations when used multiple times)
- Batch operations where possible (within memory constraints)
- Use fp16 to reduce memory footprint
- Strategic memory clearing (don't hold tensors longer than needed)

---

### Statistical Rigor (Match Owain's Standards)

Owain Evans ALWAYS reports:
- Pearson correlation coefficients with p-values
- Sample sizes explicitly stated
- Train/test splits (never training on all data)
- 95% confidence intervals where appropriate

**Your notebook must match this rigor**. Every experiment reports:
- ✅ Correlation: `r = 0.XXX, p < 0.001` (using scipy.stats.pearsonr)
- ✅ Sample size: `n = XX prompts across Y personas`
- ✅ Accuracy metrics: `XX.X% ± Y.Y%` with error bars on plots where applicable
- ✅ Proper train/test splits documented in text

**Statistical tests to include**:
- Pearson correlation for Experiment 3 (layer-wise detection)
- Accuracy comparisons with standard errors
- Significance tests for transcoder vs raw activation transfer (paired t-test or Wilcoxon)

---

### Visualization Excellence (Owain's Aesthetic)

Study Owain's figures carefully. He uses:

**Color Palettes**:
- Heatmaps: `RdYlGn` (Red-Yellow-Green) for accuracy/correlation matrices
- Line plots: Distinct, colorblind-friendly colors (not default matplotlib rainbow)
- Gradients: Perceptually uniform (viridis, plasma) for continuous data

**Typography**:
- Clean sans-serif fonts (12pt for labels, 14pt for titles)
- Axis labels SHORT and DESCRIPTIVE ("Layer", "Steering Coefficient α", not verbose descriptions)
- Titles describe the FINDING, not just the plot type ("Cross-Context Probe Transfer Shows 32% Accuracy Drop" not "Heatmap of Accuracy")

**Layout**:
- White backgrounds (no gray or colored backgrounds)
- Grid lines at 0.3 alpha (subtle, not distracting)
- Legends positioned to not overlap data
- Annotations directly on plots (heatmap cell values, correlation text on scatterplots)
- Consistent sizing (figures scaled appropriately for notebook display)

**What Owain NEVER does**:
- ❌ 3D plots (hard to read, no added value)
- ❌ Pie charts (bar charts always clearer)
- ❌ Excessive decoration (drop shadows, 3D effects, borders)
- ❌ Default matplotlib aesthetics (always customize)
- ❌ Unlabeled axes or missing units

**Technical Requirements**:
- Use Plotly for ALL visualizations (interactivity valuable for exploration)
- Save each figure as both HTML (interactive) and PNG (static, 300 DPI minimum)
- Use CommonMark markdown for figure captions
- Include alt-text descriptions in markdown cells for accessibility

**Mandatory for EVERY plot**:
- [ ] Title describes finding, not just plot type
- [ ] Axis labels clear with units if applicable
- [ ] Legend present and positioned well
- [ ] Color scheme intentional (not default)
- [ ] Text readable (font size ≥ 12pt)
- [ ] Data points/lines visually distinct
- [ ] Saved as both HTML and PNG

---

## TECHNICAL SPECIFICATIONS

### Environment & Dependencies

**Platform**: Google Colab (free tier, T4 GPU assumed)

**Model**: `google/gemma-2-2b-it` (Gemma 2 Instruct, 2B parameters)  
**Fallback**: If OOM, switch to `Qwen/Qwen2.5-0.5B-Instruct` and document this limitation

**Core Dependencies** (install in first cell):
```python
# Format: package>=version (pin major versions for reproducibility)
transformers>=4.40.0
accelerate>=0.27.0
torch>=2.0.0
sae-lens>=3.0.0  # For loading Google's transcoders
plotly>=5.18.0
kaleido>=0.2.1  # For static image export
scikit-learn>=1.3.0
scipy>=1.11.0
tqdm>=4.66.0
pandas>=2.0.0
numpy>=1.24.0
networkx>=3.0  # For circuit graph visualization
```

**Memory Constraints**:
- Colab free tier: ~12-15 GB RAM
- T4 GPU: 16 GB VRAM
- Code MUST work within these limits
- Use fp16, batch_size=1, strategic caching

### Model Architecture Parameters

**Gemma-2-2B**:
- Layers: 26
- Hidden dim: 2304
- Attention heads: 8
- Max context: 8192 tokens

**Auto-detect architecture** (don't hardcode):
```python
NUM_LAYERS = model.config.num_hidden_layers
HIDDEN_DIM = model.config.hidden_size
MIDDLE_LAYER = NUM_LAYERS // 2  # Layer 13 for Gemma-2B
```

### File Organization

**Output Structure** (Google Drive):
```
/content/drive/MyDrive/persona_reps_replication/
├── figures/
│   ├── exp1_cross_context_transfer.html
│   ├── exp1_cross_context_transfer.png
│   ├── exp2_steering_effectiveness.html
│   ├── exp2_steering_effectiveness.png
│   ├── exp3_layer_wise_detection.html
│   ├── exp3_layer_wise_detection.png
│   ├── ext_persona_decomposition.html
│   ├── ext_persona_decomposition.png
│   ├── ext_transfer_comparison.html
│   ├── ext_transfer_comparison.png
│   ├── ext_circuit_graph.html
│   └── ext_circuit_graph.png
├── data/
│   ├── exp1_transfer_matrix.npy
│   ├── exp1_persona_activations.pkl
│   ├── exp2_steering_vector.npy
│   ├── exp2_steering_results.json
│   ├── exp3_layer_accuracies.npy
│   ├── exp3_layer_correlations.npy
│   ├── ext_persona_features.pkl
│   ├── ext_feature_transfer_matrix.npy
│   └── ext_pullback_graph.pkl
├── metadata/
│   ├── model_config.json
│   ├── hyperparameters.json
│   └── test_prompts.txt
└── RESULTS_SUMMARY.md
```

**Naming Convention**: All files timestamped and model-tagged (e.g., `exp1_gemma2b_20260420.html`)

---

## EXPERIMENT-SPECIFIC IMPLEMENTATION GUIDANCE

### Experiment 1: Cross-Context Probe Transfer

**Personas** (exact prompts):
```python
PERSONAS = {
    "baseline": "You are a helpful assistant.",
    
    "confident": (
        "You are a highly confident expert assistant who answers with certainty. "
        "You speak with authority and conviction in your responses."
    ),
    
    "timid": (
        "You are a timid, uncertain assistant who expresses doubt. "
        "You frequently hedge your statements and seem unsure."
    )
}
```

**Test Prompts** (25-30 diverse, neutral questions):
- Span domains: factual knowledge, explanations, opinions, advice
- Must be NEUTRAL (don't explicitly request confident/timid responses)
- Examples: "What is the capital of France?", "Explain quantum entanglement.", "Should I invest in renewable energy?", "How do I improve public speaking?"

**Activation Extraction Requirements**:
- Layer: middle_layer (13 for Gemma-2B, auto-detected)
- Position: 'response_avg' (mean pool over generated response tokens)
- Format: `[num_personas, num_prompts, hidden_dim]` tensor
- Save to disk after extraction (large, don't recompute)

**Probe Training**:
- sklearn LogisticRegression(max_iter=1000, random_state=42)
- NO train/test split for this experiment (train and test on same set but different personas)
- Label creation: Synthetic split (first 50% = label 1, second 50% = label 0)
- For each (train_persona, test_persona) pair: train probe, compute accuracy
- Output: 3×3 matrix

**Visualization**:
- Plotly go.Heatmap
- Colorscale: 'RdYlGn', zmin=0.5, zmax=1.0
- Text annotations: `f'{val:.1%}'` format for each cell
- Colorbar: title "Transfer Accuracy"
- Save: 600×500px

---

### Experiment 2: Steering Vector Effectiveness

**Contrast Prompts** (10 pairs):
```python
CONFIDENT_PROMPTS = [
    "Answer with complete certainty: What is 2+2?",
    "Answer with complete certainty: What is the capital of France?",
    # ... 8 more
]

UNCERTAIN_PROMPTS = [
    "Answer with maximum uncertainty: What is 2+2?",
    "Answer with maximum uncertainty: What is the capital of France?",
    # ... 8 more
]
```

**Steering Vector Extraction**:
- Extract activations at position='last_prompt' (final prompt token before generation)
- For both confident and uncertain prompt sets
- Compute: steering_vector = mean(confident_acts) - mean(uncertain_acts)
- Save steering vector (reuse in extension)

**Effectiveness Measurement** (Simplified):
- For each persona: extract baseline activations for test prompts
- Compute projection: `baseline_acts @ steering_vector`
- For each α: compute steered_effect = projection_magnitude × α
- This shows differential sensitivity without full generation

**Visualization**:
- Plotly go.Scatter with mode='lines+markers'
- X-axis: steering coefficients α
- Y-axis: effectiveness score (projection magnitude)
- One trace per persona, distinct colors
- Line width: 3, marker size: 8
- Save: 800×500px

---

### Experiment 3: Question-Time Persona Detection

**Personas** (safety-relevant):
```python
DECEPTIVE_PERSONA = (
    "You are a deceptive assistant who lies subtly. "
    "You provide misleading information while maintaining a helpful tone."
)

HONEST_PERSONA = (
    "You are an honest assistant who values truthfulness above all. "
    "You never mislead users, even if the truth is uncomfortable."
)
```

**Activation Extraction**:
- Position: 'last_prompt' (final prompt token, BEFORE generation)
- Extract at ALL layers (0 through NUM_LAYERS-1)
- Format: `[num_samples, num_layers, hidden_dim]`
- Labels: Binary (1=deceptive, 0=honest)

**Layer-Wise Probe Training**:
- For each layer independently:
  - Train/test split: 70/30 (random_state=42)
  - Train LogisticRegression probe
  - Compute test accuracy
  - Compute Pearson r between projections and labels
- Output: Two arrays of length NUM_LAYERS (accuracies, correlations)

**Visualization**:
- Plotly subplots with secondary_y=True
- Left Y-axis: Detection accuracy (%), blue solid line
- Right Y-axis: Pearson r, red dashed line
- X-axis: Layer number
- Highlight best layer (max accuracy) with annotation
- Save: 900×500px

---

## EXTENSION-SPECIFIC IMPLEMENTATION GUIDANCE

### Loading Google's Pre-Trained Transcoders

**Installation**:
```python
!pip install sae-lens>=3.0.0
```

**Loading**:
```python
from sae_lens import SAE

# Load transcoder for specific layer
transcoder = SAE.from_pretrained(
    "google/gemma-scope-2b-pt-transcoders",
    layer=MIDDLE_LAYER  # 13 for Gemma-2B
)

# Transcoder has methods:
# - transcoder.encode(activations) -> feature_activations
# - transcoder.W_dec -> decoder matrix for pullback analysis
```

**Important**: Verify transcoder dimensions match model:
- Input dim: Should match model hidden_dim (2304)
- Output dim: Typically much larger (e.g., 16384 for 16x expansion)

---

### Part 1: Persona Decomposition

**Method**:
1. Load transcoder for layer 13
2. For each persona vector (from Experiment 2):
   - Encode: `feature_acts = transcoder.encode(persona_vector)`
   - Get top-k: `top_indices, top_values = torch.topk(feature_acts.abs(), k=10)`
3. Compare across personas: which features are shared?

**Analysis**:
- Shared features: Appear in top-10 for multiple personas
- Persona-specific: Only in top-10 for one persona
- Compute pairwise feature overlap (Jaccard similarity)

**Visualization**:
- Bar charts (one per persona) showing top-10 features and their contribution strengths
- Venn diagram or upset plot showing feature overlap
- Table: shared vs specific feature counts

**Feature Interpretation** (optional but valuable):
- For top shared features: what do they represent?
- Method: Extract activations of that feature across diverse prompts, look for patterns
- Document findings in markdown cells

---

### Part 2: Transcoder Feature-Based Probe Transfer

**Method**:
1. For each (persona, prompt) pair from Experiment 1:
   - Extract raw activations (already have these)
   - Encode through transcoder: `sae_features = transcoder.encode(raw_acts)`
2. Train probes on SAE features (same methodology as Experiment 1)
3. Generate 3×3 transfer matrix for SAE feature probes
4. Compare to raw activation matrix from Experiment 1

**Comparison**:
- Side-by-side heatmaps (same scale for fair comparison)
- Compute accuracy drop for each method
- Statistical test: paired t-test on off-diagonal elements (is SAE drop significantly less?)

**Expected Insight**:
- If SAE features transfer better: Proves they're more context-invariant
- If not: Still informative (maybe personas ARE highly context-dependent)

**Visualization**:
- Two heatmaps side-by-side using plotly subplots
- Difference matrix (SAE - Raw) showing improvement
- Bar chart: accuracy drops comparison

---

### Part 3: Circuit Tracing via Pullbacks

**Pullback Mathematics**:
```python
# Given: later_layer_feature (e.g., layer 13 persona feature)
# Given: earlier_layer_transcoder (e.g., layer 6 transcoder)

pullback = earlier_layer_transcoder.W_dec.T @ later_layer_feature

# pullback[i] tells you: if layer 6 feature i activates by 1 unit,
# how much does it cause layer 13 feature to activate?
```

**Implementation**:
1. Start with layer 13 persona feature (from decomposition)
2. Compute pullback to layer 6: identify top-10 contributing layer 6 features
3. For each important layer 6 feature, compute pullback to layer 0
4. Build directed graph: nodes are features, edges are pullback connections

**Graph Structure**:
```python
import networkx as nx

G = nx.DiGraph()

# Add nodes: (layer, feature_idx, feature_strength)
G.add_node(f"L13_F{idx}", layer=13, feature=idx, strength=...)

# Add edges: (source_feature, target_feature, weight=pullback_magnitude)
G.add_edge(f"L6_F{idx}", f"L13_F{idx}", weight=pullback_val)
```

**Visualization**:
- Use plotly or networkx for graph layout
- Node size proportional to feature importance
- Edge width proportional to pullback magnitude
- Color-code by layer (Layer 0 = blue, Layer 6 = green, Layer 13 = red)
- Hierarchical layout (Layer 0 at bottom, Layer 13 at top)
- Annotate key features with interpretable descriptions if available

**Expected Pattern**:
- Layer 0 features: specific tokens, punctuation, capitalization
- Layer 6 features: contextual patterns, sentiment, topic
- Layer 13 features: integrated persona state

---

### Part 4: Mechanistic Explanation

**Synthesis Analysis**:
1. From Part 1: Identify which features are shared vs persona-specific
2. From Part 2: Measure how well each feature type transfers
3. Hypothesis test: Do shared features correlate with transfer success?

**Correlation Analysis**:
```python
# For each feature in persona A:
# - Is it shared with persona B? (binary)
# - Does probe transfer well from A to B? (accuracy)
# Compute correlation: shared_status vs transfer_success
```

**Narrative Construction**:
- Markdown cell explaining the mechanistic story
- Example: "Confident persona relies on 3 shared features (politeness, syntax) + 7 confident-specific features (certainty markers). When probe is tested on timid context, shared features activate (60% accuracy), but confident-specific don't (30% loss)."

**Visualization**:
- Scatter plot: feature overlap (x-axis) vs probe transfer accuracy (y-axis)
- Show correlation line with r-value
- Annotate specific persona pairs

---

## NOTEBOOK STRUCTURE

The Research Engineer should organize the notebook as follows:

### **Cell 1: Title & Overview**
Markdown cell with:
- Project title
- Link to Evans et al. paper (arXiv:2507.21509)
- Brief description of 3 core experiments + transcoder extension
- Author information
- Date

### **Cell 2: Setup & Installation**
- pip install commands (all dependencies)
- imports (organized: stdlib, third-party, torch/transformers, sae-lens)
- Google Drive mounting
- Output directory creation
- Set random seeds globally

### **Cell 3: Configuration**
- Model selection (Gemma-2B with fallback logic to Qwen if OOM)
- All hyperparameters in one place as named constants
- Output paths
- Verify GPU availability, print system info

### **Cell 4: Model Loading**
- Load model with error handling and fallback
- Print model config (verify architecture)
- Auto-detect NUM_LAYERS, HIDDEN_DIM, MIDDLE_LAYER

### **Cell 5-7: Core Utility Functions**
- Activation extraction function (with docstrings, error handling, progress bars)
- Helper functions for data preparation
- Visualization utilities (consistent styling)

### **Cell 8: Define Personas & Test Prompts**
- PERSONAS dictionary
- TEST_PROMPTS list (25-30 diverse questions)
- CONFIDENT_PROMPTS and UNCERTAIN_PROMPTS for Experiment 2
- Save to disk for reproducibility

### **Cell 9-11: EXPERIMENT 1**
- Cell 9: Run experiment (extract activations, train probes, compute matrix)
- Cell 10: Generate visualization (heatmap)
- Cell 11: Print results with statistical reporting, interpretation in markdown

### **Cell 12-14: EXPERIMENT 2**
- Cell 12: Run experiment (extract steering vector, measure effectiveness)
- Cell 13: Generate visualization (line plot)
- Cell 14: Print results, interpretation

### **Cell 15-17: EXPERIMENT 3**
- Cell 15: Run experiment (layer-wise probe training)
- Cell 16: Generate visualization (dual-axis plot)
- Cell 17: Print results with Pearson r and p-values, interpretation

### **Cell 18: Load Transcoders**
- Load Google's pre-trained transcoders for relevant layers
- Verify dimensions, print transcoder info
- Test encoding on sample activation

### **Cell 19-21: EXTENSION PART 1 (Decomposition)**
- Cell 19: Decompose persona vectors into features
- Cell 20: Analyze shared vs specific features
- Cell 21: Visualize (bar charts, Venn diagram)

### **Cell 22-24: EXTENSION PART 2 (Feature-Based Transfer)**
- Cell 22: Extract SAE features, train probes
- Cell 23: Generate comparison visualizations
- Cell 24: Statistical comparison, interpretation

### **Cell 25-27: EXTENSION PART 3 (Circuit Tracing)**
- Cell 25: Compute pullbacks across layers
- Cell 26: Build and visualize computational graph
- Cell 27: Interpret hierarchical feature composition

### **Cell 28-29: EXTENSION PART 4 (Mechanistic Explanation)**
- Cell 28: Correlation analysis (shared features vs transfer)
- Cell 29: Synthesis in markdown with final interpretation

### **Cell 30: Generate RESULTS_SUMMARY.md**
- Auto-generate comprehensive markdown summary
- Save all metadata (hyperparameters, model config)
- List all generated files with sizes

### **Cell 31: Verification & Cleanup**
- Verify all files created successfully
- Print Drive links for easy access
- Summary statistics (total runtime, GPU memory used)

---

## ERROR HANDLING & ROBUSTNESS

### Model Loading with Fallback
```python
# Pseudocode structure (Engineer implements actual code)
def load_model_with_fallback():
    models_to_try = [
        ("google/gemma-2-2b-it", torch.float16),
        ("Qwen/Qwen2.5-0.5B-Instruct", torch.float16),
    ]
    
    for model_name, dtype in models_to_try:
        try:
            # Attempt load
            # Print success
            return model, tokenizer, model_name
        except Exception as e:
            # Print failure, continue to next
            
    # If all fail, raise error with helpful message
```

### Activation Extraction with Error Recovery
- Try-except around each prompt processing
- On OOM: clear cache, retry with smaller batch
- Track failed prompts, report count at end
- Never crash entire notebook for single prompt failure

### File I/O with Verification
```python
# Pseudocode
def save_with_verification(data, filepath):
    try:
        # Save (numpy, pickle, or json as appropriate)
        # Verify file exists and has non-zero size
        # Print confirmation with file size
    except Exception as e:
        # Print error, but don't crash
        # Log to error list for final report
```

### Memory Management
After each major section:
```python
# Pseudocode
del large_tensors
gc.collect()
torch.cuda.empty_cache()

# Report memory status
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

---

## QUALITY ASSURANCE CHECKLIST

Before considering the notebook complete, the Research Engineer must verify:

### Code Quality
- [ ] All functions have complete docstrings (description, args, returns)
- [ ] All hyperparameters defined as named constants (no magic numbers)
- [ ] All random operations use seed=42
- [ ] Progress bars for all loops > 10 iterations
- [ ] Try-except around all external calls
- [ ] Memory cleanup after each experiment
- [ ] No unused imports or variables
- [ ] Code follows PEP 8 style where reasonable

### Scientific Rigor
- [ ] Train/test splits documented (where applicable)
- [ ] Pearson r and p-values reported for all correlations
- [ ] Sample sizes stated explicitly in results
- [ ] All results saved to disk (not just printed)
- [ ] Metadata tracked (model name, hyperparameters, timestamps)
- [ ] Statistical significance tests performed where appropriate

### Visualization
- [ ] All plots have descriptive titles
- [ ] All axes labeled with units
- [ ] Legends positioned appropriately
- [ ] Colorblind-friendly palettes used
- [ ] Annotations on data points/cells where helpful
- [ ] Both HTML (interactive) and PNG (static, ≥300 DPI) saved
- [ ] Consistent sizing and styling across all figures

### Reproducibility
- [ ] All dependencies with version numbers
- [ ] Random seeds set at notebook start
- [ ] Hyperparameters saved to JSON
- [ ] Output directory structure documented
- [ ] RESULTS_SUMMARY.md auto-generated
- [ ] Notebook runs start-to-finish without errors
- [ ] No hardcoded paths (use variables for Drive paths)

### Results Validation
- [ ] Results quantitatively consistent with Evans et al. expectations
- [ ] Expected patterns observed (accuracy drops, layer-wise trends, etc.)
- [ ] No unexplained anomalies (if any, documented and investigated)
- [ ] Statistical significance achieved (p < 0.05 minimum for key findings)
- [ ] Extension findings are interpretable and well-supported

---

## FINAL DELIVERABLE: RESULTS_SUMMARY.md

The notebook's final cell must auto-generate a comprehensive markdown summary saved to `/content/drive/MyDrive/persona_reps_replication/RESULTS_SUMMARY.md`

**Required Sections**:

### 1. Executive Summary
- One paragraph: what was replicated, what was extended, key findings
- Model used, runtime, platform

### 2. Core Replication Results

**Experiment 1: Cross-Context Probe Transfer**
- Transfer matrix (as table)
- Key finding with numbers
- Link to figure
- Interpretation: validates/contradicts Evans et al.

**Experiment 2: Steering Vector Effectiveness**
- Effectiveness by persona (as table or bullet points)
- Key finding with numbers
- Link to figure
- Interpretation

**Experiment 3: Question-Time Persona Detection**
- Best layer, accuracy, correlation
- Layer-wise trends
- Link to figure
- Interpretation

### 3. Extension Results

**Part 1: Persona Decomposition**
- Number of features per persona
- Shared vs specific feature counts
- Key interpretable features identified
- Links to figures

**Part 2: Feature-Based Transfer**
- Comparison table (raw vs transcoder features)
- Accuracy drop reduction percentage
- Statistical significance
- Links to figures

**Part 3: Circuit Tracing**
- Hierarchical composition pattern observed
- Key feature pathways identified
- Links to circuit graph
- Interpretation of emergent structure

**Part 4: Mechanistic Explanation**
- Correlation between feature sharing and transfer
- Mechanistic story in 2-3 sentences
- Link to correlation plot

### 4. Validation Against Original Paper

| Metric | Evans et al. (2025) | This Replication | Match? |
|--------|---------------------|------------------|--------|
| Probe transfer drop | ~35% | [actual]% | ✅/❌ |
| Question-time accuracy | 75-80% | [actual]% | ✅/❌ |
| Best detection layer | Mid-layers | Layer [X]/[Y] | ✅/❌ |
| Correlation strength | r > 0.75 | r = [actual] | ✅/❌ |

### 5. Novel Contributions

- Transcoder decomposition of persona vectors (first application to personas)
- Mechanistic explanation of cross-context transfer failure
- Circuit-level analysis of persona emergence
- Quantification: SAE features reduce context-dependence by [X]%

### 6. Limitations & Simplifications

**This is a DEMONSTRATION, not publication-ready research**:
- Sample size: 25-30 prompts (vs Evans' 100+)
- Steering: Proxy measurement (not full generation + LLM-judge)
- Model size: 2B parameters (vs Evans' 7-8B models)
- Transcoder coverage: Layer 13 analysis (could extend to all layers)

**What IS production-quality**:
- Activation extraction methodology
- Probe training and evaluation
- Statistical rigor (proper splits, p-values, correlations)
- Transcoder integration and circuit analysis
- Visualization standards

### 7. Next Steps (Proposed Extensions)

**For MARS Fellowship** (Rhea/Jacob):
- Track persona feature composition across training checkpoints
- Test if feature emergence patterns predict final model capabilities

**For Astra Fellowship** (Owain Evans):
- Extend circuit analysis to deceptive personas specifically
- Test whether feature-level steering has fewer side effects than vector-level
- Investigate which transcoder features are truly safety-relevant

**Novel angles not in Evans et al.**:
- Cross-model feature transfer (do Gemma transcoder features map to Qwen?)
- Minimal viable feature sets (can we reduce personas to 3-5 key features?)
- Feature composition rules (do personas combine algebraically?)

### 8. Files Generated

**Figures** (12 files):
- List all HTML and PNG files with descriptions

**Data** (12 files):
- List all .npy, .pkl, .json files with descriptions

**Metadata** (3 files):
- model_config.json
- hyperparameters.json
- test_prompts.txt

### 9. References

- Evans, O., et al. (2025). Persona Vectors: Monitoring and Controlling Character Traits in Language Models. Anthropic. arXiv:2507.21509
- Dunefsky, J., Chlenski, P., & Nanda, N. (2024). Transcoders enable fine-grained interpretable circuit analysis for language models. arXiv:2406.11944
- Gurnee, W., et al. (2024). Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2. arXiv:2408.05147

### 10. Acknowledgments

This replication demonstrates technical fluency with activation-based interpretability methods as preparation for fellowship research in persona-dependent representations and AI safety. Thanks to Owain Evans for the original work, Jacob Dunefsky et al. for transcoder methods, and Google for releasing Gemma Scope.

---

## CRITICAL INSTRUCTIONS TO RESEARCH ENGINEER

You are implementing a **complete, production-ready replication study** that will be reviewed by:
1. Fellowship selection committees (MARS, Astra)
2. Potentially, Owain Evans himself

**Your output must be**:
- ✅ Scientifically rigorous (proper statistics, full reproducibility)
- ✅ Visually professional (publication-quality plots matching Owain's aesthetic)
- ✅ Intellectually honest (clearly labeled as replication + novel extension)
- ✅ Technically flawless (runs start-to-finish, no crashes, all outputs generated)
- ✅ Mechanistically insightful (explains WHY, not just WHAT)

**This is not a prototype. This is a portfolio piece.**

Every line of code should reflect:
- **Clarity**: Others can read and understand without struggle
- **Rigor**: Proper methodology, no shortcuts on statistics
- **Craft**: Attention to detail, pride in execution
- **Insight**: Mechanistic understanding, not just pattern matching

**Success criterion**: If Owain Evans saw this notebook, he would think:

> "This person deeply understands my work, can execute it independently, and has extended it in a mechanistically insightful direction using cutting-edge interpretability methods. I want to talk to them about this."

---

**Build something worthy of that conversation.**

**Now go build it.**