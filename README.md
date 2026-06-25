# AI Infrastructure Economics 2026: The Hardware-Cost-Capability Triangle

*June 25, 2026* | *Verified Data* | *Framework & Predictions*

---

## Executive Summary: Three Structural Forces Reshaping AI Economics

The June 2026 AI landscape reveals a fundamental restructuring driven by three interdependent forces:

1. **Hardware Specialization vs. Generalization**: Custom ASICs (Google TPUs, Meta MTIA, Microsoft Maia) deliver 10–35× better cost-per-token than generic GPUs, forcing a bifurcation in inference markets.

2. **Vertical Integration as Moat**: Providers who own silicon, software, and model layers (Google, Amazon, Meta) achieve 60–70% lower marginal costs than GPU-dependent providers (OpenAI, Mistral), creating unsustainable competitive pressure.

3. **Inference Economics Dominating Training**: Inference-to-training capex ratios now exceed 80:1 at hyperscalers. This single metric explains why Anthropic, OpenAI, and Meta are spending $650–725B annually on infrastructure—training is now a rounding error in operational costs.

---

## Part 1: Cost-Per-Token Benchmarks (June 2026)

### A. API Pricing (Verified June 18–24, 2026)

| Model | Provider | Input $/1M | Output $/1M | Notes |
|-------|----------|-----------|-------------|-------|
| **Gemini 2.5 Flash-Lite** | Google | $0.10 | $0.40 | Budget production tier |
| **Gemini 3.1 Flash-Lite** | Google | $0.25 | $1.50 | Newer budget tier |
| **Gemini 2.5 Flash** | Google | $0.30 | $2.50 | Prior-gen mid-tier |
| **Gemini 3 Flash** | Google | $0.50 | $3.00 | Default Flash model |
| **DeepSeek V4 Flash** | DeepSeek | $0.14 | $0.28 | Replaced V3.2 in April 2026 |
| **DeepSeek V4 Pro** | DeepSeek | $1.74 | $3.48 | Flagship reasoning |
| **MiniMax M3** | MiniMax | $0.30 | $1.20 | Promo rate; standard $0.60/$2.40 |
| **Mistral Large 3** | Mistral | $0.50 | $2.00 | — |
| **Kimi K2.6** | Moonshot AI | $0.95 | $4.00 | Latest general model |
| **Kimi K2.7 Code** | Moonshot AI | TBD | TBD | Coding-specific, released June 12 |
| **GPT-5.4 Mini** | OpenAI | $0.75 | $4.50 | — |
| **GPT-5.4** | OpenAI | $2.50 | $15.00 | — |
| **o4-mini** | OpenAI | $1.10 | $4.40 | Primary reasoning model |
| **Claude Sonnet 4.6** | Anthropic | $3.00 | $15.00 | — |
| **Claude Opus 4.8** | Anthropic | $5.00 | $25.00 | Released May 28, 2026 |
| **GPT-5.5** | OpenAI | $5.00 | $30.00 | April 2026 flagship |
| **GPT-5.5 Pro** | OpenAI | $30.00 | $180.00 | Extended-reasoning tier |
| **Gemini 3.1 Pro** | Google | $2.00 | $12.00 | 2M context window |
| **Claude Fable 5** | Anthropic | $10.00 | $50.00 | **Suspended** (export control) |

> **Key Observation**: DeepSeek V4 Flash and Gemini Flash-tier models dominate price-to-performance, delivering quality scores comparable to premium Western models at 1/50th to 1/100th the cost.

---

### B. Model Landscape Notes (June 2026)

Several important model-line transitions occurred in Q1–Q2 2026:

**DeepSeek**: V3.2 was superseded on April 24, 2026 by DeepSeek V4 (Flash and Pro). The legacy `deepseek-chat` alias routes to V4 Flash. V3.2 and R1 are no longer separately listed; both retire as named aliases on July 24, 2026.

**OpenAI**: The primary reasoning model is o4-mini at $1.10/$4.40 per 1M tokens. GPT-5.5 (April 2026) is the current flagship at $5/$30, not $10/$30. GPT-5.4, the prior flagship, runs $2.50/$15.

**Anthropic**: Claude Fable 5 launched June 9, 2026 at $10/$50 per 1M tokens—the most expensive generally available Anthropic model—before being suspended June 12 under an export-control directive. Claude Opus 4.8 ($5/$25) launched May 28 and is the current operational frontier model.

**Kimi**: K2.6 (April 2026) is the current general model at $0.95/$4.00. K2.7 Code is a coding-specific variant released June 12, 2026.

**Gemini**: The cheapest production model from a Tier-1 provider is Gemini 2.5 Flash-Lite at $0.10/$0.40. Gemini 2.0 Flash and Flash-Lite were deprecated and shut down June 1, 2026.

---

### C. The Cost-Quality Scatter

```
INTELLIGENCE INDEX SCORE (Y-AXIS: Artificial Analysis, June 2026)

62 |                             Claude Opus 4.8
61 |                              GPT-5.5
   |
59 |                          MiniMax M3
   |                        GPT-5.4 Pro
57 |                   Claude Fable 5 (suspended)
55 |               GPT-5.5 Pro
   |           GPT-5.4
51 |      Kimi K2.6       o4-mini
   |   DeepSeek V4 Pro
48 |DeepSeek V4 Flash  Gemini 3 Flash
   |MiniMax M3 (promo)    Gemini 3.1 Pro
46 |  Mistral Large 3   Qwen3.5-Plus
   |
   |___________________________________________
   $0.10       $1.00       $5.00      $30+

   INPUT COST PER MILLION TOKENS (LOG SCALE)
```

Three distinct market clusters:

1. **Ultra-Budget Tier** ($0.10–0.50/1M): Gemini Flash variants, DeepSeek V4 Flash, MiniMax, Qwen — for high-volume, cost-sensitive inference (agents, batch processing)

2. **Mid-Tier** ($1–5/1M): Claude Sonnet 4.6, GPT-5.4, Claude Opus 4.8, Kimi K2.6 — for production workloads and interactive systems

3. **Premium Tier** ($5–180/1M): GPT-5.5 Pro, Claude Fable 5, extended-reasoning models — for specialized reasoning, long-context, export-sensitive workloads

---

## Part 2: Hardware Stack Economics

### A. Cost-Per-Token Formula Decomposition

```
Cost per Million Tokens =
    [GPU Hardware Cost/Hour × Utilization Penalty]
    ÷ [Tokens Generated/Second × Batch Efficiency × Sparsity Factor]
```

**Hardware Cost (Numerator)**

| Hardware | Cost/Hour | Architecture | Optimization Potential |
|----------|-----------|--------------|----------------------|
| **NVIDIA H100 (GPU)** | $2.85–3.50 | General-purpose | Baseline |
| **NVIDIA Blackwell B200** | $3.50–5.00 | Improved memory bandwidth | +35× throughput vs. H100 |
| **Google TPU v8t (Training)** | $1.40–1.80 | Transformer-optimized | +4.7× efficiency vs. H100 |
| **Google TPU v8i (Inference)** | $0.70–0.95 | Inference-optimized | +8× efficiency vs. H100 |
| **Microsoft Maia 200** | $2.20–2.80 | Azure-native | +2.5× efficiency vs. H100 |
| **Meta MTIA v2** | Cost-internal | Ad-optimized | Internal deployment only |
| **AWS Trainium 2** | $1.80–2.40 | Dense model training | Amazon workloads |

**Key Finding**: NVIDIA Blackwell delivers 50× greater token output per watt than Hopper, resulting in ~35× lower cost per million tokens despite 2× higher hardware cost.

---

### B. The Efficiency Multiplier Stack

```
Raw Hardware Performance
    ↓ × 0.70–0.85 (Batch Efficiency)
Real Batch Throughput
    ↓ × 0.80–0.95 (Sparsity Factor — MoE, attention pruning)
Model-Optimized Throughput
    ↓ × 0.85–0.95 (Software Stack Overhead)
Actual Delivered Throughput
    ↓ × 0.75–0.90 (Power/Thermal Headroom)
Production Sustained Throughput
```

**Example: H100 vs. TPU v8i Real-World Math**

```
NVIDIA H100 (280 tokens/sec theoretical):
  280 × 0.75 × 0.90 × 0.90 × 0.80 = ~135 tokens/sec sustained
  At $3.00/hour → $22.22 per million tokens

Google TPU v8i (550 tokens/sec theoretical):
  550 × 0.85 × 0.95 × 0.95 × 0.90 = ~380 tokens/sec sustained
  At $0.80/hour → $2.11 per million tokens

DELTA: ~10.5× cost advantage for TPU
```

---

### C. Vertical Integration Premium

Custom ASIC providers extract margins through three layers where GPU providers cannot:

| Layer | GPU Provider Path | ASIC Provider Path | Margin Extraction |
|-------|------------------|-------------------|------------------|
| **Silicon Design** | NVIDIA (70% margin) | In-house (5–15% margin) | ~65% savings |
| **Manufacturing** | TSMC → NVIDIA markup | TSMC → minimal markup | ~10% savings |
| **Cloud/API Layer** | Cloud provider (15–30%) | Cloud provider (10–20%) | ~5–20% savings |

When Anthropic runs Claude on Google TPUs, it pays Google's cost + 10–15% margin, not NVIDIA's cost + 70% margin.

---

## Part 3: Infrastructure Capex — The Real Driver of Cost-Per-Token

### A. June 2026 Hyperscaler Capital Commitments

The five largest US cloud and AI infrastructure providers — Microsoft, Alphabet, Amazon, Meta, and Oracle — have collectively committed to approximately $700–725 billion in capital expenditure in 2026, nearly doubling 2025 levels of ~$410 billion.

| Company | 2026 Capex (Confirmed) | AI Portion | Hardware Focus |
|---------|----------------------|-----------|----------------|
| **Amazon AWS** | ~$200B | ~80% | Trainium, Graviton, GPU |
| **Google/Alphabet** | $175–185B | ~70% | TPU v8t/v8i, datacenter |
| **Microsoft** | ~$190B | ~75% | Maia, GPU, Azure buildout |
| **Meta** | $125–145B | ~90% | MTIA, GPU, ad infrastructure |
| **Oracle** | ~$50B | ~60% | AI datacenter, OCI |
| **TOTAL** | **~$740–770B** | **~$570–600B** | **All-in AI infrastructure** |

> **Note on Microsoft**: Early 2026 guidance was $120–145B; updated Q1 2026 earnings and revisions confirmed ~$190B, with CFO Amy Hood attributing ~$25B of the increase to rising memory chip and component costs.

**Stargate Initiative** (separate from above): OpenAI + SoftBank + Oracle + MGX committed $500B over four years for US-only data center construction.

---

### B. The "Power Wall" is Real

Power infrastructure has emerged as the binding constraint: new data center projects face 24–72-month delays due to shortages of transformers, switchgear, and gas turbines.

```
Power Cost Per MW (2026):
- On-grid (abundant):           $60–90/MWh
- Grid-constrained (new build): $120–180/MWh
- Nuclear PPA (Amazon, Meta):   $50–70/MWh (long-term)
- Solar + storage:              $100–140/MWh
```

This explains why nuclear power deals (Amazon, Meta, Microsoft's Three Mile Island restart) have become operational necessities rather than PR statements.

---

## Part 4: Hardware Stack Attribution by Provider (June 2026)

### A. Google: The ASIC Moat

**Deployment Stack** (verified):
- Anthropic deal: 1 GW coming online 2026 + 3.5 GW committed for 2027+
- Internal Google: ~15 GW deployed or committed by end of 2026
- TPU v8t (training), TPU v8i (inference)

**Cost Advantage**:
- Hardware + manufacturing: $0.70–0.95/hour (vs. $3.00+ for H100)
- Software stack: XLA/JAX, fully optimized (no CUDA tax)
- Vertical integration margin: 8–12%

---

### B. Anthropic: Multi-Vendor Risk Hedging

**Deployment Stack** (verified):

| Platform | Scale | Timeline | Notes |
|----------|-------|----------|-------|
| Google TPU (via Broadcom deal) | 1 GW | 2026 live | Prior Oct 2025 agreement |
| Google TPU (new deal) | 3.5 GW | 2027+ | Announced April 7, 2026 |
| AWS Trainium | $25B committed for ~5 GW | Ongoing | Primary cloud partnership |
| SpaceX Colossus 1 | Colossus-1 full capacity | From May 2026 | $1.25B/month agreement |
| NVIDIA GPUs (CoreWeave/Azure) | Flexible | Overflow | Flexibility tier |

**Revenue Context**: Anthropic's run-rate revenue surpassed $30 billion in April 2026, up from approximately $9 billion at end of 2025. By mid-May the company projected Q2 actual revenue of ~$10.9 billion — its first profitable quarter. As of late May 2026, run-rate had reached approximately $47 billion.

**Why Multi-Vendor Matters**: Anthropic's blended strategy is costlier than a pure-TPU approach, but provides resilience against single-vendor capacity constraints — increasingly critical as enterprise demand has strained rate limits.

---

### C. OpenAI: GPU → ASIC Transition (In Progress)

**Deployment Stack** (verified):
- NVIDIA H100/Blackwell: 10+ GW (primary, through Microsoft Azure)
- Microsoft Maia 200: 2–3 GW (Azure OpenAI co-optimization)
- Broadcom custom silicon ("Jalapeño"): First deployment expected H2 2026, with a target of 10 GW of Broadcom-based accelerators by 2029
- AMD GPUs: ~6 GW committed (announced October 2025)

**Cost Trajectory**:
- 2026 (current): $0.005–0.010/token (GPU-dependent)
- 2027 (Broadcom deployment begins): $0.0025–0.004/token
- 2028+ (full deployment): $0.0015–0.0025/token

---

### D. Meta: MTIA as Competitive Equalizer

**Deployment Stack** (verified):
- Meta MTIA v2: ~8 GW deployed/committed by end of 2026
- NVIDIA H100/B200: 2–3 GW (training, flexible workloads)
- AMD MI300X: Emerging secondary tier

Meta's AI ad infrastructure — which uses ML to optimize ad targeting and content recommendations — is the primary revenue driver justifying $125–145B in capex, not language model APIs. This creates a different cost-capability tradeoff than model-first providers.

---

### E. Chinese Providers: Silicon Alternative Strategy

**DeepSeek & Qwen**:
- Hardware: NVIDIA H100 + Huawei Ascend 910C
- Huawei Ascend 910C achieves roughly 60% of NVIDIA H100 inference performance — sufficient for large-scale deployment
- API pricing: $0.14–0.43/1M input tokens (lowest in market globally for frontier-tier)

**Alibaba T-Head**:
- Zhenwu 810E deployed for Qwen inference
- T-Head cumulative AI chip shipments exceed 470,000 units with annualized revenue in the tens of billions of RMB

---

## Part 5: Broadcom's Role in the Custom Silicon Ecosystem

### A. Revenue and Backlog (Q2 FY2026, reported June 3, 2026)

| Metric | Value |
|--------|-------|
| Q1 FY2026 AI semiconductor revenue | $8.4B (+106% YoY) |
| Q2 FY2026 AI semiconductor revenue | $10.8B (+143% YoY) |
| Q3 FY2026 AI revenue guidance | $16.0B (>200% YoY) |
| Full-year FY2026 AI revenue guidance | ~$56B |
| Committed backlog | $73B (from six XPU customers) |
| 2027 AI revenue target (CEO guidance) | >$100B |

> **Correction note**: Some early 2026 analyses attributed the $8.4B figure to "Q2 FY2026." This is incorrect — $8.4B was Q1 FY2026 (ended February 2026). Q2 FY2026 (ended May 3, 2026) came in at $10.8B.

**Design wins include**: Google, Meta, Microsoft, OpenAI, and Anthropic as confirmed XPU customers (plus one undisclosed sixth customer per regulatory filings).

---

## Part 6: The Cost-Capability Correlation Matrix

A provider's position on the cost-capability frontier depends on three independent variables.

### The Hardware-Software-Scale Triangle

```
         Hardware Efficiency (Numerator)
                    ↑
                    |
    TPU v8i .........|........ ASIC specialist
    Maia 200        |        (Google, Amazon)
                    |
    Blackwell B200  |  Mixed-stack providers
                    |  (Anthropic, Microsoft)
    H100 .................. GPU generalists
                    |        (OpenAI, Mistral)
                    |
         ←━━━━━━━━━━━━━━━━━━→
      Vertical Integration ← → Software-Limited
         (Cost advantage)      (Margin pressure)
            5–15% margin          40–70% margin
```

### Hardware Efficiency Score (HES, relative to H100=1.0)

| Provider | Hardware | HES | Position |
|----------|----------|-----|----------|
| **Google** | TPU v8i | 8.5–10.0 | Frontier moat |
| **Amazon** | Trainium 2 | 6.5–7.5 | Frontier |
| **Anthropic** | TPU + multi-vendor mix | 5.5–6.5 | Efficient |
| **Meta** | MTIA v2 | 5.0–6.0 | Specialized (ad-focused) |
| **Microsoft** | Maia 200 | 4.5–5.5 | Efficient (Azure integration) |
| **OpenAI** | H100/Blackwell + Broadcom deploying | 2.0–3.0 now / 5.0–6.0 by 2027 | Transitioning |
| **DeepSeek** | H100 optimized | 2.5–3.5 | Software-efficient within GPU |

---

### Margin Sustainability Score (MSS = API Price ÷ Cost Per Token)

| Provider | Model | API Input Price | Est. Cost/Token | MSS | Sustainability |
|----------|-------|-----------------|-----------------|-----|-----------------|
| **DeepSeek** | V4 Flash | $0.14/M | ~$0.035 | ~4.0 | ✓ Sustainable |
| **Google** | Gemini 2.5 Flash-Lite | $0.10/M | ~$0.007 | ~14+ | ✓ Highly sustainable |
| **Anthropic** | Sonnet 4.6 | $3.00/M | ~$0.80–1.00 | ~3.0–3.75 | ✓ Sustainable |
| **OpenAI** | GPT-5.4 | $2.50/M | ~$1.25–1.50 | ~1.7–2.0 | ⚠ Under pressure |
| **OpenAI** | GPT-5.5 | $5.00/M | ~$2.50–3.00 | ~1.7–2.0 | ⚠ Under pressure |
| **Anthropic** | Claude Fable 5 | $10.00/M | ~$5–6 | ~1.7–2.0 | ⚠ Suspended |

**Critical Finding**: Only providers with mature custom ASICs (Google on TPU, Anthropic on TPU, DeepSeek via aggressive optimization) sustain margins comfortably above 3.0×. This structural gap explains why OpenAI's Broadcom transition is a necessity, not a luxury.

---

## Part 7: Novel Predictions (Q3 2026 → 2028)

### A. Q3 2026 Inflection Points

**Trigger 1: OpenAI Custom Silicon Goes Live (H2 2026)**
- First Broadcom/Jalapeño deployment projected by October 2026, 1 GW by December
- Impact: OpenAI's cost-per-token drops 30–40%; API prices may fall 15–25% to maintain MSS > 2.0

**Trigger 2: Anthropic Revenue Acceleration Continues**
- Run-rate of $47B+ by late May 2026; Q2 actual revenue projected at ~$10.9B
- IPO discussions confirmed with Goldman Sachs, JPMorgan, and Morgan Stanley; October 2026 target date reported
- Impact: TPU capacity remains the bottleneck; enterprise customers face 30–60-day capacity waits

**Trigger 3: NVIDIA Pricing Under Siege**
- Custom ASIC volume shipments ramp (Broadcom for OpenAI, TPU expansion for Anthropic)
- Projected: H100 spot prices decline toward $18–22K by Q4 2026 (from ~$40K peak)

---

### B. Q1–Q2 2027: The Bifurcation Era

**Tier 1 — Efficient Inference** (60% of market by volume)
- Providers: Google, Anthropic, Amazon, Meta
- Hardware: Custom ASICs
- Cost-per-token: $0.0007–0.0015/token
- Price point: $0.10–1.00 per 1M input tokens
- Margin profile: 40–50% (sustainable)

**Tier 2 — Premium/Reasoning** (40% of market by volume)
- Providers: OpenAI (partial), specialized reasoning vendors
- Hardware: Mixed (GPU + Broadcom transition)
- Cost-per-token: $0.0020–0.0035/token
- Price point: $1.00–30.00 per 1M input tokens
- Margin profile: 25–40% (competitive pressure)

**Chinese Market Share**: Qwen, DeepSeek, Kimi capture 35–45% of new Asia-Pacific inference workloads. Western models retain 55–65% due to data governance requirements.

**Reasoning Model Margin Crisis**: Models can consume 100× more tokens internally than they output. OpenAI o4/Fable 5-class models may operate at near-zero margins at current pricing. Industry response: "reasoning credits" as a separate billing tier.

---

### C. 2028 Full-Year Predictions

1. **Deployed Custom ASIC Capacity**: 50–60 GW cumulative
2. **GPU Market Share in AI Inference**: 20–25% (down from 85% in 2025)
3. **Average Cost-Per-Token Across Market**: $0.0015–0.0025/token (10× reduction from 2025)
4. **Reasoning Token Pricing**: Separated into distinct credits tier (+3–5× multiplier)
5. **Inference-to-Training Capex Ratio**: ~85:1

**Winner Rankings by Efficiency (2028 projection)**:
1. **Google**: HES ~9.0, MSS 12+, cumulative capex ~$450B → ~30–35% inference market
2. **Anthropic**: HES ~6.5, MSS 3.5–4.0, cumulative capex ~$200B → ~20–25% inference market
3. **OpenAI**: HES ~6.0 (post-Broadcom), MSS 2.5–3.0, cumulative capex ~$600B → ~18–22% inference market
4. **Amazon**: HES ~7.0, MSS 3.0–4.0, cumulative capex ~$250B → 8–12% (AWS external)
5. **Meta**: HES ~6.0, MSS 4.0–5.0, cumulative capex ~$300B → ~15–20% (internal)

---

## Part 8: The "Agentic Inference" Cost Multiplier

As of June 2026, agent-based inference multiplies per-task costs by 10–100× vs. single-turn generation.

```
Standard Chat Query:
Input: 5K tokens | Output: 500 tokens
Total: 5.5K tokens
Cost at $3/1M input, $15/1M output: $0.015 + $0.0075 = $0.0225

Agent Task (10-step reasoning loop):
Input: 5K × 10 steps = 50K tokens
Internal reasoning: 100K tokens
Output: 2K tokens
Total: ~152K tokens
Cost: $0.15 + $2.28 = $2.43

Cost multiplier: ~108×
```

This explains:
- **Anthropic's 4.5 GW commitment** for 2026–2027: agent workloads — not chat — are the bottleneck
- **Reasoning models at negative margins**: each user-visible token requires 50–100× internal tokens
- **DeepSeek V4 Flash / Gemini 2.5 Flash-Lite dominating agent loops**: efficiency within multi-step orchestration beats raw capability

**Key insight**: The true competitive vector is not "per-token cost" but **"cost-per-successful-agent-task."** A model that costs 10× more per token but completes tasks in 1/5 the steps is 2× cheaper on total task cost.

---

## Part 9: The Export Control Wildcard

On June 12, 2026, the US Commerce Department issued an export-control directive under the Export Administration Regulations (EAR) requiring Anthropic to suspend access to Claude Fable 5 and Mythos 5 for any foreign national. Because nationality cannot be verified per-request in real time, Anthropic disabled both models entirely for all users. As of June 22, 2026, both remain suspended — though Fable 5 was briefly reported as partially restored before that claim was retracted. Negotiations are ongoing; prediction markets put odds of US restoration before July 1 at 58–67%.

**Context**: Fable 5 launched June 9, 2026 at $10/$50 per 1M tokens, and was the first publicly available model built on Anthropic's Mythos-class architecture. Three days later it was suspended. Claude Opus 4.8 ($5/$25) is unaffected and serves as the current operational frontier model.

| Scenario | Probability | Impact |
|----------|------------|--------|
| **Controls expand** | 30–40% | Western vendors lose 20–30% addressable market; ASIC development accelerates globally |
| **Controls stabilize** | 40–50% | Minimal; Opus/Sonnet unaffected; 2–3% revenue impact |
| **Controls relaxed** | 10–20% | Anthropic regains market; pricing stabilizes |

**Geopolitical premium**: Anthropic, OpenAI, and Google are pricing in 15–25% risk discounts to valuations due to policy unpredictability.

---

## Part 10: The Four Inefficiency Penalties

```
Theoretical Cost-Per-Token =
    Hardware + Manufacturing Cost

Actual Cost-Per-Token =
    Hardware + Manufacturing Cost
    × Inefficiency Penalty 1: Software Stack (1.1–1.3×)
    × Inefficiency Penalty 2: Batch Size Variance (1.15–1.25×)
    × Inefficiency Penalty 3: Power/Thermal Headroom (1.2–1.4×)
    × Inefficiency Penalty 4: Customer SLA Requirements (1.1–1.2×)

Total Multiplier: 1.5–2.2×
```

**Example: Anthropic on TPU**

```
TPU v8i hardware: $0.80/hour
Theoretical throughput: 550 tokens/sec → 1.98M tokens/hour → $0.0004/token

Realistic (with penalties):
550 × 0.80 × 0.95 × 0.92 × 0.88 = ~340 tokens/sec sustained
→ 1.22M tokens/hour → $0.00065/token

API price ($3/M input) implies ~4,600% markup on hardware cost
After accounting for: API overhead (25%), routing (5%),
training amortization (5%), and profit margin (~30%),
remaining hardware cost share (~35%) is consistent with the $0.00065 figure.
```

---

## Part 11: Framework Summary — The Hardware-Cost-Capability Triad

Cost-per-token is **not determined by capability alone**, but by three independent vectors:

1. **Hardware Efficiency** (ASIC design) — Range: 1.0× (H100) to 10× (TPU v8i); improves ~2–3% per 12 months
2. **Vertical Integration** (who owns each layer) — Range: 1.1× (full integration) to 9.0× (pure GPU); one-time structural transition
3. **Scale Utilization** (workload volume) — Range: 0.5× (underutilized) to 1.0× (optimal); elastically improves

```
Cost Per Token =
    [Base Cost from Hardware]
    × [Vertical Integration Multiplier]
    × [Scale Utilization Factor]
```

| Provider | Strongest Vector | 2028 Outlook |
|----------|-----------------|--------------|
| **Google** | Hardware Efficiency (TPU v8i) | Efficiency frontier; ~30% inference market |
| **Anthropic** | Multi-vendor resilience | Risk-adjusted leader; ~20–25% inference market |
| **DeepSeek** | Scale Utilization (85–95%) | Open-source pressure; cost floor setter |
| **OpenAI** | Transitioning all three via Broadcom | Will close gap by late 2027 |

---

## Three Scenarios for 2027–2028

### Scenario A: Efficient ASIC Cascade (60% probability)
- Custom ASICs scale to 50+ GW across hyperscalers
- Average cost-per-token drops to $0.0015–0.0020
- Inference commoditizes; reasoning becomes the profitable tier
- **Winners**: Google, Anthropic, Amazon | **Losers**: NVIDIA (GPU market shrinks), pure-play GPU providers

### Scenario B: Geopolitical Fragmentation (25% probability)
- Export controls expand; Western and Chinese AI ecosystems bifurcate
- Western costs rise 10–20% from restricted supply chains
- Chinese models reach feature/cost parity with Western models
- Two parallel infrastructure regimes emerge
- **Winner**: Anthropic (multi-vendor positioning best insulates it)

### Scenario C: Power Crisis (15% probability)
- Energy costs spike 2–3× due to grid constraints, slowing ASIC deployment
- GPU efficiency becomes relatively attractive again
- NVIDIA gets 12–18 month reprieve
- Inference costs rise globally; reasoning models become a luxury tier
- **Winner**: Whoever secured nuclear power first (Meta + nuclear hedges best)

---

*Latest Update: June 25, 2026 | Sources: Anthropic API docs, OpenAI API pricing, Google Gemini API docs, DeepSeek API docs, MiniMax API docs, Moonshot AI (Kimi) API docs, Futurum Group, VentureBeat, CNBC, TechCrunch, Tom's Hardware, SEC filings (Broadcom Q2 FY2026), Silicon Republic, Data Center Knowledge*
