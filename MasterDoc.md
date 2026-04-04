# **PROJECT: Resilience ROI Dashboard**

**Subtitle:** Climate Adaptation Intelligence for Sovereign Wealth Funds **Target:** ML@Purdue Catapult Hackathon (36-Hour Sprint)

## **1\. SYSTEM OVERVIEW**

* **Vision:** To become the Bloomberg Terminal for Planetary Adaptation Capital.  
* **Mission:** To build a decision-support engine that ingests coarse ISIMIP data, simulates systemic interventions via Graph Neural Networks (GNNs), and flags "Tail-Risk Escalations" (sudden ecosystem collapses). The system outputs natively searchable video heatmaps (via `sentrysearch`), allowing capital allocators to instantly retrieve projects that yield the highest "Economic Loss Avoided."  
* **Key Innovation:** We combine **Extreme-Regime Forecasting** (predicting sudden tipping points) with **Generative Downscaling** principles, making our simulations hyper-fast. We then use **Native Video Embeddings** to make complex physics searchable by financial analysts.

## **2\. ARCHITECTURE DESIGN**

**End-to-End System Flow:**

\[Coarse ISIMIP3b Data\] \-\> (Temperature, Precipitation)

         |

         v

\[Tail-Risk Predictor\] \---\> (Calculates volatility/momentum to flag 95th-percentile tipping points)

         |

         v

\[Base GNN Model\] \--------\> (Learns systemic dependencies & baseline risk)

         |

         v

\[Simulation Engine\] \<----- (Injects Intervention "Deltas")

         |                 (Uses Generative/Interpolation concepts to downscale to high-res MP4s)

         v

\[SentrySearch Pipeline\] \-\> (chunker.py \-\> Gemini/Qwen3-VL embeddings \-\> ChromaDB)

         |

         v

\[Streamlit Dashboard\] \---\> (Plays exact video clip matching user text query \+ ROI Metrics)

## **3\. CORE MODULES**

1. **Data Processing & Uncertainty Modeling:** Ingests coarse ISIMIP data. Applies an "Uncertainty Penalty" to precipitation-heavy regions to correct for known global-model biases (Ito et al., 2020).  
2. **Tail-Risk Escalation Engine:** Analyzes the time-series data of each graph node. Calculates short-term momentum and volatility. If risk exceeds the 95th percentile, the node is flagged for "Extreme Regime Escalation" (Gurjar & Camp, 2026).  
3. **Graph Neural Network (The Propagator):** Calculates how stabilization from an intervention (e.g., mangroves) ripples across trade routes and adjacent geographic nodes.  
4. **Generative Downscaling Renderer:** Converts the low-res GNN output into high-resolution spatial heatmaps (MP4s) simulating the visual fidelity of diffusion models (Hess et al., 2023).  
5. **Resilience ROI Engine:** Calculates the ratio of cost to systemic loss avoided.  
6. **SentrySearch Video Embedder:** Splits simulation MP4s into chunks and projects the raw video into a 768d vector space using Gemini 1.5 Pro.

## **4\. DATA STRATEGY**

* **Primary:** **ISIMIP3b** (Water & Agriculture). To overcome the NetCDF latency trap in a 36-hour hackathon, extract only the coarse grids (e.g., 2°x2°) and rely on the Generative Downscaling concept for visual output.  
* **Merge Tactic:** Flatten the grid into a Graph. Node features \= `[Temp, Precip, Volatility, Momentum, GDP]`.

## **5\. SIMULATION STRATEGY & TAIL RISK**

Use **Perturbation Modeling**.

* **The Interventions:** 1\. *Coastal Mangrove Restoration ($1B)* 2\. *Regenerative Agriculture ($1B)*  
* **The "Tail-Risk" Trigger:** Interventions are automatically prioritized by the system if they neutralize a node exhibiting high *volatility* (indicating an impending phase shift/collapse).

## **6\. RESILIENCE ROI METRIC (With Uncertainty)**

$$ROI\_{resilience} \= \\left( \\frac{\\sum (L\_{baseline} \- L\_{intervention}) \\times \\gamma^t}{Cost\_{intervention}} \\right) \\pm U\_{precip}$$

*Where* $U\_{precip}$ *is the confidence interval penalty derived from ISIMIP precipitation uncertainty bounds.*

## **7\. SEMANTIC SEARCH DESIGN (VIA SENTRYSEARCH)**

* **Visual Rendering:** Simulations rendered as MP4s.  
* **Chunking:** `sentrysearch/chunker.py` downscales and drops static frames.  
* **Native Embedding:** `sentrysearch/gemini_embedder.py` maps visual meaning without text translation.  
* **Retrieval:** Text query $\\rightarrow$ Vector Match $\\rightarrow$ Trimmer extracts exact timestamp $\\rightarrow$ Displayed in Streamlit.

## **8\. TECH STACK**

* **Data Processing / ML:** `xarray`, `PyTorch Geometric (PyG)`, `pandas` (for rolling volatility calculations).  
* **Video Rendering:** `matplotlib.animation`, `ffmpeg-python`.  
* **Semantic Layer:** `sentrysearch` repo, `google-genai`, `chromadb`.  
* **Frontend:** `Streamlit`.

## **9\. TEAM DIVISION**

**Hacker 1: Data & Uncertainty (The Scientist)**

* Extract ISIMIP data. Implement the momentum/volatility mathematical formulas for the Tail-Risk Predictor.

**Hacker 2: ML & Simulation (The Architect)**

* Build the GNN. Apply intervention perturbations. Render the output into MP4 video heatmaps using interpolation (Generative Downscaling).

**Hacker 3: SentrySearch & Product (The Product Lead)**

* Index Hacker 2's MP4s into ChromaDB using the `sentrysearch` codebase. Build the Streamlit UI.

## **10\. HACKATHON EXECUTION PLAN (36 HOURS)**

* **Hour 0-4:** Data subsetting; environment setup; verify `sentrysearch` with dummy video.  
* **Hour 4-12:** Build Tail-Risk math formulas; establish basic GNN.  
* **Hour 12-20:** Generate 100 simulation MP4s and export ROI CSVs.  
* **Hour 20-28:** Index MP4s via Gemini/Qwen using `sentrysearch`.  
* **Hour 28-34:** Wire Streamlit UI. Connect search bar to ChromaDB retrieval.  
* **Hour 34-36:** Pitch Polish.

## **11\. KEY RESEARCH CITATIONS (FOR THE JUDGES)**

* **Generative Downscaling:** *Hess et al. (2023). "Deep Learning for Bias-Correcting CMIP6-Class Earth System Models." Earth's Future.* (Justifies our fast rendering/downscaling approach).  
* **Ensemble Uncertainty:** *Ito et al. (2020). "Uncertainties in climate change projections covered by the ISIMIP... subsets." GMD.* (Justifies our $U\_{precip}$ confidence intervals).  
* **Extreme-Regime Tipping Points:** *Gurjar & Camp (2026). "Predicting Tail-Risk Escalation in IDS Alert Time Series."* (Justifies using volatility/momentum to predict sudden climate collapse).  
* **Native Video Search:** *`sentrysearch` Open Source Pipeline.* (Justifies bypassing RAG for complex scientific data).