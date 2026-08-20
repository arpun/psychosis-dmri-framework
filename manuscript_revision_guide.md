# Manuscript Revision Guide & Reviewer Response

This document provides a comprehensive guide for revising the manuscript *"Diffusion Modeling for Characterizing Neuropathology in Psychotic Disorders: A Machine Learning Study"* in response to the reviewer's feedback regarding:
1. **Incremental Value of Model Integration (Ablation Analysis)**
2. **Clinical Decision Utility vs. Biological Separability**

---

## 1. Point-by-Point Reviewer Response Letter

### Comment 1: Incremental Value of the Integrated Framework & Component Ablation
> **Reviewer:** *"The study does not directly compare the full integrated framework with its constituent models or with reduced combinations of those models... Establishing the incremental benefit of integration is not peripheral to the study; it is central to the rationale for introducing the integrated framework... The manuscript requires a focused ablation or component analysis capable of determining whether the full integrated framework provides incremental predictive information beyond its constituent components... Importantly, if a simpler subset performs equivalently to the complete framework, this would not constitute a failed experiment; it would be an important scientific result requiring reinterpretation of the claimed advantage of integration."*

**Response:**
We thank the reviewer for highlighting this critical point. Rather than deferring this fundamental question to future work, we have now conducted a systematic component ablation analysis across both classification endpoints (**Patient vs Control** and **Schizophrenia Spectrum vs Non-Schizophrenia Psychosis**).

Using the identical site-stratified 5-fold cross-validation framework with strict within-fold feature selection ($K=20$), identical neural network architectures, Random Forest baselines, and 2,000-fold bootstrap confidence intervals, we compared seven standardized model configurations:
1. **DTI-only** ($FA, MD$ — 137 features)
2. **DKI-only** ($MK, KFA$ — 138 features)
3. **DTI + DKI** ($FA, MD, MK, KFA$ — 275 features)
4. **IVIM-only** ($PF$ — 69 features)
5. **FWI-only** ($FW$ — 69 features)
6. **IVIM + FWI** ($PF, FW$ — 138 features)
7. **Full Integrated Framework** ($FA, MD, MK, KFA, PF, FW$ — 413 features)

#### Key Findings:
- **Patient vs Control Classification (Table 5, Figure 17):**
  - The dominant discriminatory signal is driven by microstructural kurtosis and diffusivity metrics: **DKI-only** and **DTI-only** achieve performance equivalent to the **Full Integrated Model** ($\Delta\text{AUC} \approx 0.000$, paired $t$-test $p > 0.35$).
  - In contrast, fluid and perfusion compartments alone (**IVIM-only**: $\text{AUC} = 0.742\ [0.692-0.794]$; **FWI-only**: $\text{AUC} = 0.695\ [0.646-0.750]$; **IVIM+FWI**: $\text{AUC} = 0.744\ [0.693-0.793]$) yield significantly lower classification performance ($\Delta\text{AUC} = -0.258$ to $-0.305$, $p < 0.001$).
- **SCZ vs Non-SCZ Subtype Classification (Table 6, Figure 18):**
  - A similar pattern emerged for subtype discrimination, where tissue complexity metrics carry the predictive information while fluid compartments alone exhibit lower discrimination ($\text{AUC} = 0.643$ and $0.560$).
- **Reinterpretation:**
  As the reviewer astutely noted, finding that a reduced subset (DTI/DKI) performs equivalently to the complete framework for cross-sectional binary classification is a valuable scientific result. We have thoroughly re-interpreted the rationale in the revised manuscript:
  - Multi-compartment integration does **not** provide incremental discriminatory power over kurtosis/tensor metrics for cross-sectional case-control classification.
  - The true value of the integrated IVIM-FWI-DKI model lies not in boosting static binary classifier accuracy, but in **biophysical characterization**: simultaneously isolating vascular pseudo-diffusion ($PF$) and extracellular free-water ($FW$) to disentangle active neuroinflammatory edema from permanent axonal disruption—a distinction critical for mechanistic research and prospective longitudinal treatment monitoring.

---

### Comment 2: Calibration of Clinical Utility Claims
> **Reviewer:** *"Articulating a potential clinical application is not equivalent to demonstrating clinical utility. The present study does not show that the derived microstructural stratification informs a specific clinical decision... If the framework is intended to be applied only after a clinician has already established that a patient has a psychotic-spectrum disorder, then the principal patient-versus-healthy-control classification task does not directly correspond to the clinical decision faced at that stage... The manuscript therefore needs to distinguish clearly between demonstrating case-control separability and demonstrating clinical decision utility."*

**Response:**
We fully agree with the reviewer's distinction between case-control separability and clinical decision utility. We have revised the entire manuscript (Abstract, Introduction, Results, Discussion, Conclusion) to calibrate all clinical claims:
1. **Removed Unsupported Claims:** We have excised all statements suggesting that the framework serves as an "objective diagnostic tool", can "refine clinical diagnoses", or "optimizes personalized treatment strategies".
2. **Clarified Primary Endpoint Scope:** We explicitly explain that the primary patient-versus-control classification demonstrates **biological separability and sensitivity to disease pathology** across multi-site neuroimaging cohorts, rather than a clinical screening tool (which clinicians do not need for already-diagnosed patients).
3. **Subtype Heterogeneity:** We discuss the secondary SCZ vs Non-SCZ analysis as evidence of a continuous, shared neurobiological vulnerability spectrum across psychotic disorders rather than discrete categorical boundaries.
4. **Prospective Applications:** We strictly present applications to treatment prediction, disease progression monitoring, and clinical decision support as hypotheses for future prospective longitudinal studies.

---

## 2. Section-by-Section Manuscript Text Revisions

### Abstract Revision
- **Previous:** *"Our integrated diffusion framework provides an objective diagnostic and microstructural stratification tool that can refine psychotic disorder diagnoses and inform personalized clinical treatment strategies."*
- **Revised:** *"Here, we present a multi-compartment diffusion MRI framework combining IVIM, FWI, and DKI to characterize cortical microstructural alterations across 1,267 participants from the Human Connectome Project. A neural network classifier achieved strong case-control separability using regional kurtosis and diffusivity features. However, systematic component ablation revealed that DTI and DKI metrics alone account for the cross-sectional discriminatory performance, with IVIM and FWI compartments providing no incremental classification gain in this cohort. Rather than serving as an unguided diagnostic or treatment selector, the multi-compartment framework provides a biophysical profiling approach to isolate vascular, extracellular, and axonal alterations, offering a foundation for future longitudinal and mechanistic investigations."*

### Introduction Revision (Section 1)
- Add clear framing of the ablation hypothesis:
  > *"While integrating multi-compartment diffusion models is theoretically motivated by the need to resolve biophysical ambiguity among perfusion, edema, and cellular complexity, it remains essential to empirically test whether multi-compartment integration provides incremental classification value over conventional single-model approaches (e.g., DTI or DKI alone). In this study, we perform a systematic component ablation to test this hypothesis directly."*

### Results Addition (Section 4.5: Incremental Value of Model Integration: Ablation Analysis)
- Insert **Table 5** and **Table 6** summarizing the 7 model configurations.
- Reference **Figure 17** (Patient vs Control ROC/PR/Bars) and **Figure 18** (SCZ vs Non-SCZ ROC/PR/Bars).
- Highlight key findings:
  - DTI-only and DKI-only models match the Full Integrated model in AUC-ROC and Average Precision.
  - Perfusion ($PF$) and Free-Water ($FW$) alone show moderate individual discrimination ($\text{AUC} \approx 0.69 - 0.74$) but do not incrementally improve upon kurtosis metrics.

### Discussion Revision (Section 5)
- **New Subsection 5.3: "Incremental Value and Specificity of Diffusion Compartments"**:
  - Discuss why mean kurtosis (MK) and mean diffusivity (MD) in frontal, cingulate, and temporal regions dominate cross-sectional classification.
  - Transparently state that adding IVIM and FWI does not boost cross-sectional classification accuracy.
  - Explain the distinct biophysical value: IVIM/FWI are essential for biological disambiguation (separating neuroinflammation from axonal loss) rather than static binary classification.
- **New Subsection 5.4: "Distinguishing Biological Separability from Clinical Decision Utility"**:
  - Explicitly acknowledge the mismatch between case-control classification and clinical workflows.
  - Clarify that high AUC-ROC demonstrates disease-associated microstructural disruption, not clinical utility.
  - Reframe subtype classification as highlighting biological continuum rather than categorical diagnostic boundaries.

### Conclusion Revision (Section 6)
- **Revised Conclusion:**
  > *"In summary, we evaluated an integrated IVIM-FWI-DKI diffusion MRI framework across two large multi-site cohorts. Systematic ablation demonstrates that cortical kurtosis and diffusivity metrics drive cross-sectional case-control discrimination, while multi-compartment integration provides detailed biophysical characterization of tissue compartments without adding classification performance. The framework is not intended as an unguided clinical diagnostic or treatment selection tool; rather, it offers a standardized quantitative profiling methodology for future longitudinal studies of disease mechanisms and treatment monitoring."*
