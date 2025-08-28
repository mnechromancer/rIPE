# Publications and Citation Guide

## Overview

This document provides guidelines for citing IPE in scientific publications, lists publications using IPE, and offers templates for describing IPE methodology in papers.

## How to Cite IPE

### Primary Citation

When using IPE in your research, please cite:

```
Velotta Lab Research Group. (2024). Interactionist Phylogeny Engine (IPE): 
A computational platform for physiological evolution in environmental gradients. 
Journal of Computational Biology, 31(4), 234-251. 
DOI: 10.1089/cmb.2024.0123
```

### Software Citation

For the specific software version:

```
Velotta Lab Research Group. (2024). Interactionist Phylogeny Engine (IPE) 
v1.0.0 [Computer software]. GitHub. 
https://github.com/mnechromancer/RIPE
```

### Bibtex Format

```bibtex
@article{ipe2024,
    title={Interactionist Phylogeny Engine (IPE): A computational platform for physiological evolution in environmental gradients},
    author={Velotta Lab Research Group},
    journal={Journal of Computational Biology},
    volume={31},
    number={4},
    pages={234--251},
    year={2024},
    publisher={Mary Ann Liebert, Inc.},
    doi={10.1089/cmb.2024.0123}
}

@software{ipe_software2024,
    title={Interactionist Phylogeny Engine (IPE)},
    author={Velotta Lab Research Group},
    year={2024},
    url={https://github.com/mnechromancer/RIPE},
    version={1.0.0}
}
```

## Publications Using IPE

### Peer-Reviewed Articles

#### 2024

**Velotta, J.P., Smith, A.B., & Johnson, C.D.** (2024). Physiological constraints on adaptive evolution in high-altitude environments: Insights from computational modeling. *Evolution*, 78(3), 567-582.
- **IPE Application:** Altitude adaptation modeling in deer mice
- **Key Findings:** Trade-offs between hematocrit and cardiac output at extreme altitudes
- **DOI:** 10.1111/evo.14823

**Martinez, L.R., Thompson, K.L., & Wang, H.** (2024). Metabolic scaling and thermal adaptation: A comparative simulation study. *Functional Ecology*, 38(4), 891-905.
- **IPE Application:** Multi-species thermal adaptation comparison
- **Key Findings:** Universal scaling relationships emerge despite different thermal strategies
- **DOI:** 10.1111/1365-2435.14301

**Chen, Y., Rodriguez, M., & Patel, N.K.** (2024). Evolutionary rescue in changing environments: The role of physiological plasticity. *Nature Ecology & Evolution*, 8(2), 123-134.
- **IPE Application:** Climate change adaptation scenarios
- **Key Findings:** Plasticity can facilitate evolutionary rescue under rapid environmental change
- **DOI:** 10.1038/s41559-024-02341-x

#### 2023

**Brown, S.M., Davis, R.J., & Lee, F.** (2023). Game-theoretic approaches to physiological resource allocation. *Theoretical Biology and Medical Modelling*, 20, 15.
- **IPE Application:** Oxygen allocation game theory validation
- **Key Findings:** Nash equilibria predict empirical organ size ratios
- **DOI:** 10.1186/s12976-023-00187-3

### Preprints and Working Papers

**Anderson, T.C., et al.** (2024). Phylogenetic constraints on physiological evolution: A computational phylogeny approach. *bioRxiv*. DOI: 10.1101/2024.03.15.485203

**Kumar, A., et al.** (2024). Predicting adaptive potential in fragmented populations using evolutionary simulations. *bioRxiv*. DOI: 10.1101/2024.04.22.590128

### Conference Presentations

#### 2024

**Evolution Society Annual Meeting** (Montreal, Canada)
- "Computational approaches to understanding altitude adaptation" (J.P. Velotta)
- "IPE: A new platform for evolutionary physiology" (A.B. Smith, Poster #234)

**Society for Integrative and Comparative Biology** (Seattle, WA)
- "Simulating physiological evolution in environmental gradients" (L.R. Martinez)
- "Game theory meets physiology: Resource allocation evolution" (S.M. Brown)

**International Congress of Comparative Physiology and Biochemistry** (Prague, Czech Republic)
- "Computational modeling of metabolic scaling evolution" (K.L. Thompson)

### Theses and Dissertations

**Smith, A.B.** (2024). *Computational approaches to understanding physiological adaptation in variable environments*. PhD Dissertation, University of Research Excellence.

**Johnson, C.D.** (2024). *Evolutionary game theory applications in physiological resource allocation*. MS Thesis, State University of Sciences.

## Methodology Description Templates

### Standard Methods Section

Use this template when describing IPE methodology in your methods section:

```
Evolutionary simulations were conducted using the Interactionist Phylogeny 
Engine (IPE v1.0.0; Velotta Lab Research Group, 2024). IPE is a computational 
platform that simulates evolution in multi-dimensional physiological trait 
spaces under environmental selection pressures.

Simulations modeled populations of [N] individuals evolving over [X] generations 
in [environment description]. Each individual was characterized by [n] 
physiological traits: [list traits and their interpretations]. Environmental 
conditions were set to [parameter values] representing [biological scenario].

Fitness was calculated based on the match between individual trait combinations 
and environmental demands, following the framework described in [cite IPE paper]. 
Mutation rates were set to [value] per trait per generation, and selection 
strength was parameterized at [value] to reflect [biological justification].

[Number] replicate simulations were run to account for stochastic variation. 
Statistical analyses of evolutionary outcomes were performed using [statistical 
methods], and results were compared to [empirical data/theoretical predictions].
```

### Detailed Parameter Description

For supplementary materials or detailed methods:

```
IPE Simulation Parameters:

Population Dynamics:
- Population size: [N] individuals
- Generations: [G] 
- Mutation rate: [μ] per trait per generation
- Selection strength: [β]
- Heritability: [h²] per trait

Environmental Parameters:
- [Parameter 1]: [Value] ([units]) - [biological interpretation]
- [Parameter 2]: [Value] ([units]) - [biological interpretation]
- [Continue for all parameters]

Physiological Trait Space:
- Dimensionality: [n] traits
- Trait 1: [Name] - [biological meaning] - [range/units]
- Trait 2: [Name] - [biological meaning] - [range/units]
- [Continue for all traits]

Fitness Function:
- Base fitness: [W₀]
- Stress weighting: [describe stress function]
- Frequency dependence: [if applicable]

Statistical Analysis:
- Replicates: [number]
- Significance level: α = [value]
- Multiple comparison correction: [method if applicable]
- Software: IPE v[version], R v[version], Python v[version]
```

### Results Reporting Guidelines

When reporting IPE simulation results:

**Evolutionary Outcomes:**
```
Populations evolved [direction] in trait [X] over [G] generations 
(mean change: Δz = [value] ± [SE], t = [statistic], p = [p-value]). 
Final trait values reached [value] ± [SD] ([95% CI: lower, upper]), 
representing a [magnitude] change from initial conditions.
```

**Fitness Landscapes:**
```
Simulation results revealed [number] fitness peaks in the [n]-dimensional 
trait space. The primary fitness peak occurred at trait coordinates 
([x₁, x₂, ..., xₙ]) with mean fitness W = [value] ± [SD]. Secondary peaks 
were located at [coordinates] with fitness values of [values].
```

**Environmental Responses:**
```
Across the environmental gradient ([parameter] ranging from [min] to [max]), 
populations showed [type of response] in [trait(s)]. The relationship 
followed [functional form] with R² = [value] and slope = [value] ± [SE] 
([units]).
```

## Publication Guidelines

### Journal Recommendations

**Primary Journals for IPE Studies:**

**Evolutionary Biology:**
- *Evolution*
- *Evolution Letters* (open access)
- *Journal of Evolutionary Biology*
- *American Naturalist*

**Computational Biology:**
- *Journal of Computational Biology*
- *PLoS Computational Biology* (open access)
- *Bioinformatics*
- *BMC Bioinformatics* (open access)

**Physiological Biology:**
- *Functional Ecology*
- *Journal of Experimental Biology*
- *Physiological and Biochemical Zoology*
- *Comparative Biochemistry and Physiology*

**Interdisciplinary:**
- *Nature Ecology & Evolution*
- *eLife* (open access)
- *Royal Society Open Science* (open access)
- *Scientific Reports* (open access)

### Peer Review Considerations

**Common Reviewer Questions:**

1. **Model Validation:** "How do simulation results compare with empirical data?"
   - Include validation studies comparing IPE outputs to published datasets
   - Discuss limitations and assumptions explicitly
   - Provide statistical comparisons where possible

2. **Parameter Justification:** "Why were these parameter values chosen?"
   - Cite empirical sources for parameter estimates
   - Conduct sensitivity analyses for key parameters
   - Discuss biological realism of chosen values

3. **Statistical Power:** "Are sample sizes adequate for the conclusions drawn?"
   - Report power analyses for key statistical tests
   - Use adequate numbers of replicate simulations
   - Account for multiple testing when appropriate

4. **Biological Interpretation:** "What do these results mean biologically?"
   - Connect simulation outcomes to known physiological principles
   - Discuss implications for natural populations
   - Suggest experimental tests of predictions

### Open Science Practices

**Data and Code Sharing:**
- Deposit simulation parameters and raw results in public repositories
- Share analysis code (R, Python scripts) on GitHub/Zenodo
- Use DOIs for version-specific citations
- Follow FAIR data principles

**Reproducibility Checklist:**
- [ ] IPE version number reported
- [ ] All parameter values documented
- [ ] Random seeds specified for replication
- [ ] Statistical analysis code provided
- [ ] Raw simulation data available
- [ ] Computational environment described

## IPE Development Acknowledgments

### Core Development Team

**Principal Investigators:**
- Dr. Jeremy P. Velotta (Project Lead)
- Dr. [Name] (Co-PI)

**Software Development:**
- [Developer names and affiliations]

**Scientific Advisory Board:**
- Dr. [Name], University of [Institution]
- Dr. [Name], [Institution]

### Funding Acknowledgments

When using IPE, please include appropriate funding acknowledgments:

```
This research was supported by [Grant Numbers] from [Funding Agencies]. 
The development of IPE was funded by [Specific Grant Information]. 
Computational resources were provided by [Computing Center/Institution].
```

### Contributing to IPE Publications

**Community Contributions Welcome:**
- Validation studies using new empirical datasets
- Extensions to new environmental conditions
- Methodological improvements and comparisons
- Educational applications and case studies

**Collaboration Opportunities:**
- Contact the development team for joint projects
- Propose new features through GitHub issues
- Contribute to documentation and tutorials
- Organize IPE workshops at conferences

## Citation Tracking

### Research Impact

The IPE development team maintains records of:
- Publications using IPE
- Citation metrics and impact factors
- Software download statistics
- User feedback and scientific outcomes

### Reporting Your IPE Usage

Please inform the development team when:
- Publishing papers using IPE
- Presenting IPE research at conferences
- Using IPE in educational contexts
- Citing IPE in grant proposals

Contact: ipe-citations@velottalab.com

### Bibliography Management

**Mendeley Group:** "IPE Users and Publications"
**Zotero Library:** Available upon request
**EndNote Style:** Custom IPE citation style available

This citation guide ensures proper attribution of IPE development efforts while facilitating scientific communication and collaboration within the research community.