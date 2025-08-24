# Interactionist Phylogeny Engine - Design Document
**Version 1.1 - Velotta Lab Edition**

## Executive Summary

The Interactionist Phylogeny Engine (IPE) represents a paradigm shift in evolutionary modeling—from historical reconstruction to predictive simulation. Specifically tailored for the Velotta Lab's research on high-altitude adaptation and phenotypic plasticity, IPE maps evolutionary trajectories through physiologically-explicit state spaces defined by organism-environment strategic interactions. The system predicts evolutionary responses to environmental challenges like hypoxia, temperature stress, and salinity changes by computing game-theoretic equilibria across multi-dimensional parameter spaces, with special emphasis on rapid contemporary evolution and maladaptive plasticity.

## Vision Statement

Create a computational framework that treats evolution as navigation through physiological possibility space, where phylogenetic relationships emerge from strategic convergence rather than common descent. Enable the Velotta Lab researchers to:
- Predict evolutionary responses to high-altitude environments
- Model the evolution of phenotypic plasticity (adaptive and maladaptive)
- Simulate rapid adaptation in response to environmental change
- Integrate molecular, physiological, and organismal data
- Visualize complex trait evolution across environmental gradients

## Core Concepts

### Physiological State Space Architecture

Every organism-environment pair occupies a position in high-dimensional state space defined by:
- **Environmental parameters** 
  - Oxygen partial pressure (PO₂)
  - Temperature and thermal variability
  - Altitude/depth gradients
  - Salinity (for aquatic systems)
  - Resource availability
  
- **Physiological capacity space**
  - Cardiovascular parameters (heart mass, hematocrit, blood viscosity)
  - Respiratory traits (lung capacity, diffusion efficiency)
  - Metabolic characteristics (BMR, VO₂max, thermogenic capacity)
  - Osmoregulatory capabilities (ion transport capacity)
  - Cellular traits (mitochondrial density, capillarization)
  
- **Plasticity dimensions**
  - Acclimation capacity
  - Reaction norm shape
  - Reversibility of plastic responses
  - Costs of plasticity
  
- **Performance metrics**
  - Aerobic scope
  - Thermal tolerance range
  - Swimming/locomotor performance
  - Growth rate under stress

Evolution becomes pathfinding through this space, constrained by:
- Physiological trade-offs (e.g., oxygen delivery vs blood viscosity)
- Developmental constraints
- Thermodynamic boundaries
- Historical contingency from ancestral states

### Strategic Phylogeny with Plasticity

Traditional phylogeny asks "who is related to whom?"
Strategic phylogeny asks "what physiological strategies are accessible from where?"

Key innovations for plasticity research:
- **Maladaptive valleys**: Regions where ancestral plasticity reduces fitness
- **Genetic assimilation paths**: Trajectories toward constitutive expression
- **Plasticity-first evolution**: Phenotypic changes preceding genetic changes
- **Contemporary evolution**: Rapid adaptation over 10-100 generations

### Physiological Game Theory

Every level involves strategic resource allocation:
- **Hypoxia allocation game**: Distribute limited oxygen among competing tissues
- **Thermogenesis trade-off**: Balance heat production vs oxygen consumption
- **Osmoregulation game**: Allocate energy between ion pumping and other functions
- **Growth vs survival**: Invest in immediate survival or future reproduction

## System Architecture

### Layer 1: Physiological State Engine
**Purpose**: Define and navigate physiologically-explicit state space
- High-altitude specific state representation
- Organ-system integration
- Metabolic network modeling
- Plasticity reaction norm tracking
- Environmental gradient mapping

### Layer 2: Adaptive Game Framework
**Purpose**: Compute physiological trade-offs and optimal strategies
- Hypoxia response games
- Thermogenesis optimization
- Osmoregulatory balance
- Multi-organ resource allocation
- Plasticity cost-benefit analysis

### Layer 3: Contemporary Evolution Simulator
**Purpose**: Model rapid evolutionary dynamics
- Standing genetic variation
- Phenotypic selection differentials
- Plasticity evolution (genetic assimilation)
- Gene flow and population structure
- Environmental stochasticity

### Layer 4: Experimental Phylogeny Tracker
**Purpose**: Connect to empirical data
- Common garden experiment modeling
- Reciprocal transplant simulation
- Acclimation response curves
- Transcriptomic integration
- Morphometric trajectories

### Layer 5: Prediction Module with Validation
**Purpose**: Generate testable hypotheses
- Trait evolution forecasting
- Plasticity trajectory prediction
- Range shift modeling
- Experimental outcome prediction
- Uncertainty quantification with empirical bounds

### Layer 6: Lab-Integrated Visualization
**Purpose**: Interactive exploration for research
- Altitude gradient visualizer
- Plasticity landscape mapper
- Organ system dashboard
- Gene expression heatmaps
- Real-time experimental comparison

## User Experience Design

### Primary Interface: Altitude Adaptation Explorer

**3D Physiological Space**
- Navigate through PO₂ × Temperature × Altitude space
- Color gradients show metabolic performance
- Particle clouds represent population variation
- Ghost trajectories show acclimation paths
- Overlays for organ-specific responses

**Interactive Controls**
- Altitude slider (0-5000m) with real-time PO₂ calculation
- Temperature controls with seasonal variation
- Plasticity parameter adjustments
- Generation counter for rapid evolution
- Data import for field measurements

### Secondary Interface: Plasticity Evolution Workspace

**Reaction Norm Visualizer**
- Genotype × Environment interaction plots
- Maladaptive response highlighting
- Genetic assimilation animation
- Costs of plasticity calculator
- Ancestral vs derived comparison

### Tertiary Interface: Multi-Organ System Monitor

**Integrated Physiology Dashboard**
- Heart: mass, output, hypertrophy index
- Lungs: ventilation, diffusion capacity
- Blood: hematocrit, P50, viscosity
- Muscle: fiber type, mitochondrial density
- Real-time resource allocation flows

### Research Tools Interface: Hypothesis Testing Suite

**Experimental Design Module**
- Common garden setup wizard
- Reciprocal transplant designer
- Selection experiment simulator
- Power analysis calculator
- Expected outcome generator

## Visual Design Language

### Scientific Aesthetics
- **Physiological accuracy**: Anatomically-informed organ representations
- **Data clarity**: Publication-ready figure generation
- **Environmental realism**: Topographic altitude gradients
- **Molecular detail**: Gene network visualizations

### Color System for Physiological States
- **Oxygen saturation**: Deep purple (hypoxic) → bright red (normoxic)
- **Metabolic rate**: Blue (low) → yellow → red (high)
- **Plasticity**: Green (adaptive) → gray (neutral) → orange (maladaptive)
- **Altitude gradient**: Sea level blues → alpine whites
- **Selection strength**: Transparent → opaque red

## Simulation Modes

### Field Research Mode
Simulate specific field sites:
- **Mount Evans, CO** (4,350m): Deer mouse summit populations
- **Rocky Mountain National Park**: Elevation transects
- **Connecticut Rivers**: Alewife landlocking events
- **Lake Champlain**: Freshwater invasion dynamics

### Laboratory Mode
Match experimental conditions:
- **Hypobaric chambers**: Simulate altitude in lab
- **Temperature cycles**: Diurnal and seasonal variation
- **Salinity challenges**: Freshwater to seawater transitions
- **Exercise trials**: Swimming tunnels and treadmills

### Rapid Evolution Mode
Contemporary evolution scenarios:
- **Climate change responses**: 50-year projections
- **Invasion dynamics**: 20-generation establishment
- **Urbanization adaptation**: City heat island effects
- **Conservation scenarios**: Managed relocation outcomes

### Teaching Mode for Lab Training
- **Graduate student tutorials**: Physiological ecology principles
- **Undergraduate labs**: Virtual altitude experiments
- **Journal club companion**: Explore papers' predictions
- **Grant proposal tool**: Visualize proposed research

## Technical Requirements

### Performance Targets
- Real-time simulation of 10⁴ individuals
- Smooth visualization across 12 altitude zones
- Sub-second plasticity calculations
- 60 FPS for rotating 3D landscapes
- Instant gene expression heatmap updates

### Integration Requirements
- Import from Sable Systems respirometry
- Connect to DESeq2/EdgeR outputs
- Export to Prism/R for statistics
- Link to NCBI gene databases
- Sync with lab PostgreSQL server

## Use Cases for Velotta Lab Research

### Current Project Applications

1. **Deer Mouse High-Altitude Adaptation**
   - Model thermogenic capacity evolution
   - Predict organ mass optimization
   - Simulate seasonal selection pressures
   - Project climate change impacts

2. **Alewife Freshwater Evolution**
   - Model osmoregulatory trait changes
   - Predict swimming performance trade-offs
   - Simulate parallel evolution events
   - Forecast invasion success

3. **Maladaptive Plasticity Research**
   - Quantify plasticity reduction rates
   - Identify genetic assimilation hotspots
   - Predict compensatory mutations
   - Model plasticity-environment mismatches

### Future Research Directions

1. **Multi-Species Comparisons**
   - Convergent high-altitude solutions
   - Phylogenetic comparative methods
   - Cross-species plasticity patterns

2. **Climate Change Projections**
   - Range shift predictions
   - Phenological mismatch risks
   - Evolutionary rescue scenarios

3. **Conservation Applications**
   - Assisted migration planning
   - Captive breeding optimization
   - Reintroduction site selection

## Development Phases

### Phase 1: Core Physiology Engine (Months 1-3)
- Implement altitude-specific state space
- Build hypoxia response models
- Create organ system framework
- Develop metabolic calculators

### Phase 2: Plasticity Module (Months 4-6)
- Reaction norm representation
- Maladaptive response detection
- Genetic assimilation algorithms
- Plasticity cost functions

### Phase 3: Data Integration (Months 7-9)
- Respirometry data import
- Transcriptome mapping
- Morphometric integration
- Field measurement calibration

### Phase 4: Prediction & Validation (Months 10-12)
- Experimental outcome prediction
- Model validation framework
- Uncertainty quantification
- Hypothesis generation tools

### Phase 5: Lab Deployment (Months 13-15)
- User training materials
- Custom workflows for lab projects
- Publication figure generation
- Collaborative features

## Success Metrics

### Research Impact
- Accurate prediction of experimental outcomes
- Novel hypotheses generated and tested
- Integration into lab's publication pipeline
- Adoption by collaborating labs

### Educational Value
- Student engagement in lab meetings
- Use in university courses
- Public outreach effectiveness
- Grant proposal enhancement

### Technical Performance
- Model accuracy against empirical data
- Computational efficiency
- User task completion time
- System reliability

## Risk Mitigation

### Scientific Risks
- **Oversimplification**: Modular complexity levels
- **Parameter uncertainty**: Sensitivity analysis built-in
- **Validation challenges**: Extensive empirical testing

### Technical Risks
- **Data integration complexity**: Standardized import formats
- **Computational demands**: Cloud computing options
- **User learning curve**: Graduated complexity modes

## Future Vision

### Version 2.0 - Molecular Integration
- Full transcriptomic modeling
- Protein interaction networks
- Epigenetic inheritance
- Metabolomic pathways

### Version 3.0 - Ecosystem Scale
- Community assembly
- Predator-prey dynamics
- Disease evolution
- Microbiome interactions

### Long-term Goals
- Real-time field station integration
- Automated experiment design
- AI-assisted hypothesis generation
- Global adaptation observatory

## Design Philosophy

The IPE Velotta Lab Edition embodies the principle that organisms and environments mutually construct each other through physiological interactions. Evolution emerges not from genetic programs alone but from the dynamic interplay between phenotypic plasticity and environmental challenges. The tool makes visible what traditional approaches obscure: the vast space of unrealized physiological possibilities, the maladaptive valleys that must be crossed, and the rapid timescales on which contemporary evolution unfolds.

This specialized version transforms abstract evolutionary theory into concrete, measurable predictions about real organisms facing real environmental challenges—from deer mice on mountaintops to alewives in freshwater lakes. By grounding simulations in physiological reality, researchers can move seamlessly between computational predictions and experimental validation, accelerating the pace of discovery in evolutionary physiology.