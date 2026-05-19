---
layout: page
title: Research
permalink: /research/
description: >-
  Research themes, contributions, and publications by Dr Cyril Pernet —
  open neuroimaging standards, statistical methods, and reproducible brain imaging.
---

My research centres on making neuroimaging more rigorous and reproducible.
This spans three overlapping areas: community standards for data organisation
and sharing, statistical methods for M/EEG and fMRI, and the tools and
infrastructure that make open science tractable at scale.

## Research Themes

### Open Neuroimaging Standards

Much of the field's reproducibility problem is upstream of analysis — in how data
are collected, described, and shared. I co-lead or contribute to several community
efforts that define how this should work: BIDS extensions for EEG, Genetics, and
PET; the OHBM COBIDAS guidelines for M/EEG and MRI; and the Open Brain Consent
templates that allow participants to authorise open sharing under GDPR. The common
thread is building community consensus around practices that make individual studies
reusable.

Related:  
[COBIDAS MEEG](https://cobidasmeeg.wordpress.com/) ·
[BIDS](https://bids.neuroimaging.io/) ·
[Open Brain Consent](https://open-brain-consent.readthedocs.io/en/stable/) ·
Join us on the [BIDS EEG Derivatives (BEP021)](https://github.com/bids-standard/bep021) `Standard`

**BIDS extension for electrophysiology derivatives**

Co-leading the BIDS Enhancement Proposal for storing and sharing EEG pre-processing and analysis derivatives. Defines a community standard for derivative file naming, folder structure, and metadata.

**Tags:** BIDS, EEG, Standard
**Links:** [GitHub](https://github.com/bids-standard/bep021) . [Get involved](https://bids.neuroimaging.io/get_involved.html)

### Statistical Methods for Neuroimaging

Standard parametric methods applied mass-univariately at the sensor or voxel level carry assumptions that are routinely violated in neuroimaging data — non-normality, outliers, mis-specified models, and inflated false positives from multiple comparisons. I develop and maintain tools for robust estimation (trimmed means, robust correlations, Winsorised statistics), hierarchical GLM for EEG/MEG (LIMO MEEG), adaptive
thresholding for single-subject fMRI maps, and methods for single-trial ERP analysis.

Related: [LIMO MEEG](/software/) · [SPMup](/software/) ·
[Robust Statistical Toolbox](https://github.com/CPernet/Robust_Statistical_Toolbox)

### Brain Structure, Function & Cognition

Empirically, my work spans categorization processing (language, faces, objects, ..), fMRI task design and signal quality, structural analyses of the cortex, and — more recently — transdiagnostic psychopathology and interactions with environmental factors. A consistent methodological concern is whether the numbers we report are well-defined and interpretable: % signal change, effect sizes, confidence intervals, and visualisations that convey uncertainty honestly.

---

## Selected Publications

### Reproducibility & Best Practices

**Issues and recommendations from the OHBM COBIDAS MEEG committee for reproducible EEG and MEG research**  
Pernet, C., Garrido, M., Gramfort, A., Maurits, N., Michel, C., Pang, E., Salmelin, R., Schoffelen, JM., Valdes-Sosa, P., & Puce, A. (2020). *Nature Neuroscience.*  
[doi](https://www.nature.com/articles/s41593-020-00709-0) · [Website](https://cobidasmeeg.wordpress.com/)

**Open and reproducible neuroimaging: from study inception to publication**  
Niso, G. et al. (2022). *NeuroImage, 119623.*  
[doi](https://doi.org/10.1016/j.neuroimage.2022.119623) · [Interactive book](https://agahkarakuzu.github.io/oreoni/)

**Improving functional magnetic resonance imaging reproducibility**  
Pernet, C. & Poline, J-B. (2015). *GigaScience, 4, 15.*  
[doi](https://academic.oup.com/gigascience/article/4/1/s13742-015-0055-8/2707541)

**Data visualization for inference in tomographic brain imaging**  
Pernet, C. R. & Madan, C. R. (2019). *European Journal of Neuroscience, 51, 695–705.*  
[doi](https://onlinelibrary.wiley.com/doi/full/10.1111/ejn.14430) · [Code](https://github.com/CPernet/MRI_FaceData_Wakeman-Henson)

**Improving standards in brain-behavior correlation analyses**  
Rousselet, G. A. & Pernet, C. R. (2012). *Frontiers in Human Neuroscience, 6, 119.*  
[doi](https://www.frontiersin.org/articles/10.3389/fnhum.2012.00119/full)

**Can We Standardize Clinical Functional Neuroimaging Procedures?**  
Beisteiner, R., Pernet, C. R. & Stippich, C. (2019). *Frontiers in Neurology, 8, 1153.*  
[doi](https://www.frontiersin.org/articles/10.3389/fneur.2018.01153/full)

**Brainhack: developing a culture of open, inclusive, community-driven neuroscience**  
Gau, R. et al. (2021). *Neuron, 109, 1769–1775.*  
[PDF](https://www.cell.com/neuron/pdf/S0896-6273(21)00231-2.pdf)

**Visual object categorization in the brain: what can we really learn from ERP peaks?**  
Rousselet, G. A., Pernet, C. R., Caldara, R. & Schyns, P. G. (2011). *Frontiers in Human Neuroscience, 5, 156.*  
[doi](https://www.frontiersin.org/articles/10.3389/fnhum.2011.00156/full)

**Quantifying the Time Course of Visual Object Processing Using ERPs: It's Time to Up the Game**  
Rousselet, G. A. & Pernet, C. R. (2011). *Frontiers in Psychology, 2, 107.*  
[doi](https://www.frontiersin.org/articles/10.3389/fpsyg.2011.00107/full)

**Single-trial analyses: why bother?**  
Pernet, C. R., Sajda, P. & Rousselet, G. A. (2011). *Frontiers in Psychology, 2, 322.*  
[doi](https://www.frontiersin.org/articles/10.3389/fpsyg.2011.00322/full)

### Open Data & Standards

**EEG-BIDS, an extension to the brain imaging data structure for electroencephalography**  
Pernet, C. R., Appelhoff, S., Gorgolewski, K., Flandin, G., Phillips, C., Delorme, A. & Oostenveld, R. (2019). *Scientific Data, 6, 103.*  
[doi](https://www.nature.com/articles/s41597-019-0104-8)

**PET-BIDS, an extension to the brain imaging data structure for positron emission tomography**  
Nørgaard, M. et al. (2022). *Scientific Data, 9, 65.*  
[doi](https://www.nature.com/articles/s41597-022-01164-1)

**The genetics-BIDS extension: Easing the search for genetic data associated with human brain imaging**  
Moreau, C., Jean-Louis, M., Ross, B., Markiewicz, C., Turner, J., Calhoun, V., Nichols, T. & Pernet, C. (2020). *GigaScience, 9(10), giaa104.*  
[doi](https://academic.oup.com/gigascience/article/9/10/giaa104/5928221)

**The Open Brain Consent: Informing research participants and obtaining consent to share brain imaging data**  
The Open Brain Consent Working Group (2021). *Human Brain Mapping.*  
[doi](https://psyarxiv.com/f6mnp/) · [Website](https://open-brain-consent.readthedocs.io/en/stable/)

**On the Long-term Archiving of Research Data**  
Pernet, C., Svarer, C., Blair, R. et al. (2023). *Neuroinformatics, 21, 243–246.*  
[doi](https://doi.org/10.1007/s12021-023-09621-x)

**Improving data availability for brain image biobanking in healthy subjects: practice-based suggestions from an international multidisciplinary working group**  
Shenkin, S. D., Pernet, C., Nichols, T. E., Poline, J. B. et al. (2017). *NeuroImage, 153, 399–409.*  
[doi](https://www.sciencedirect.com/science/article/abs/pii/S1053811917301416)

**Longitudinal multi-centre brain imaging studies: guidelines and practical tips for accurate and reproducible imaging endpoints and data sharing**  
Wiseman, S. J., Meijboom, R., Valdés Hernández, M. D. C., Pernet, C. et al. (2019). *Trials, 20, 1.*  
[doi](https://trialsjournal.biomedcentral.com/articles/10.1186/s13063-018-3113-6)

**#EEGManyLabs: Investigating the Replicability of Influential EEG Experiments**  
Pavlov, Y. et al. (2021). *Cerebral Cortex.*  
[doi](https://psyarxiv.com/528nr/)

### Methods & Applications

**Mixture model for single-subject fMRI thresholding**  
Gorgolewski, K. et al. (2012). *Frontiers in Human Neuroscience.*  
[doi](https://www.frontiersin.org/articles/10.3389/fnhum.2012.00245/full) · [Code](https://github.com/CPernet/spmup/tree/master/adaptative_threshold)

**BOLD signal decomposition: correcting HRF parameter estimates and computing percentage signal change**  
Pernet, C. R. (2014). *Frontiers in Neuroscience.*  
[doi](https://www.frontiersin.org/articles/10.3389/fnins.2014.00001/full) · [Code](https://github.com/CPernet/spmup/tree/master/hrf)

---

[Full publication list on Google Scholar &rarr;](http://scholar.google.co.uk/citations?user=yz6s_e8AAAAJ&hl=en){: target="_blank" rel="noopener"}
