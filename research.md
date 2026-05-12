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

Related: [COBIDAS MEEG](https://cobidasmeeg.wordpress.com/) ·
[BIDS](https://bids.neuroimaging.io/) ·
[Open Brain Consent](https://open-brain-consent.readthedocs.io/en/stable/)

### Statistical Methods for Neuroimaging

Standard parametric methods applied mass-univariately at the sensor or voxel level
carry assumptions that are routinely violated in neuroimaging data — non-normality,
outliers, mis-specified models, and inflated false positives from multiple comparisons.
I develop and maintain tools for robust estimation (trimmed means, robust correlations,
Winsorised statistics), hierarchical GLM for EEG/MEG (LIMO MEEG), adaptive
thresholding for single-subject fMRI maps, and methods for single-trial ERP analysis.

Related: [LIMO MEEG](/software/) · [SPMup](/software/) ·
[Robust Statistical Toolbox](https://github.com/CPernet/Robust_Statistical_Toolbox)

### Brain Structure, Function & Cognition

Empirically, my work spans face and object processing (ERP latencies and magnitudes),
fMRI task design and signal quality, vascular parcellation of the cortex, and —
more recently — transdiagnostic psychopathology and pain–reward interactions.
A consistent methodological concern is whether the numbers we report are
well-defined and interpretable: % signal change, effect sizes, confidence
intervals, and visualisations that convey uncertainty honestly.

---

## Selected Publications

### Reproducibility & Best Practices

<ul class="pub-list">

  <li class="pub-item">
    <span class="pub-title">Issues and recommendations from the OHBM COBIDAS MEEG committee for reproducible EEG and MEG research</span>
    <span class="pub-authors">Pernet, C., Garrido, M., Gramfort, A., Maurits, N., Michel, C., Pang, E., Salmelin, R., Schoffelen, JM., Valdes-Sosa, P., &amp; Puce, A. (2020)</span>
    <span class="pub-venue">Nature Neuroscience</span>
    <div class="pub-links">
      <a class="pub-link" href="https://www.nature.com/articles/s41593-020-00709-0" target="_blank" rel="noopener">doi</a>
      <a class="pub-link" href="https://cobidasmeeg.wordpress.com/" target="_blank" rel="noopener">Website</a>
    </div>
  </li>

  <li class="pub-item">
    <span class="pub-title">Open and reproducible neuroimaging: from study inception to publication</span>
    <span class="pub-authors">Niso, G. et al. (2022)</span>
    <span class="pub-venue">NeuroImage, 119623</span>
    <div class="pub-links">
      <a class="pub-link" href="https://doi.org/10.1016/j.neuroimage.2022.119623" target="_blank" rel="noopener">doi</a>
      <a class="pub-link" href="https://agahkarakuzu.github.io/oreoni/" target="_blank" rel="noopener">Interactive book</a>
    </div>
  </li>

  <li class="pub-item">
    <span class="pub-title">Improving functional magnetic resonance imaging reproducibility</span>
    <span class="pub-authors">Pernet, C. &amp; Poline, J-B. (2015)</span>
    <span class="pub-venue">GigaScience, 4, 15</span>
    <div class="pub-links">
      <a class="pub-link" href="https://academic.oup.com/gigascience/article/4/1/s13742-015-0055-8/2707541" target="_blank" rel="noopener">doi</a>
    </div>
  </li>

  <li class="pub-item">
    <span class="pub-title">Data visualization for inference in tomographic brain imaging</span>
    <span class="pub-authors">Pernet, C. R. &amp; Madan, C. R. (2019)</span>
    <span class="pub-venue">European Journal of Neuroscience, 51, 695–705</span>
    <div class="pub-links">
      <a class="pub-link" href="https://onlinelibrary.wiley.com/doi/full/10.1111/ejn.14430" target="_blank" rel="noopener">doi</a>
      <a class="pub-link" href="https://github.com/CPernet/MRI_FaceData_Wakeman-Henson" target="_blank" rel="noopener">Code</a>
    </div>
  </li>

  <li class="pub-item">
    <span class="pub-title">Improving standards in brain-behavior correlation analyses</span>
    <span class="pub-authors">Rousselet, G. A. &amp; Pernet, C. R. (2012)</span>
    <span class="pub-venue">Frontiers in Human Neuroscience, 6, 119</span>
    <div class="pub-links">
      <a class="pub-link" href="https://www.frontiersin.org/articles/10.3389/fnhum.2012.00119/full" target="_blank" rel="noopener">doi</a>
    </div>
  </li>

  <li class="pub-item">
    <span class="pub-title">Can We Standardize Clinical Functional Neuroimaging Procedures?</span>
    <span class="pub-authors">Beisteiner, R., Pernet, C. R. &amp; Stippich, C. (2019)</span>
    <span class="pub-venue">Frontiers in Neurology, 8, 1153</span>
    <div class="pub-links">
      <a class="pub-link" href="https://www.frontiersin.org/articles/10.3389/fneur.2018.01153/full" target="_blank" rel="noopener">doi</a>
    </div>
  </li>

  <li class="pub-item">
    <span class="pub-title">Brainhack: developing a culture of open, inclusive, community-driven neuroscience</span>
    <span class="pub-authors">Gau, R. et al. (2021)</span>
    <span class="pub-venue">Neuron, 109, 1769–1775</span>
    <div class="pub-links">
      <a class="pub-link" href="https://www.cell.com/neuron/pdf/S0896-6273(21)00231-2.pdf" target="_blank" rel="noopener">PDF</a>
    </div>
  </li>

  <li class="pub-item">
    <span class="pub-title">Visual object categorization in the brain: what can we really learn from ERP peaks?</span>
    <span class="pub-authors">Rousselet, G. A., Pernet, C. R., Caldara, R. &amp; Schyns, P. G. (2011)</span>
    <span class="pub-venue">Frontiers in Human Neuroscience, 5, 156</span>
    <div class="pub-links">
      <a class="pub-link" href="https://www.frontiersin.org/articles/10.3389/fnhum.2011.00156/full" target="_blank" rel="noopener">doi</a>
    </div>
  </li>

  <li class="pub-item">
    <span class="pub-title">Quantifying the Time Course of Visual Object Processing Using ERPs: It's Time to Up the Game</span>
    <span class="pub-authors">Rousselet, G. A. &amp; Pernet, C. R. (2011)</span>
    <span class="pub-venue">Frontiers in Psychology, 2, 107</span>
    <div class="pub-links">
      <a class="pub-link" href="https://www.frontiersin.org/articles/10.3389/fpsyg.2011.00107/full" target="_blank" rel="noopener">doi</a>
    </div>
  </li>

  <li class="pub-item">
    <span class="pub-title">Single-trial analyses: why bother?</span>
    <span class="pub-authors">Pernet, C. R., Sajda, P. &amp; Rousselet, G. A. (2011)</span>
    <span class="pub-venue">Frontiers in Psychology, 2, 322</span>
    <div class="pub-links">
      <a class="pub-link" href="https://www.frontiersin.org/articles/10.3389/fpsyg.2011.00322/full" target="_blank" rel="noopener">doi</a>
    </div>
  </li>

</ul>

### Open Data & Standards

<ul class="pub-list">

  <li class="pub-item">
    <span class="pub-title">EEG-BIDS, an extension to the brain imaging data structure for electroencephalography</span>
    <span class="pub-authors">Pernet, C. R., Appelhoff, S., Gorgolewski, K., Flandin, G., Phillips, C., Delorme, A. &amp; Oostenveld, R. (2019)</span>
    <span class="pub-venue">Scientific Data, 6, 103</span>
    <div class="pub-links">
      <a class="pub-link" href="https://www.nature.com/articles/s41597-019-0104-8" target="_blank" rel="noopener">doi</a>
    </div>
  </li>

  <li class="pub-item">
    <span class="pub-title">PET-BIDS, an extension to the brain imaging data structure for positron emission tomography</span>
    <span class="pub-authors">Nørgaard, M. et al. (2022)</span>
    <span class="pub-venue">Scientific Data, 9, 65</span>
    <div class="pub-links">
      <a class="pub-link" href="https://www.nature.com/articles/s41597-022-01164-1" target="_blank" rel="noopener">doi</a>
    </div>
  </li>

  <li class="pub-item">
    <span class="pub-title">The genetics-BIDS extension: Easing the search for genetic data associated with human brain imaging</span>
    <span class="pub-authors">Moreau, C., Jean-Louis, M., Ross, B., Markiewicz, C., Turner, J., Calhoun, V., Nichols, T. &amp; Pernet, C. (2020)</span>
    <span class="pub-venue">GigaScience, 9(10), giaa104</span>
    <div class="pub-links">
      <a class="pub-link" href="https://academic.oup.com/gigascience/article/9/10/giaa104/5928221" target="_blank" rel="noopener">doi</a>
    </div>
  </li>

  <li class="pub-item">
    <span class="pub-title">The Open Brain Consent: Informing research participants and obtaining consent to share brain imaging data</span>
    <span class="pub-authors">The Open Brain Consent Working Group (2021)</span>
    <span class="pub-venue">Human Brain Mapping</span>
    <div class="pub-links">
      <a class="pub-link" href="https://psyarxiv.com/f6mnp/" target="_blank" rel="noopener">doi</a>
      <a class="pub-link" href="https://open-brain-consent.readthedocs.io/en/stable/" target="_blank" rel="noopener">Website</a>
    </div>
  </li>

  <li class="pub-item">
    <span class="pub-title">On the Long-term Archiving of Research Data</span>
    <span class="pub-authors">Pernet, C., Svarer, C., Blair, R. et al. (2023)</span>
    <span class="pub-venue">Neuroinformatics, 21, 243–246</span>
    <div class="pub-links">
      <a class="pub-link" href="https://doi.org/10.1007/s12021-023-09621-x" target="_blank" rel="noopener">doi</a>
    </div>
  </li>

  <li class="pub-item">
    <span class="pub-title">Improving data availability for brain image biobanking in healthy subjects: practice-based suggestions from an international multidisciplinary working group</span>
    <span class="pub-authors">Shenkin, S. D., Pernet, C., Nichols, T. E., Poline, J. B. et al. (2017)</span>
    <span class="pub-venue">NeuroImage, 153, 399–409</span>
    <div class="pub-links">
      <a class="pub-link" href="https://www.sciencedirect.com/science/article/abs/pii/S1053811917301416" target="_blank" rel="noopener">doi</a>
    </div>
  </li>

  <li class="pub-item">
    <span class="pub-title">Longitudinal multi-centre brain imaging studies: guidelines and practical tips for accurate and reproducible imaging endpoints and data sharing</span>
    <span class="pub-authors">Wiseman, S. J., Meijboom, R., Valdés Hernández, M. D. C., Pernet, C. et al. (2019)</span>
    <span class="pub-venue">Trials, 20, 1</span>
    <div class="pub-links">
      <a class="pub-link" href="https://trialsjournal.biomedcentral.com/articles/10.1186/s13063-018-3113-6" target="_blank" rel="noopener">doi</a>
    </div>
  </li>

  <li class="pub-item">
    <span class="pub-title">#EEGManyLabs: Investigating the Replicability of Influential EEG Experiments</span>
    <span class="pub-authors">Pavlov, Y. et al. (2021)</span>
    <span class="pub-venue">Cerebral Cortex</span>
    <div class="pub-links">
      <a class="pub-link" href="https://psyarxiv.com/528nr/" target="_blank" rel="noopener">doi</a>
    </div>
  </li>

</ul>

### Methods & Applications

<ul class="pub-list">

  <li class="pub-item">
    <span class="pub-title">Mixture model for single-subject fMRI thresholding</span>
    <span class="pub-authors">Gorgolewski, K. et al. (2012)</span>
    <span class="pub-venue">Frontiers in Human Neuroscience</span>
    <div class="pub-links">
      <a class="pub-link" href="https://www.frontiersin.org/articles/10.3389/fnhum.2012.00245/full" target="_blank" rel="noopener">doi</a>
      <a class="pub-link" href="https://github.com/CPernet/spmup/tree/master/adaptative_threshold" target="_blank" rel="noopener">Code</a>
    </div>
  </li>

  <li class="pub-item">
    <span class="pub-title">BOLD signal decomposition: correcting HRF parameter estimates and computing percentage signal change</span>
    <span class="pub-authors">Pernet, C. R. (2014)</span>
    <span class="pub-venue">Frontiers in Neuroscience</span>
    <div class="pub-links">
      <a class="pub-link" href="https://www.frontiersin.org/articles/10.3389/fnins.2014.00001/full" target="_blank" rel="noopener">doi</a>
      <a class="pub-link" href="https://github.com/CPernet/spmup/tree/master/hrf" target="_blank" rel="noopener">Code</a>
    </div>
  </li>

</ul>

---

## Preprints

- Poldrack, R. et al. (2023). [The Past, Present, and Future of the Brain Imaging Data Structure (BIDS)](https://arxiv.org/abs/2309.05768). *arXiv.*
- Randau, M. et al. (2023). [Transdiagnostic psychopathology in the light of robust single-trial event-related potentials](https://www.authorea.com/users/641988/articles/668350-transdiagnostic-psychopathology-in-the-light-of-robust-single-trial-event-related-potentials). *Authorea.*
- Hoskin, R., Pernet, C. & Talmi, D. (2023). [Interactions between the representations of pain and reward suggest dynamic shifts in reference point](https://www.biorxiv.org/content/10.1101/2023.07.20.549309v1). *bioRxiv.*

---

[Full publication list on Google Scholar &rarr;](http://scholar.google.co.uk/citations?user=yz6s_e8AAAAJ&hl=en){: target="_blank" rel="noopener"}
