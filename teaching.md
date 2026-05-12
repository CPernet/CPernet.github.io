---
layout: page
title: Teaching & Resources
permalink: /teaching/
description: >-
  Lecture slides, tutorials, and open educational resources in fMRI, M/EEG,
  statistics, and digital signal processing — by Dr Cyril Pernet.
---

Lecture slides, tutorials, and notes developed over 15+ years of teaching at
summer schools, workshop courses, and university courses. All slides are freely
available. Topics span fMRI design and analysis, M/EEG signal processing and
statistics, general linear modelling, robust statistics, and digital signal
processing.

---

## fMRI

Most of what you need to think about for fMRI is described in the OHBM
[MRI COBIDAS](https://www.biorxiv.org/content/10.1101/054262v2){: target="_blank"}.
Understanding the subtleties of design, QA, and statistics requires some basic
knowledge of how MRI works — see the
[MRI physics lecture](#foundations) in the SPM archive below.

### Standalone Lectures

- [Tissue Segmentation](https://drive.google.com/file/d/0B5CS2-RHxR96d2FpeGpuMWlEejg/view?usp=sharing){: target="_blank"}
- [fMRI Preprocessing](https://drive.google.com/file/d/1i85D-1me4ufZO2KDkJa2TZU2plqm1O9n/view?usp=sharing){: target="_blank"}
- [fMRI Statistics and Inference](https://drive.google.com/file/d/1rLRagQAcDN0n4VVm-_x-ATAoco01vD_b/view?usp=sharing){: target="_blank"}
- [fMRI Connectivity](https://drive.google.com/file/d/17FhMKOsqsUAx1TBEOwfoFbiiMUlZ-8tv/view?usp=sharing){: target="_blank"}

### Experimental Design Notes

Functional MRI designs are more constrained than behavioural experiments because we
must think simultaneously about experimental effects and fMRI acquisition parameters.
Three broad families of design exist:

**Block designs** are powerful for detection — localising regions that respond
differentially to a condition. Blocks of ~16 s optimise the trade-off between BOLD
signal strength and noise at typical acquisition frequencies; shorter blocks do not
allow the response to return to baseline, reducing contrast.

**Event-related designs** allow estimation of the haemodynamic response shape per
condition and support random stimulus ordering, but have lower detection power than
blocks. Rapid event-related designs introduce null events to create differential ISI
overlap, enabling a full characterisation of the response.

**Mixed designs** combine block-level state effects with event-level transient
responses, enabling study of processes that operate at different timescales.

**Adaptation (carry-over) designs** exploit the BOLD refractory period: if two
successive stimuli differ in a property coded by a region, the response to the
second stimulus is less suppressed (fMRI-A). The linear BOLD assumption holds for
ISI ≥ 4 s; continuous carry-over designs (Aguirre, *NeuroImage*, 2007) extend this
to continuously changing stimulus streams.

See also: [Design Optimisation](https://drive.google.com/file/d/1kQf2R1cyMYuFzyPACDZJQBHna_W6Ew4n/view?usp=sharing){: target="_blank"}
(in the SPM archive below).

### Quality Assurance

Quantifying noise at every stage of the processing workflow is essential for
reliable results. See:

- Blog: [Cleaning BOLD time series](https://neurostatscyrilpernet.blogspot.com/2016/12/clean-bold-for-better-stats.html){: target="_blank"} — sources of noise and how to measure them
- Blog: [Minimising motion artefacts](https://neurostatscyrilpernet.blogspot.com/2016/12/minimizing-motion-artefacts-maybe.html){: target="_blank"} — motion metrics and mitigation strategies
- [SPMup on GitHub](https://github.com/CPernet/spmup){: target="_blank"} — MATLAB toolbox that automates QA and incorporates noise measures into analyses

### Statistics Notes

**Adaptive thresholding for single-subject maps.**
The distribution of voxel-level statistics at the subject level is not always
central (mean ≠ 0). Standard fixed thresholds therefore over- or under-estimate
effects. Modelling the distribution as a mixture of a non-central Gaussian (null)
and positive/negative Gamma distributions for the tails identifies the crossing
point that optimally separates signal from noise.
[Paper](https://www.frontiersin.org/articles/10.3389/fnhum.2012.00245/full){: target="_blank"} ·
[SPMup code](https://github.com/CPernet/spmup/tree/master/adaptative_threshold){: target="_blank"}

**Boosting beta estimates with HRF derivatives.**
When the haemodynamic response function is mis-specified (e.g. onset earlier than
the standard model), adding the temporal derivative improves model fit but leaves
the amplitude estimate biased. A correction step recovers the true amplitude.
[Tutorial paper](https://www.frontiersin.org/articles/10.3389/fnins.2014.00001/full){: target="_blank"} ·
[SPMup code](https://github.com/CPernet/spmup/tree/master/hrf){: target="_blank"}

**Computing percentage signal change.**
After convolving regressors with the HRF, the design matrix does not scale to 1,
so raw beta values are not interpretable as % BOLD change without rescaling. The
choice of reference (maximum of a single trial, maximum of the design) is arbitrary
but must be consistent.
[Tutorial paper](https://www.frontiersin.org/articles/10.3389/fnins.2014.00001/full){: target="_blank"} ·
[SPMup code](https://github.com/CPernet/spmup/blob/master/utlilities/spmup_psc.m){: target="_blank"}

---

## M/EEG

Most of what you need for reproducible M/EEG research is described in the
[OHBM COBIDAS MEEG guidelines](https://cobidasmeeg.wordpress.com/){: target="_blank"}.

### Standalone Lectures

- [Hierarchical Linear Modelling for EEG](https://drive.google.com/file/d/1ERoVhBYmY0GvVHfWVBtqTgutC6y92-xN/view?usp=sharing){: target="_blank"}
- [LIMO Contrasts](https://drive.google.com/file/d/151mboLQ3rmOPp5qQuwB9haJbjuTDQulk/view?usp=sharing){: target="_blank"}
- [Multiple Comparisons Correction](https://drive.google.com/file/d/1Ju-86hqUaZSGiQ_aw9-UeMo6CSm0w7Tb/view?usp=sharing){: target="_blank"}

### Video Tutorials

A full video tutorial series on EEG data organisation and analysis with EEGLAB and
BIDS is available on YouTube:

- [EEGLAB BIDS Tutorial Series](https://www.youtube.com/playlist?list=PLXc9qfVbMMN3II4EnVQNjOeVl-UprWlnM){: target="_blank"} — playlist covering BIDS import/export, metadata, and group analysis

### Statistics Notes

**LIMO MEEG** provides full sensor- or source-space hierarchical GLM for EEG and
MEG data, 100% compatible with EEGLAB (GUI integration via the STUDY framework)
and FieldTrip. See [LIMO on GitHub](https://github.com/LIMO-EEG-Toolbox/limo_meeg){: target="_blank"}
and the [Software page](/software/) for details.

**Multiple comparison corrections for M/EEG** must account for the massive number
of simultaneous tests across sensors, sources, and time. As part of LIMO,
bootstrap-based corrections and cluster-level inference have been developed
specifically for the M/EEG sensor–time space.
See: [Blog post on bootstrapping and multiple comparisons](https://neurostatscyrilpernet.blogspot.com/2019/06/bootstrapping-and-multiple-comparisons.html){: target="_blank"}

---

## Statistics & Signal Processing

### General Linear Modelling

Neuroimaging is built on linear algebra: understanding matrix operations directly
gives you the GLM, regression, ANOVA, and most of inferential statistics.

- [Introduction to vectors](https://drive.google.com/file/d/1F3CspBl8MOHN3Q97beQgCo3acGpQ7ynH/view?usp=sharing){: target="_blank"} — the connection between geometry and algebra
- [Basic matrix operations](https://drive.google.com/file/d/1GDAtbSf1ZH6GYcpPmIDwDEMkgRnGFRQy/view?usp=sharing){: target="_blank"} — a short review covering all the essentials
- [MIT Linear Algebra (18.06)](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/){: target="_blank"} — the gold-standard free course (Gilbert Strang)
- [MATLAB GLM page](/external/glm/GLM_lectures.html) — interactive worked examples: simple regression, multiple regression, ANOVA via matrix inversion
- [What makes a contrast orthogonal?](https://drive.google.com/file/d/1Huk0sSUGnxc8nn_7Qyv4GuyQwTJSiu2v/view?usp=sharing){: target="_blank"} — a short write-up on linear independence and orthogonality in the GLM
- [Linear independence, orthogonality and correlation](https://www.jstor.org/stable/2683250){: target="_blank"} — a classic one-page article

### Robust Statistics

Ordinary Least Squares has a 0% breakdown point — a single outlier can dominate
the solution. Trimmed means, robust correlations, and tests based on Winsorised
estimates provide reliable alternatives that still have good power under normality.

- [Robust Correlations toolbox](https://github.com/CPernet/Robust-Correlations){: target="_blank"} — Pearson alternatives with confidence intervals; skipped, percentage-bend, and Spearman correlations
- [Robust Statistical Toolbox](https://github.com/CPernet/Robust_Statistical_Toolbox){: target="_blank"} — tests based on robust estimators for one-sample, two-sample, and factorial designs

### Digital Signal Processing

Fourier analysis underlies most of what we do in signal processing, from band-pass
filtering EEG to characterising the BOLD noise spectrum.

- [Fourier Analysis — lecture notes with MATLAB code](https://drive.google.com/file/d/1pNByEl9jKSHRKWbMfK_fYCOjB2FRfNqF/view?usp=sharing){: target="_blank"} — 1D and 2D Fourier transforms, filtering, practical applications

---

## SPM Edinburgh Course Archive (2010–2019)
{: #spm-archive}

A curated archive of lectures from the Edinburgh SPM course, run annually from
2010 to 2019. Slides cover the full analysis pipeline from MRI physics to
Bayesian inference.

**Thank you** to all colleagues who gave lectures over the years:
Dr Devasuda Anblagan, Dr John Ashburner, Dr Roselyne Chauvin, Dr Ian Charest,
Dr Justin Chumbley, Dr Jean Daunizeau, Dr Christian Gaser, Dr Nikolaus Kriegeskorte,
Dr Daniele Marinazzo, Dr Martin McFarquhar, Dr Alexa Morcom, Dr Thomas Nichols,
Dr Jean-Baptiste Poline, Dr Christophe Phillips, Dr Mohamed Seghier, Dr Jason Taylor
& Dr Thomas Wolpers.

**Thank you** to the hundreds of colleagues and students who attended the course.

### Foundations
{: #foundations}

- [MRI physics: what are we measuring?](https://drive.google.com/file/d/1fY3zgHCAwKK3OYXdnOH83_13OppvG-_H/view?usp=sharing){: target="_blank"}
- [The BOLD signal](https://drive.google.com/file/d/1NKSyBIge7lDRt6TLyunY9kfgAQ_PwoeJ/view?usp=sharing){: target="_blank"}
- [A quick introduction to SPM](https://drive.google.com/file/d/1gm7Gyebr-ySXxu6y3u6hk0qkKYZmObcf/view?usp=sharing){: target="_blank"}

### Preprocessing

- [Slice timing correction (within-subject)](https://drive.google.com/file/d/1-cuDjwZGRwtcyeU0lvyd0qBwVUmLHRXW/view?usp=sharing){: target="_blank"}
- [Realignment and coregistration (within-subject)](https://drive.google.com/file/d/1ijY-cQ6zKaciIac3gTSneAjFlmsuO1Bz/view?usp=sharing){: target="_blank"}
- [Normalisation and smoothing (between-subject)](https://drive.google.com/file/d/15M4_QW39A--R9T6HEC-yIs9MPb3aOBCh/view?usp=sharing){: target="_blank"}
- [Morphometry: volumes](https://drive.google.com/file/d/1K5-rJK-ZzBpx6Cv8oWMdc7RBI7E_CM9h/view?usp=sharing){: target="_blank"}
- [Morphometry: surfaces](https://drive.google.com/file/d/1Re9LLgrsXrZibXl58rECHUKC9qjg7kRK/view?usp=sharing){: target="_blank"}

### Experimental & fMRI Designs

- [Experimental design](https://drive.google.com/file/d/1nSvKpmbnlx741DUFxLGsVoLOa6rFL6Hy/view?usp=sharing){: target="_blank"}
- [Design optimisation](https://drive.google.com/file/d/1kQf2R1cyMYuFzyPACDZJQBHna_W6Ew4n/view?usp=sharing){: target="_blank"}

### Univariate Statistical Modelling

- [The General Linear Model](https://drive.google.com/file/d/1KYl2kAoGHFPt3xFtB-otgCF0a2zeKFSm/view?usp=sharing){: target="_blank"}
- [Random Effects Modelling](https://drive.google.com/file/d/1-fSplwph0LGErNJxKjA_6QHQoHf4VOoY/view?usp=sharing){: target="_blank"}
- [Contrasts](https://drive.google.com/file/d/1L_a_v-Xwug_8oGROb5-ApkHNf9yvKIF6/view?usp=sharing){: target="_blank"}
- [Non-parametric modelling](https://drive.google.com/file/d/1jGPjx6ViGp3tK23mIZOVo4_TBYxRAXbI/view?usp=sharing){: target="_blank"}

### Multivariate Statistical Modelling

- [Pattern Recognition](https://drive.google.com/file/d/1ebD8th-4HfY8h7ZvF1WYcOyIV6hbzjSm/view?usp=sharing){: target="_blank"}
- [Representational Similarity Analysis](https://drive.google.com/file/d/1bcGEtxDdghnm8b9vDpjxUhiGmeu0aQmv/view?usp=sharing){: target="_blank"}

### Statistical Inference & Visualisation

- [Designs and inference for fMRI](https://drive.google.com/file/d/1vpWJiF5xyAqXey6do4rSKkZZdm5Xfn9I/view?usp=sharing){: target="_blank"}
- [Inference in univariate and multivariate models](https://drive.google.com/file/d/19HGmkR6QBH2y2ClbDIVl3pz08hIRLVRK/view?usp=sharing){: target="_blank"}
- [Multiple comparisons correction](https://drive.google.com/file/d/1boJH_yY8NRHtaMJRvWExYX5uAzJg1FLK/view?usp=sharing){: target="_blank"}
- [Multiple comparisons correction, levels of inference, circularity](https://drive.google.com/file/d/1Lvt2jbXhw5KjqXmpJwv4rrulW6ZhZxUl/view?usp=sharing){: target="_blank"}
- [Inference and result visualisation](https://drive.google.com/file/d/12uq_ez2fFBtvZjypQP7y8A98cRMXFiHF/view?usp=sharing){: target="_blank"}

### Bayesian Modelling & Inference

- [Bayes probability, modelling and use in SPM](https://drive.google.com/file/d/1oelbMadFxWRwD_aSjqAIyMzMuYN900Xb/view?usp=sharing){: target="_blank"}
- [Bayesian inference: model comparison and model selection](https://drive.google.com/file/d/19Wm2pK3Ks8b9ykQ--C28gmNsJDRmJrkK/view?usp=sharing){: target="_blank"}
- [Multivariate Bayes for decoding](https://drive.google.com/file/d/1Qlt0RHs-KSVZ0ImsEgHclKlu6W_BkhA9/view?usp=sharing){: target="_blank"}
