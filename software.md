---
layout: default
title: Software
permalink: /software/
description: >-
  Open-source tools and scientific software by Dr Cyril Pernet —
  M/EEG analysis, fMRI pipelines, BIDS, and open neuroimaging infrastructure.
---

<div class="container">
<div class="sw-page">

<!-- ── PAGE HEADER ──────────────────────────────────────────── -->
<header class="page-header">
  <h1>Software &amp; Projects</h1>
  <p class="page-subtitle">
    Open-source tools for the neuroimaging and statistics community.
    All code is freely available — use it, cite it, contribute to it.
  </p>
</header>

<!-- ── M/EEG ANALYSIS ───────────────────────────────────────── -->
<section class="sw-group">
  <h2 class="sw-group-title">M/EEG Analysis</h2>
  <ul class="sw-list" role="list">

    <li class="sw-card">
      <div class="sw-card-head">
        <h3 class="sw-card-title">
          <a href="https://limo-eeg-toolbox.github.io/limo_meeg/" target="_blank" rel="noopener">LIMO MEEG</a>
        </h3>
        <span class="sw-status sw-status--active">Active</span>
      </div>
      <p class="sw-card-full">LInear MOdelling for M/EEG data</p>
      <p class="sw-card-desc">
        MATLAB toolbox running on top of EEGLAB and FieldTrip for mass-univariate
        statistical modelling of M/EEG data. Supports hierarchical GLMs, robust
        estimation, bootstrapping, and multiple comparisons correction across the
        full sensor space and time domain.
      </p>
      <div class="sw-tags">
        <span class="sw-tag">MATLAB</span>
        <span class="sw-tag">EEGLAB</span>
        <span class="sw-tag">FieldTrip</span>
        <span class="sw-tag">Statistics</span>
        <span class="sw-tag">M/EEG</span>
      </div>
      <div class="sw-links">
        <a href="https://github.com/LIMO-EEG-Toolbox/limo_meeg" target="_blank" rel="noopener">GitHub</a>
        <a href="https://limo-eeg-toolbox.github.io/limo_meeg/" target="_blank" rel="noopener">Documentation</a>
      </div>
    </li>

    <li class="sw-card">
      <div class="sw-card-head">
        <h3 class="sw-card-title">
          <a href="https://github.com/sccn/bids-matlab-tools" target="_blank" rel="noopener">BIDS-MATLAB Tools</a>
        </h3>
        <span class="sw-status sw-status--active">Active</span>
      </div>
      <p class="sw-card-full">EEGLAB BIDS import / export</p>
      <p class="sw-card-desc">
        EEGLAB plugin for importing and exporting EEG data in BIDS format.
        Enables fully reproducible EEG pipelines from raw data organisation
        to group-level analysis, with support for BIDS metadata and provenance.
      </p>
      <div class="sw-tags">
        <span class="sw-tag">MATLAB</span>
        <span class="sw-tag">EEGLAB</span>
        <span class="sw-tag">BIDS</span>
        <span class="sw-tag">EEG</span>
      </div>
      <div class="sw-links">
        <a href="https://github.com/sccn/bids-matlab-tools" target="_blank" rel="noopener">GitHub</a>
        <a href="https://www.youtube.com/playlist?list=PLXc9qfVbMMN3II4EnVQNjOeVl-UprWlnM" target="_blank" rel="noopener">Video tutorials</a>
      </div>
    </li>

    <li class="sw-card">
      <div class="sw-card-head">
        <h3 class="sw-card-title">
          <a href="https://github.com/bids-standard/bep021" target="_blank" rel="noopener">BIDS EEG Derivatives (BEP021)</a>
        </h3>
        <span class="sw-status sw-status--standard">Standard</span>
      </div>
      <p class="sw-card-full">BIDS extension for electrophysiology derivatives</p>
      <p class="sw-card-desc">
        Co-leading the BIDS Enhancement Proposal for storing and sharing
        EEG pre-processing and analysis derivatives. Defines a community
        standard for derivative file naming, folder structure, and metadata.
      </p>
      <div class="sw-tags">
        <span class="sw-tag">BIDS</span>
        <span class="sw-tag">EEG</span>
        <span class="sw-tag">Standard</span>
      </div>
      <div class="sw-links">
        <a href="https://github.com/bids-standard/bep021" target="_blank" rel="noopener">GitHub</a>
        <a href="https://bids.neuroimaging.io/get_involved.html" target="_blank" rel="noopener">Get involved</a>
      </div>
    </li>

  </ul>
</section>

<!-- ── fMRI ANALYSIS ─────────────────────────────────────────── -->
<section class="sw-group">
  <h2 class="sw-group-title">fMRI Analysis</h2>
  <ul class="sw-list" role="list">

    <li class="sw-card">
      <div class="sw-card-head">
        <h3 class="sw-card-title">
          <a href="https://github.com/CPernet/spmup" target="_blank" rel="noopener">SPMup</a>
        </h3>
        <span class="sw-status sw-status--active">Active</span>
      </div>
      <p class="sw-card-full">SPM Utility Plus</p>
      <p class="sw-card-desc">
        MATLAB library extending SPM for fMRI quality assurance, data boosting,
        and adaptive thresholding. Provides additional diagnostic plots, robust
        GLM options, and utilities to get more out of standard SPM analyses.
      </p>
      <div class="sw-tags">
        <span class="sw-tag">MATLAB</span>
        <span class="sw-tag">SPM</span>
        <span class="sw-tag">fMRI</span>
        <span class="sw-tag">QA</span>
      </div>
      <div class="sw-links">
        <a href="https://github.com/CPernet/spmup" target="_blank" rel="noopener">GitHub</a>
      </div>
    </li>

  </ul>
</section>

<!-- ── INFRASTRUCTURE & STANDARDS ───────────────────────────── -->
<section class="sw-group">
  <h2 class="sw-group-title">Infrastructure &amp; Standards</h2>
  <ul class="sw-list" role="list">

    <li class="sw-card">
      <div class="sw-card-head">
        <h3 class="sw-card-title">
          <a href="https://openneuropet.github.io/" target="_blank" rel="noopener">OpenNeuroPET</a>
        </h3>
        <span class="sw-status sw-status--active">Active</span>
      </div>
      <p class="sw-card-full">Open PET neuroimaging — BIDS, tools &amp; pipelines</p>
      <p class="sw-card-desc">
        International collaboration to standardise PET neuroimaging data using
        BIDS, develop open analysis pipelines, and facilitate PET data sharing.
        Includes BIDS-PET specification work, conversion tools, and a public
        dataset repository.
      </p>
      <div class="sw-tags">
        <span class="sw-tag">Python</span>
        <span class="sw-tag">PET</span>
        <span class="sw-tag">BIDS</span>
        <span class="sw-tag">Data sharing</span>
      </div>
      <div class="sw-links">
        <a href="https://openneuropet.github.io/" target="_blank" rel="noopener">Website</a>
        <a href="https://github.com/openneuropet" target="_blank" rel="noopener">GitHub</a>
      </div>
    </li>

    <li class="sw-card">
      <div class="sw-card-head">
        <h3 class="sw-card-title">
          <a href="https://public-neuro.github.io/" target="_blank" rel="noopener">PublicnEUro</a>
        </h3>
        <span class="sw-status sw-status--platform">Platform</span>
      </div>
      <p class="sw-card-full">EU platform for open neuroimaging data</p>
      <p class="sw-card-desc">
        European platform for sharing and accessing brain imaging data,
        developed within an EU consortium. Provides curated datasets,
        federated access, and infrastructure for large-scale collaborative
        neuroimaging research.
      </p>
      <div class="sw-tags">
        <span class="sw-tag">Data sharing</span>
        <span class="sw-tag">BIDS</span>
        <span class="sw-tag">EU</span>
        <span class="sw-tag">Platform</span>
      </div>
      <div class="sw-links">
        <a href="https://public-neuro.github.io/" target="_blank" rel="noopener">Website</a>
      </div>
    </li>

    <li class="sw-card">
      <div class="sw-card-head">
        <h3 class="sw-card-title">
          <a href="https://open-brain-consent.readthedocs.io/en/stable/" target="_blank" rel="noopener">Open Brain Consent</a>
        </h3>
        <span class="sw-status sw-status--standard">Standard</span>
      </div>
      <p class="sw-card-full">GDPR-compliant consent templates for open neuroimaging</p>
      <p class="sw-card-desc">
        Community resource providing ready-to-use informed consent templates
        that comply with GDPR and allow participants' data to be shared openly.
        Maintained by an international group; available in multiple languages.
      </p>
      <div class="sw-tags">
        <span class="sw-tag">Open science</span>
        <span class="sw-tag">GDPR</span>
        <span class="sw-tag">Ethics</span>
      </div>
      <div class="sw-links">
        <a href="https://open-brain-consent.readthedocs.io/en/stable/" target="_blank" rel="noopener">Documentation</a>
        <a href="https://github.com/con/open-brain-consent" target="_blank" rel="noopener">GitHub</a>
      </div>
    </li>

  </ul>
</section>

<!-- ── FOOTER CTA ────────────────────────────────────────────── -->
<div class="sw-cta">
  <p>
    All repositories, issues, and contribution guidelines are on
    <a href="https://github.com/CPernet" target="_blank" rel="noopener">GitHub&nbsp;&rarr;</a>
  </p>
  <p class="sw-cta-sub">
    Interested in collaborating on any of these projects?
    <a href="/contact/">Get in touch.</a>
  </p>
</div>

</div><!-- /.sw-page -->
</div><!-- /.container -->
