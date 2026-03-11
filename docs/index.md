---
hide:
  - navigation
  - toc
---

<div class="hero">

<div class="hero-logo">
<svg viewBox="0 0 580 220" xmlns="http://www.w3.org/2000/svg" aria-label="Canopée">
  <!-- Two-panel flat layout -->
  <rect width="580" height="220" fill="transparent"/>

  <!-- Left panel accent bar -->
  <rect x="0" y="0" width="178" height="3" fill="#3fb85e"/>

  <!-- ── Flat canopy icon ── -->
  <!-- Arc 4 - outermost -->
  <path d="M 16 180 Q 89 28 162 180"
        fill="none" stroke="#1a4726" stroke-width="2" stroke-linecap="butt"/>
  <!-- Arc 3 -->
  <path d="M 30 180 Q 89 50 148 180"
        fill="none" stroke="#245e34" stroke-width="2.5" stroke-linecap="butt"/>
  <!-- Arc 2 -->
  <path d="M 48 180 Q 89 76 130 180"
        fill="none" stroke="#32864a" stroke-width="3" stroke-linecap="butt"/>
  <!-- Arc 1 — brightest -->
  <path d="M 68 180 Q 89 104 110 180"
        fill="none" stroke="#3fb85e" stroke-width="3.5" stroke-linecap="butt"/>
  <!-- Trunk -->
  <line x1="89" y1="180" x2="89" y2="108"
        stroke="#245e34" stroke-width="2" stroke-linecap="butt"/>
  <!-- Base bar -->
  <line x1="10" y1="180" x2="168" y2="180"
        stroke="#1a4726" stroke-width="1.5"/>
  <!-- Tick marks -->
  <rect x="14"  y="174" width="2.5" height="8" fill="#1a4726"/>
  <rect x="28"  y="174" width="2.5" height="8" fill="#245e34"/>
  <rect x="46"  y="174" width="2.5" height="8" fill="#32864a"/>
  <rect x="66"  y="174" width="2.5" height="8" fill="#3fb85e"/>
  <rect x="108" y="174" width="2.5" height="8" fill="#3fb85e"/>
  <rect x="128" y="174" width="2.5" height="8" fill="#32864a"/>
  <rect x="146" y="174" width="2.5" height="8" fill="#245e34"/>
  <rect x="160" y="174" width="2.5" height="8" fill="#1a4726"/>
  <!-- Apex — flat square -->
  <rect x="83" y="24" width="12" height="12" fill="#3fb85e"/>
  <rect x="86" y="27" width="6"  height="6"  fill="#0b1f10"/>

  <!-- Vertical rule divider -->
  <rect x="190" y="0" width="1.5" height="220" fill="#1a4726"/>

  <!-- ── Wordmark ── -->
  <!-- "canop" -->
  <text x="210" y="120"
        font-family="Georgia, 'Times New Roman', serif"
        font-size="96" font-weight="400" letter-spacing="-2"
        fill="#d4edd8">canop</text>
  <!-- "é" in green -->
  <text x="430" y="120"
        font-family="Georgia, 'Times New Roman', serif"
        font-size="96" font-weight="400" letter-spacing="-2"
        fill="#3fb85e">&#233;</text>
  <!-- "e" back to white -->
  <text x="474" y="120"
        font-family="Georgia, 'Times New Roman', serif"
        font-size="96" font-weight="400" letter-spacing="-2"
        fill="#d4edd8">e</text>

  <!-- Rule under wordmark: two-tone -->
  <rect x="210" y="133" width="420" height="2.5" fill="#1a4726"/>
  <rect x="210" y="133" width="100" height="2.5" fill="#3fb85e"/>

  <!-- Tagline -->
  <text x="212" y="162"
        font-family="'Courier New', 'Lucida Console', monospace"
        font-size="11" letter-spacing="3.5"
        fill="#32864a">TYPE-SAFE PYTHON CONFIGURATION</text>
</svg>
</div>

<p>
  Composition, computed fields, discriminated unions,<br>
  and a sweep engine — all in pure Python.
</p>

<div class="hero-buttons">
  <a href="getting-started/installation/" class="hero-btn hero-btn--primary">Get Started →</a>
  <a href="getting-started/quickstart/" class="hero-btn hero-btn--secondary">Quickstart</a>
</div>

</div>

```python
from canopee import ConfigBase
from pydantic import Field, computed_field

class TrainingConfig(ConfigBase):
    lr: float        = Field(default=1e-3, gt=0, le=1)
    epochs: int      = 10
    batch_size: int  = 128

    @computed_field
    @property
    def warmup_steps(self) -> int:
        return self.epochs * 100

cfg  = TrainingConfig()
fast = cfg | {"lr": 3e-4, "epochs": 5}   # immutable merge — new instance
fast.warmup_steps                          # → 500, auto-recomputed
```

---

## Why Canopée?

<div class="feature-grid">

<div class="feature-card">
<span class="feature-card__icon">🔒</span>
<div class="feature-card__title">Immutable by default</div>
<div class="feature-card__desc">
Every config is a frozen Pydantic model. Instances are safe to cache, hash, and share across threads without defensive copying.
</div>
</div>

<div class="feature-card">
<span class="feature-card__icon">⚡</span>
<div class="feature-card__title">Computed fields</div>
<div class="feature-card__desc">
Declare derived values as <code>@computed_field</code> properties — pure Pydantic v2, no magic, full IDE support and serialisation.
</div>
</div>

<div class="feature-card">
<span class="feature-card__icon">🧩</span>
<div class="feature-card__title">Discriminated unions</div>
<div class="feature-card__desc">
Model optimizer/scheduler/architecture variants as tagged unions. Pydantic dispatches to the right class automatically — type-safe and serialisable.
</div>
</div>

<div class="feature-card">
<span class="feature-card__icon">🔗</span>
<div class="feature-card__title">Elegant composition</div>
<div class="feature-card__desc">
Override any field at any depth with <code>cfg | {"optimizer.lr": 3e-4}</code>. Every override re-runs all validators and recomputes all derived fields.
</div>
</div>

<div class="feature-card">
<span class="feature-card__icon">🏗️</span>
<div class="feature-card__title">Type-safe Evolve</div>
<div class="feature-card__desc">
Use <code>cfg.evolve(lr=3e-4)</code> for an IDE-autocomplete-friendly, type-checked way to create a modified copy of your config without any metaclass magic.
</div>
</div>

<div class="feature-card">
<span class="feature-card__icon">🎯</span>
<div class="feature-card__title">Sweep engine</div>
<div class="feature-card__desc">
Define distributions over config fields, generate variants via grid, random, or Optuna strategies — with a single, consistent API.
</div>
</div>

<div class="feature-card">
<span class="feature-card__icon">⌨️</span>
<div class="feature-card__title">CLI Reflection</div>
<div class="feature-card__desc">
Decorate any function with <code>@clify</code> to generate a full CLI from your config. Supports nested models via dot-paths and multiple backends (Argparse, Click, Typer).
</div>
</div>

</div>

---

## Canopée vs Hydra

<table class="comparison-table">
<thead>
<tr>
  <th>Feature</th>
  <th>Canopée</th>
  <th>Hydra</th>
</tr>
</thead>
<tbody>
<tr>
  <td>Config language</td>
  <td><span class="yes">Pure Python</span></td>
  <td><span class="no">YAML-first</span></td>
</tr>
<tr>
  <td>Type safety</td>
  <td><span class="yes">Pydantic v2 — full</span></td>
  <td><span class="partial">Structured Configs — verbose</span></td>
</tr>
<tr>
  <td>Derived fields</td>
  <td><span class="yes"><code>@computed_field</code></span></td>
  <td><span class="partial">String interpolation <code>${field}</code></span></td>
</tr>
<tr>
  <td>Immutability</td>
  <td><span class="yes">Frozen by default</span></td>
  <td><span class="no">Mutable OmegaConf</span></td>
</tr>
<tr>
  <td>IDE autocomplete</td>
  <td><span class="yes">Full — <code>evolve()</code> with native Pydantic</span></td>
  <td><span class="partial">Limited</span></td>
</tr>
<tr>
  <td>Discriminated unions</td>
  <td><span class="yes">Native Pydantic discriminator</span></td>
  <td><span class="no">Manual type tags</span></td>
</tr>
<tr>
  <td>Sweep support</td>
  <td><span class="yes">Grid / Random / Optuna built-in</span></td>
  <td><span class="partial">Basic multirun only</span></td>
</tr>
<tr>
  <td>CLI support</td>
  <td><span class="yes">Native <code>@clify</code> (Argparse/Click/Typer)</span></td>
  <td><span class="yes">First-class (Argparse-like)</span></td>
</tr>
<tr>
  <td>Serialisation</td>
  <td><span class="yes">JSON / TOML / Python dict</span></td>
  <td><span class="partial">YAML-tied</span></td>
</tr>
<tr>
  <td>Maintenance</td>
  <td><span class="yes">Active</span></td>
  <td><span class="no">Stalled</span></td>
</tr>
</tbody>
</table>

---

## Install

```bash
pip install canopee

# With Optuna sweep support
pip install canopee[optuna]

# Everything
pip install canopee[all]
```

**Requirements:** Python ≥ 3.11, Pydantic ≥ 2.6

---

## Dive in

<div class="feature-grid">

<div class="feature-card">
<span class="feature-card__icon">🚀</span>
<div class="feature-card__title"><a href="getting-started/quickstart/">5-minute quickstart</a></div>
<div class="feature-card__desc">Define your first config, use <code>evolve()</code>, and run a sweep in under 5 minutes.</div>
</div>

<div class="feature-card">
<span class="feature-card__icon">🔬</span>
<div class="feature-card__title"><a href="examples/mnist/">MNIST example</a></div>
<div class="feature-card__desc">Complete experiment config with multiple optimizers, schedulers, and model architectures — all type-safe.</div>
</div>

<div class="feature-card">
<span class="feature-card__icon">📘</span>
<div class="feature-card__title"><a href="api/core/">API Reference</a></div>
<div class="feature-card__desc">Complete reference for ConfigBase, ConfigStore, Sweep, and all distributions.</div>
</div>

</div>