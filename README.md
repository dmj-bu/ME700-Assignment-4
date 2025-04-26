# ME700-Assignment-4

## Overview

This repository contains three separate studies for Assignment 4, each demonstrating key aspects of finite element analysis (FEA) using FEniCSx:

- Part A: Analytical Comparison
- Part B: Mesh Refinement Study
- Part C: Code Failure Demonstration

Each part is presented as an individual Jupyter Notebook.

---

## File Structure

### Part A — Analytical Comparison

**File:** `linear_elastic_analytical_comparison.ipynb`

- Solves a linear elastic problem with known analytical solution.
- Compares numerical and analytical displacement fields.
  
---

### Part B — Mesh Refinement Study

**File:** `elasto-plastic_ph_refinement.ipynb`

- Solves an elasto-plastic beam bending problem.
- Performs both **h-refinement** (refining mesh size) and **p-refinement** (increasing polynomial degree).
- Tracks maximum displacement as the quantity of interest.
- Plots convergence of error with respect to mesh refinement and polynomial order.

---

### Part C — Code Failure Demonstration

**File:** `linear_elastic_failure_study.ipynb`

- Alters the mesh geometry to introduce highly skewed, nearly degenerate elements.
- Demonstrates a failure during variational form assembly due to ambiguous integration domains.

---

