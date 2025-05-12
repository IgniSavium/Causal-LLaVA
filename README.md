# Causal-LLaVA: Causal Disentanglement for Mitigating Hallucination in Multimodal Large Language Models

[![Paper](https://img.shields.io/badge/arXiv.2501.xxxxx-orange )](https://arxiv.org/abs/... )
[![License](https://img.shields.io/github/license/yourname/Causal-LLaVA )](https://github.com/yourname/Causal-LLaVA/blob/main/LICENSE )

This is the official codebase for the paper:

> **Causal-LLaVA: Causal Disentanglement for Mitigating Hallucination in Multimodal Large Language Models**  
> Anonymous Author, NeurIPS 2025 Submission.

The code will be released shortly. Please stay tuned or star this repository to receive updates.

---

## 🔍 Overview

We propose a causality-driven framework to reduce hallucinations in multimodal large language models (MLLMs) by disentangling object representations across modalities through causal intervention.

- **Causal-Driven Projector**: Modifies visual feature projection to learn disentangled representations.
- **Causal Intervention Module**: Performs causal intervention in the LLM layer to remove spurious correlations.

---

## 📚 Paper Abstract

> Recent advances in Large Language Models (LLMs) have extended their capabilities to the multimodal domain, giving rise to Multimodal Large Language Models (MLLMs). While MLLMs perform well on tasks such as image captioning and visual question answering, they still suffer from *hallucination*—generating content inconsistent with visual inputs. A key form, *object hallucination*, arises when models describe nonexistent objects. This paper explores how dataset biases contribute to object hallucinations in MLLMs from a *representation learning* perspective. Our analysis reveals that training on biased datasets—where certain objects frequently co-occur—causes over-*entanglement* in their semantic representations across modalities. Consequently, the model may mistakenly activate the representation of an object that often appears with the input object but is actually absent, leading to hallucinations. To address this, we propose a causality-based disentanglement framework using causal intervention to deconfound the biases introduced by frequent object co-occurrences in the data. Specifically, we introduce a Causal-Driven Projector in the visual pathway and a Causal Intervention Module in the final LLM transformer layer, working together to mitigate spurious correlations. Visualization analyses confirm improved representation separability, and experiments show notable reduction in hallucinations while preserving strong performance across multiple comprehension benchmarks.

---

## 📁 Repository Structure (TBD)

Once the code is released, the structure will include:

---

## 📄 License

This project is released under the [MIT License](LICENSE).

