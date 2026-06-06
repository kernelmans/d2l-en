# Deep Learning Anatomy Roadmap

This repository is a personal fork of `d2l-ai/d2l-en`, the Dive into Deep Learning book.

The goal of this file is not to replace the book. It is a personal reading map for studying deep learning as computational anatomy: tensors, shapes, transformations, losses, gradients, modules, optimizers, and architectures.

## Central method

For every chapter, ask the same anatomical questions:

- What enters the system?
- What is the shape of the input?
- What transformation happens?
- Which parameters are learned?
- What loss measures the error?
- How does the gradient flow backward?
- What does the optimizer change?
- What would break if this component were removed?

## Phase 1: Tensor and mathematical matter

Start with:

- `chapter_preliminaries`
- `chapter_appendix-mathematics-for-deep-learning`

Study:

- tensors
- shapes
- broadcasting
- matrix multiplication
- calculus
- probability
- automatic differentiation

Goal:

Understand the raw material of deep learning before studying models.

## Phase 2: Linear models

Study:

- `chapter_linear-regression`
- `chapter_linear-classification`

Focus on:

- features
- weights
- bias
- logits
- softmax
- cross-entropy
- gradient descent

Core intuition:

A model is a parameterized transformation. Training means modifying parameters to reduce a loss.

## Phase 3: Multilayer perceptrons

Study:

- `chapter_multilayer-perceptrons`

Focus on:

- hidden layers
- activation functions
- nonlinear transformations
- overfitting
- dropout
- initialization

Bridge to micrograd:

Micrograd shows the scalar mechanism of backpropagation. This chapter shows the same mechanism at tensor scale.

## Phase 4: PyTorch builder's guide

Study:

- `chapter_builders-guide`

Focus on:

- `nn.Module`
- `forward()`
- parameters
- custom layers
- sequential models
- saving and loading parameters
- GPU devices

Architecture lens:

Read PyTorch as a component system: modules, state, execution path, parameters, and reusable blocks.

## Phase 5: Optimization anatomy

Study:

- `chapter_optimization`

Focus on:

- learning rate
- minibatch stochastic gradient descent
- momentum
- Adam
- scheduling
- convergence
- stability

Core intuition:

Architecture gives the model a body. Optimization gives it a learning dynamics.

## Phase 6: Convolutional neural networks

Study selectively:

- `chapter_convolutional-neural-networks`
- `chapter_convolutional-modern`

Focus on:

- convolution filters
- local patterns
- feature maps
- pooling
- hierarchical visual abstraction

Transferable idea:

Deep learning learns layered representations: simple local structures become complex global structures.

## Phase 7: Sequence models

Study selectively:

- `chapter_recurrent-neural-networks`
- `chapter_recurrent-modern`

Focus on:

- sequence modeling
- hidden state
- recurrent computation
- long-range dependency problems

Bridge to Transformers:

RNNs show why attention became such a major architectural rupture.

## Phase 8: Attention and Transformers

Study carefully:

- `chapter_attention-mechanisms-and-transformers`

Focus on:

- queries
- keys
- values
- attention scores
- softmax attention weights
- multi-head attention
- positional encodings
- Transformer blocks

Bridge to LLMs:

This is the conceptual bridge from general deep learning to GPT-like architectures.

## Phase 9: NLP and pretraining

Study:

- `chapter_natural-language-processing-pretraining`
- `chapter_natural-language-processing-applications`

Focus on:

- tokenization
- embeddings
- language modeling
- pretraining
- fine-tuning
- downstream tasks

Connect with:

- `kernelmans/LLMs-from-scratch`
- `kernelmans/gpt2_transformer_lab`
- `kernelmans/micrograd`

## Personal repository ecosystem

Use this fork as the broad foundation:

- `micrograd`: minimal scalar backpropagation anatomy
- `d2l-en`: broad deep learning foundation
- `LLMs-from-scratch`: GPT and LLM construction from scratch
- `gpt2_transformer_lab`: inspection of pretrained GPT-2 mechanisms

## First local commands

```bash
git clone --depth 1 https://github.com/kernelmans/d2l-en.git
cd d2l-en
code .
```

If already cloned:

```bash
git pull origin master
```

## First study target

Start with `chapter_preliminaries`, then move to `chapter_linear-regression` and `chapter_linear-classification`.

Do not try to finish the whole book. Build a personal anatomical notebook for each key concept.
