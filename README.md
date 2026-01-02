# Wick
Wick is a model weight quantization library I am building to practice my skills in model optimization and Rust.

This README is mostly written for myself to keep track of the project's progress and to provide a reference for future work.

## Project structure
This repository contains both python frontend and Rust backend.

Python is responsible for loading model and exporting quantized version of the model.

Rust is responsible for the quantization of tensors.

PyO3 and Maturin are used to bridge Python and Rust: maturin is used to build the Rust backend into a dynamic link library, and PyO3 is used to expose the Rust backend to Python.

### Workflow and example commands

#### Testing the Rust library in Python environment

`maturin develop` to build the Rust backend into a dynamic link library and expose it to Python.

Add the function signatures in `./python/wick/__init__.pyi` to expose them to the editor.

## Roadmap

### MVP (by 2026-02-01)
- Core: Implement block-wise quantization algorithm in Rust.
- IO: Support safetensors format model input. Support gguf format model output.
- Testing: Write unit tests for the Rust backend. Write integration tests for backend-frontend integration.
- Documentation: Write comprehensive documentation for Wick.

## Resources

[Deep Learning Systems](https://dlsyscourse.org/)

Build from scratch a complete deep learning library (called “Needle” - Necessary Elements of Deep Learning), capable of efficient GPU-based operations, automatic differentiation, and support for parameterized layers, loss functions, data loaders, and optimizers.

[Tiny ML and Efficient Deep Learning](https://hanlab.mit.edu/courses/2024-fall-65940)

Model compression, pruning, quantization, neural architecture search, distributed training, data/model parallelism, gradient compression, and on-device fine-tuning.

[Bits and Bytes](./papers/bits_and_bytes.pdf)

The paper that introduced the Bits and Bytes block-wise quantization algorithm.

[The Rust Book (interactive)](https://rust-book.cs.brown.edu/)
