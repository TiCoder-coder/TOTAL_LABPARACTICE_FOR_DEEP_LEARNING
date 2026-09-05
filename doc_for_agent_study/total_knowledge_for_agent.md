# Tổng Hợp Kiến Thức Deep Learning - Agent Knowledge Base

> **Ngày tạo:** 2026-07-24  
> **Agent:** Main Agent (Cursor)  
> **Workspace:** `/Users/ticoder-coder/Documents/DEEP_LEARNING/LAB&PRACTICE`

---

## Mục Lục

1. [Khóa Học (Courses)](#1-khóa-học-courses)
2. [PyTorch & Frameworks](#2-pytorch--frameworks)
3. [GitHub Repositories](#3-github-repositories)
4. [Blogs & Tutorials](#4-blogs--tutorials)
5. [Papers - Foundational](#5-papers---foundational)
6. [Papers - Transformer & Attention (2025)](#6-papers---transformer--attention-2025)
7. [Computer Vision Papers](#7-computer-vision-papers)
8. [Lộ Trình Học](#8-lộ-trình-học)

---

## 1. Khóa Học (Courses)

### 1.1 Beginner Level

| Khóa Học | Nhà Cung Cấp | Link | Ghi Chú |
|-----------|---------------|------|----------|
| Elements of AI | University of Helsinki | https://www.elementsofai.com/ | Khóa AI cho người không có nền tảng kỹ thuật, hoàn toàn miễn phí, có certificate |
| AI for Everyone | DeepLearning.AI | https://www.coursera.org/learn/ai-for-everyone | Khóa conceptual về AI cho leader |
| Google Intro to Generative AI | Google Skills | https://www.cloudskillsboost.google/paths/118 | Badge miễn phí |

### 1.2 Intermediate Level

| Khóa Học | Nhà Cung Cấp | Link | Ghi Chú |
|-----------|---------------|------|----------|
| Practical Deep Learning for Coders | fast.ai | https://course.fast.ai/ | Project-first, PyTorch, Hugging Face. 9 lessons. Miễn phí hoàn toàn |
| MIT 6.S191: Introduction to Deep Learning | MIT | https://introtodeeplearning.com/ | University-grade, updated hàng năm, từ NN đến LLM và agents |
| CS50 AI with Python | Harvard | https://cs50.harvard.edu/ai/ | Rigorous, university-level, free audit |
| Hugging Face LLM Course | Hugging Face | https://huggingface.co/learn/llm-course | Industry-relevant, free certificate |
| Hugging Face Agents Course | Hugging Face | https://huggingface.co/learn/agents-course | ~3-4 hrs/week, certificate miễn phí |
| Kaggle Learn | Kaggle | https://www.kaggle.com/learn | Micro-courses, clean path vào ML |

### 1.3 Advanced Level

| Khóa Học | Nhà Cung Cấp | Link | Ghi Chú |
|-----------|---------------|------|----------|
| Neural Networks: Zero to Hero | Andrej Karpathy | https://karpathy.ai/zero-to-hero.html | Build GPT từ scratch, hiểu sâu cách LLM hoạt động |
| CS229 Machine Learning | Stanford | https://cs229.stanford.edu/ | Graduate-level, YouTube, mathematical ML foundation |

### 1.4 Courses Đáng Chú Ý Khác

| Khóa Học | Nhà Cung Cấp | Link | Ghi Chú |
|-----------|---------------|------|----------|
| OpenAI Academy | OpenAI | https://openai.com/education | Free courses cho OpenAI ecosystem |
| Anthropic Academy | Anthropic | https://anthropic.com/academy | Claude API, certificates miễn phí |
| Google ML Crash Course | Google Developers | https://developers.google.com/machine-learning/recommendation/crash-course | 12 modules, beginner+ |

---

## 2. PyTorch & Frameworks

### 2.1 Official Resources

| Resource | Link | Mô Tả |
|----------|------|--------|
| PyTorch Official Documentation | https://pytorch.org/docs/ | Tài liệu chính thức, best reference |
| PyTorch Tutorials | https://pytorch.org/tutorials/ | Official tutorials |
| PyTorch GitHub | https://github.com/pytorch/pytorch | Source code |

### 2.2 PyTorch Tutorials & Books

| Resource | Tác Giả | Link | Mô Tả |
|----------|---------|------|--------|
| Learn PyTorch for Deep Learning | MrDBourke | https://www.learnpytorch.io/ | Code-first, project-based, free online book. Clone: `git clone https://github.com/mrdbourke/pytorch-deep-learning.git` |
| PyTorch in One Hour | Sebastian Raschka, PhD | https://sebastianraschka.com/teaching/pytorch-1h/ | Concise overview về autograd, nn.Module |
| Machine Learning with PyTorch and Scikit-Learn | Sebastian Raschka | https://sebastianraschka.com/books/machine-learning-with-pytorch-and-scikit-learn/ | Comprehensive ML với PyTorch |
| PyTorch Comprehensive Tutorial | Han Fang | https://hanfang.info/posts/2025/08/pytorch-comprehensive-tutorial/ | Từ tensor cơ bản đến attention, mixed precision |
| Deep Learning with PyTorch in 1 Hour | - | https://towardsdatascience.com/the-basics-of-deep-learning-with-pytorch-in-1-hour/ | Towards Data Science guide |

### 2.3 PyTorch Performance & Optimization

| Resource | Tác Giả | Link | Topics |
|----------|---------|------|--------|
| PyTorch Memory Optimization | Sebastian Raschka | https://sebastianraschka.com/blog/2023/pytorch-memory-optimization.html | Memory management |
| PyTorch Training Speedups | Sebastian Raschka | https://sebastianraschka.com/blog/2023/pytorch-faster.html | Training optimization |
| Mixed Precision for LLMs | Sebastian Raschka | https://sebastianraschka.com/blog/2023/llm-mixed-precision-copy.html | FP16, BF16 training |
| Gradient Accumulation | Sebastian Raschka | https://sebastianraschka.com/blog/2023/llm-grad-accumulation.html | Single-GPU LLM finetuning |

### 2.4 Framework Ecosystem

| Framework | Link | Mô Tả |
|-----------|------|--------|
| fastai | https://fast.ai/ | High-level DL library built on PyTorch |
| Hugging Face Transformers | https://huggingface.co/docs/transformers/ | SOTA NLP models |
| Hugging Face PEFT | https://huggingface.co/docs/peft/ | Parameter-Efficient Fine-Tuning |
| Hugging Face Diffusers | https://huggingface.co/docs/diffusers/ | Diffusion models |
| Gradio | https://gradio.app/ | ML demo UI |

---

## 3. GitHub Repositories

### 3.1 Core Learning Repositories

| Repository | Stars | Link | Mô Tả |
|-----------|-------|------|--------|
| mrdbourke/pytorch-deep-learning | - | https://github.com/mrdbourke/pytorch-deep-learning | PyTorch Deep Learning course by MrDBourke |
| huggingface/transformers | 125k+ | https://github.com/huggingface/transformers | SOTA pretrained models (PyTorch, TensorFlow, JAX) |
| fastai/fastbook | - | https://github.com/fastai/fastbook | Fast.ai book (Jupyter notebooks) |

### 3.2 Transformers & LLMs

| Repository | Link | Mô Tả |
|-----------|------|--------|
| stanford-cs324/llm-introduction | https://github.com/stanford-cs324/llm-introduction | Stanford LLM Introduction |
| princeton-pli/MixiT | https://github.com/princeton-pli/MixiT | Transformer components analysis |

---

## 4. Blogs & Tutorials

### 4.1 Visual Learning (Jay Alammar)

| Blog Post | Link | Mô Tả |
|-----------|------|--------|
| The Illustrated Transformer | http://jalammar.github.io/illustrated-transformer/ | **CLASSIC** - Visual explanation of Transformer |
| The Illustrated GPT-2 | https://jalammar.github.io/illustrated-gpt2/ | Decoder-only transformer visual guide |
| The Illustrated BERT | https://jalammar.github.io/illustrated-bert/ | BERT và transfer learning |
| The Illustrated Word2vec | https://jalammar.github.io/illustrated-word2vec/ | Word embeddings |
| How GPT3 Works | https://jalammar.github.io/illustrated-gpt3/ | GPT-3 visual guide |
| Visual Guide to BERT | https://jalammar.github.io/illustrated-bert/ | First-time BERT usage |
| Visual and Interactive Neural Network Math | https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/ | Neural network math |
| **NEW: How Transformer LLMs Work Course** | https://www.llm-book.com/ | Updated course với animations |

> **Note:** Jay Alammar đã chuyển sang Substack: https://substack.com/@jayalammar

### 4.2 Sebastian Raschka's Blog

| Blog Post | Link | Topics |
|-----------|------|--------|
| Blog Archive | https://sebastianraschka.com/blog/ | Chronological ML notes |
| Research Notes | https://sebastianraschka.com/notes/ | Paper summaries, benchmarks |
| PyTorch Memory Optimization | Link ở trên | Memory management |
| PyTorch Faster Training | Link ở trên | Training optimization |
| From GPT-2 to GPT-OSS Analysis | https://newsletter.languagemodels.co/p/the-illustrated-gpt-oss | GPT-OSS architecture analysis |

### 4.3 Other Notable Blogs

| Blog | Link | Topics |
|------|------|--------|
| Lil'Log | https://lilianweng.github.io/ | Deep learning, RL, NLP |
| Distill.pub | https://distill.pub/ | Interactive ML explanations |
| The Batch (Andrew Ng) | https://www.deeplearning.ai/the-batch/ | Weekly AI newsletter |
| Towards Data Science | https://towardsdatascience.com/ | ML/DL articles |

---

## 5. Papers - Foundational

### 5.1 Neural Network Basics

| Paper | Năm | Link | Contribution |
|-------|-----|------|--------------|
| ImageNet Classification with Deep Convolutional Neural Networks (AlexNet) | 2012 | https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf | CNN breakthrough, GPU training, 184k+ citations |
| Gradient-Based Learning Applied to Document Recognition (LeNet) | 1998 | - | Early CNN architecture |

### 5.2 Computer Vision

| Paper | Năm | Link | Contribution |
|-------|-----|------|--------------|
| Deep Residual Learning for Image Recognition (ResNet) | 2016 | https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf | Residual connections, 152 layers, ILSVRC 2015 winner |
| Identity Mappings in Deep Residual Networks | 2016 | - | ResNet v2, better propagation |
| Very Deep Convolutional Networks for Large-Scale Image Recognition (VGGNet) | 2015 | - | Stacked 3x3 convolutions |
| Going Deeper with Convolutions (GoogLeNet/Inception) | 2015 | - | Inception module, multi-scale |

### 5.3 Transformer & Attention (Original)

| Paper | Năm | Link | Contribution |
|-------|-----|------|--------------|
| **Attention Is All You Need** | 2017 | - | **FOUNDATIONAL** - Transformer architecture, self-attention |
| BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | 2018 | - | Bidirectional encoding, GLUE leaderboard |
| GPT-2: Language Models are Unsupervised Multitask Learners | 2019 | - | GPT architecture, zero-shot |

---

## 6. Papers - Transformer & Attention (2025)

### 6.1 Theoretical Analysis

| Paper | Conference | Link | Key Finding |
|-------|------------|------|------------|
| The underlying structures of self-attention: symmetry, directionality, and emergent dynamics | **ICML 2025** | https://proceedings.mlr.press/v267/saponati25a.html | Bidirectional training → symmetry; Autoregressive → directionality. Validated on GPT, LLaMA3, Mistral |
| Consensus Is All You Get: The Role of Attention in Transformers | **ICML 2025** | https://proceedings.mlr.press/v267/abella25a.html | All tokens asymptotically converge to each other across layers |
| Attention Mechanism, Max-Affine Partition, and Universal Approximation | **NeurIPS 2025** | https://proceedings.neurips.cc/paper_files/paper/2025/file/ae6c7dbd9429b3a75c41b5fb47e57c9e-Paper-Conference.pdf | **Single-layer, single-head softmax attention is a universal approximator** for sequence-to-sequence functions |
| Is Random Attention Sufficient for Sequence Modeling? | 2025 | https://arxiv.org/html/2506.01115v3 | Random frozen attention can form induction heads and perform competitively on language modeling |

### 6.2 Key Findings Summary

```
1. SYMMETRY vs DIRECTIONALITY:
   - Bidirectional (BERT) → symmetric attention matrices
   - Autoregressive (GPT) → directional, column-dominant

2. TOKEN CONVERGENCE:
   - All tokens converge to each other across transformer layers
   - This is why deeper layers lose positional information

3. UNIVERSAL APPROXIMATION:
   - Single-head softmax attention alone can approximate ANY continuous
     sequence-to-sequence function (given sufficient width)

4. RANDOM ATTENTION:
   - Frozen QK weights can still perform well
   - MLPs are responsible for knowledge memorization
   - Attention handles in-context reasoning
```

---

## 7. Computer Vision Papers

### 7.1 CNN Evolution

| Paper | Năm | Key Innovation |
|-------|-----|----------------|
| LeNet-5 | 1998 | Early CNN for digit recognition |
| AlexNet | 2012 | Deep CNN, ReLU, dropout, GPU training |
| VGGNet | 2014 | Deep 3x3 conv stacks |
| GoogLeNet | 2014 | Inception module, parallel paths |
| ResNet | 2016 | Residual connections, skip connections |
| Inception-v4 | 2016 | Inception + residual |
| MobileNet | 2017 | Depthwise separable convolutions |
| EfficientNet | 2019 | Compound scaling (depth, width, resolution) |

### 7.2 Modern Vision Transformers

| Paper | Năm | Key Innovation |
|-------|-----|----------------|
| ViT: Vision Transformer | 2020 | Transformer for images |
| Swin Transformer | 2021 | Hierarchical, shifted windows |
| MAE | 2022 | Masked autoencoder, self-supervised |
| DINO | 2021 | Self-supervised ViT |

### 7.3 Object Detection

| Paper | Năm | Key Innovation |
|-------|-----|----------------|
| R-CNN | 2014 | Region proposals |
| YOLO | 2016 | Real-time detection |
| Mask R-CNN | 2017 | Instance segmentation |

---

## 8. Lộ Trình Học

### 8.1 Beginner Path (3-6 tháng)

```
Phase 1: Python & Math Foundation (2-4 tuần)
├── Python: https://www.python.org/
├── NumPy: https://numpy.org/
├── Calculus basics
└── Linear Algebra basics

Phase 2: Machine Learning Basics (4-6 tuần)
├── Kaggle ML Course: https://www.kaggle.com/learn
├── Elements of AI: https://www.elementsofai.com/
└── Andrew Ng's ML Course (audit free)

Phase 3: Deep Learning Intro (4-8 tuần)
├── MIT 6.S191: https://introtodeeplearning.com/
├── Fast.ai Part 1: https://course.fast.ai/
└── Learn PyTorch: https://www.learnpytorch.io/
```

### 8.2 Intermediate Path (3-6 tháng)

```
Phase 4: NLP with Transformers (4-8 tuần)
├── Jay Alammar's Illustrated posts
├── Hugging Face Course: https://huggingface.co/learn
├── Attention Is All You Need paper
└── BERT paper

Phase 5: Computer Vision (4-8 tuần)
├── Stanford CS231n (audit free)
├── fast.ai Part 1 (CV section)
└── ResNet paper

Phase 6: Advanced Topics (ongoing)
├── Andrej Karpathy Zero to Hero
├── LLM Course
└── Paper reading (section 5, 6, 7)
```

### 8.3 Specialization Paths

```
NLP/LLM Path:
├── Hugging Face Transformers docs
├── LLM Course
├── LangChain basics
└── Fine-tuning LLMs

Computer Vision Path:
├── ViT paper
├── Diffusers library
└── Object detection fundamentals

Research Path:
├── CS229 Stanford
├── Read recent papers (arxiv, NeurIPS, ICML)
└── Reproduce paper results
```

---

## 9. Tài Nguyên Bổ Sung

### 9.1 YouTube Channels

| Channel | Link | Topics |
|---------|------|--------|
| 3Blue1Brown Neural Networks | https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZC07-9KK | Visual intuition for neural networks |
| Andrej Karpathy | https://www.youtube.com/@AndrejKarpathy | LLM, neural networks |
| Sentdex | https://www.youtube.com/@sentdex | Python, ML, pygame |

### 9.2 Books (Free Online)

| Book | Link | Notes |
|------|------|-------|
| Dive into Deep Learning | https://d2l.ai/ | Interactive, PyTorch/TensorFlow |
| Fastbook | https://github.com/fastai/fastbook | Fast.ai companion |
| Speech and Language Processing | https://web.stanford.edu/~jurafsky/slp3/ | NLP comprehensive |

### 9.3 Practice Platforms

| Platform | Link | Notes |
|---------|------|-------|
| Kaggle | https://www.kaggle.com/competitions | competitions, notebooks |
| Papers with Code | https://paperswithcode.com/ | SOTA, code |
| Hugging Face Spaces | https://huggingface.co/spaces | demos |
| Colab | https://colab.research.google.com/ | Free GPU |

---

## 10. Quick Reference

### 10.1 Độ quan trọng của Papers

```
⭐⭐⭐⭐⭐ MUST READ:
- Attention Is All You Need (2017)
- AlexNet (2012)
- ResNet (2016)

⭐⭐⭐⭐ SHOULD READ:
- BERT (2018)
- GPT-2 (2019)
- ViT (2020)

⭐⭐⭐ INTERESTING (2025):
- Consensus Is All You Get (ICML 2025)
- Attention Universal Approximation (NeurIPS 2025)
- Self-Attention Structures (ICML 2025)
```

### 10.2 Cách Đọc Paper Hiệu Quả

```
1. Abstract → đọc trước để hiểu contribution
2. Figures → xem hình trước
3. Methodology → hiểu approach
4. Experiments → xem results
5. Related Work → hiểu context
6. Code (nếu có) → reproduce
```

---

## Changelog

| Ngày | Agent | Thay đổi |
|------|-------|-----------|
| 2026-07-24 | Main Agent (Cursor) | Tạo file, tổng hợp tài liệu |

---

## Agent Acknowledgement

**Tôi xác nhận đã đọc và hiểu toàn bộ nội dung file này.**

**Cam kết:**
- Cập nhật khi có tài liệu mới
- Verify links trước khi recommend
- Ưu tiên tài liệu có bằng chứng (papers > blogs > random tutorials)
- Nếu link chết → tìm nguồn thay thế và cập nhật

**Signature:** `Main Agent (Cursor) — 2026-07-24`
