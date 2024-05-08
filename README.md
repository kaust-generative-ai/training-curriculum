# generative-ai-training

This repository develops a training curriculum for Generative AI. Rather than starting with the technical aspects of training and fine-tuning Generative AI models, the curriculum will first teach students the skills and knowledge that they need to start deploying Generative AI models immediately. Once students master the skills to deploy ethical and responsible Generative AI solutions, they will then focus on the more technical aspects of prompt-engineering, fine-tuning, and training of Generative AI models from scratch. Below is the current outline of the curriculum.

## Practical Deployment of Generative AI Models

### Deployment:

Cover various strategies for deploying Generative AI models, including on-premises, cloud-based deployment, and edge device deployments. Cover the pros and cons of each strategy and the factors to consider when choosing a deployment strategy.

### Model Optimization:

Cover techniques for optimizing Generative AI models for deployment, such as model pruning, quantization, and distillation. Cover the trade-offs between model size, speed, and performance.

### Monitoring and Maintenance: 

Cover the importance of monitoring the performance of deployed models and updating them as needed. Discuss potential issues that might arise during deployment and how to troubleshoot them.

## Ethical and Responsible Use of Generative AI

### Ethical Considerations: 

Discuss the ethical considerations of deploying LLMs, including issues of bias, fairness, transparency, and accountability. Discuss real-world examples where these issues have arisen and how they were addressed.

### Responsible AI Practices: 

Teach students about best practices for responsible AI, including techniques for auditing and mitigating bias in LLMs, transparency and explainability techniques, and guidelines for human oversight.

### Legal and Regulatory Considerations: 

Provide an overview of the legal and regulatory landscape for AI and LLMs. Discuss issues such as data privacy, intellectual property rights, and liability.

## Practical Evaluation of LLMs

### Introduction to Evaluation Metrics: 

This section would cover the various metrics used to evaluate the performance of LLMs. This could include Perplexity, BLEU, ROUGE, and others. Each metric should be explained in detail, including what it measures and its strengths and weaknesses.

### Evaluating LLMs: 

Teach students how to evaluate the performance of LLMs using the discussed metrics. This should include practical exercises where students can apply these metrics to evaluate LLMs on various tasks.

### Hands-on Exercises: 

Provide students with practical exercises where they can practice evaluating LLMs. We will use real-world examples and datasets provided by our partners for these exercises to give students practical experience.

## Practical Prompt Optimization for LLMs

### Introduction to Prompt Optimization: 

Discuss the concept of prompt optimization and its importance in effectively using LLMs. Explain how carefully optimized prompts can guide the model's responses and improve its performance.

### Techniques for Prompt Engineering: 

Teach students various techniques for designing effective prompts. This would include methods for crafting initial prompts, techniques for iterative refinement, and strategies for testing and evaluating prompts.

### Techniques for Prompt Tuning: 

Prompt-tuning introduces additional parameters into prompts and then optimizes those parameters using supervised samples. 

### Prompt Engineering vs Prompt Tuning: 

Discuss the numerous trade-offs involved when deciding which of prompt engineering or prompt tuning (or both!) is the best approach for tuning an LLM.

## Practical Fine-Tuning of LLMs

### Introduction to Fine-Tuning: 

Discuss the concept of fine-tuning and its importance in effectively using LLMs. Explain how fine-tuning adjusts the pre-trained models to perform specific tasks.

### "Full" Fine-Tuning: 

Cover approaches such as Transfer Learning and Knowledge Distillation.

### Parameter Efficient Fine-Tuning: 

Cover approaches such as Prefix Tuning and Low Rank Adaptation (LoRA).

### Instruction Tuning: 

Cover the basic ideas of Instruction Tuning as well as extensions such as the popular Reinforcement Learning through Human Feedback (RLHF).

### Alignment: 

Cover approaches to fine-tuning, such as Direct Preference Optimization (DPO), where the goal of the fine-tuning is to improve the alignment of LLMs with human preferences.

## Practical Training of LLMs from Scratch

### Introduction to LLMs: 

Overview of LLMs and their applications. Understanding the Transformer Model Architecture. The Original LLM Scaling Laws. Discussion on whether to build or use pre-trained LLM models.

### Hardware and Scaling: 

Understanding the hardware requirements for training LLMs. The role of memory and compute efficiency in LLM training. Techniques for parallelization, including gradient accumulation, asynchronous stochastic gradient descent optimization, and micro-batching.

### Data Collection and Pre-processing: 

The importance of dataset diversity and quality in LLM training. Techniques for dataset collection, including crawling public data, online publication or book repositories, code data from GitHub, Wikipedia, news, social media conversations, etc. Dataset pre-processing. Tokenization.

### Pre-Training and Model Evaluation: 

Steps involved in pre-training an LLM. Techniques for model evaluation. Understanding bias and toxicity in LLMs. Instruction tuning.
