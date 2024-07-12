# generative-ai-training

This repository develops a training curriculum for Generative AI. Rather than starting with the technical aspects of training and fine-tuning Generative AI models, the curriculum will first teach students the skills and knowledge that they need to start deploying Generative AI models immediately. Once students master the skills to deploy ethical and responsible Generative AI solutions, they will then focus on the more technical aspects of prompt-engineering, fine-tuning, and training of Generative AI models from scratch. Below is the current outline of the curriculum.

## Practical Deployment of Generative AI Models

### Deployment:

Cover various strategies for deploying Generative AI models, including on-premises, cloud-based deployment, and edge device deployments. Cover the pros and cons of each strategy and the factors to consider when choosing a deployment strategy.

#### Cloud-based Deployment

* [DeepLearning AI: Serverless LLM Apps using AWS Bedrock](https://www.deeplearning.ai/short-courses/serverless-llm-apps-amazon-bedrock/)
* [DeepLearning AI: Developing Generative AI Apps using Microsoft Semantic Kernel](https://www.deeplearning.ai/short-courses/microsoft-semantic-kernel/)
* [DeepLearning AI: Understanding and Applying Text Embeddings with Vertex AI](https://www.deeplearning.ai/short-courses/google-cloud-vertex-ai/)
* [DeepLearning AI: Pair Programming with LLMs](https://www.deeplearning.ai/short-courses/pair-programming-llm/)

#### Local Deployment

* [DeepLearning AI: Open Source Models with HuggingFace](https://www.deeplearning.ai/short-courses/open-source-models-hugging-face/)
* [DeepLearning AI: Building Generative AI Apps](https://www.deeplearning.ai/short-courses/building-generative-ai-applications-with-gradio/)
* [Ollama](https://ollama.com) ([GitHub](https://github.com/ollama/ollama))
* [Llama.cpp](https://github.com/ggerganov/llama.cpp)
* [LlamaFile](https://github.com/Mozilla-Ocho/llamafile)
* [Latent Space Podcast: Tiny Model Revolution](https://www.latent.space/p/cogrev-tinystories)
  
#### Edge Deployment

* [DeepLearning AI: Introduction to Device AI](https://www.deeplearning.ai/short-courses/introduction-to-on-device-ai/)
* [Machine Learning Compilation](https://llm.mlc.ai) ([GitHub](https://github.com/mlc-ai/mlc-llm))
* [Deploying LLMs in your Web Browser](https://github.com/mlc-ai/web-llm)
* [NVIDIA Orin SDK](https://developer.nvidia.com/blog/deploy-large-language-models-at-the-edge-with-nvidia-igx-orin-developer-kit/)
* [NVIDIA Holoscan SDK](https://developer.nvidia.com/holoscan-sdk) ([GitHub](https://github.com/nvidia-holoscan/holoscan-sdk))
* [NVIDIA Holohub](https://github.com/nvidia-holoscan/holohub)

#### UI/UX

* [Open WebUI](https://github.com/open-webui/open-webui)
* [Blog Post: Emerging UI/UX patterns for AI applications](https://uxdesign.cc/emerging-interaction-patterns-in-generative-ai-experiences-8c351bb3392a)
  
### Model Optimization:

Cover techniques for optimizing Generative AI models for deployment, such as model pruning, quantization, and distillation. Cover the trade-offs between model size, speed, and performance.

* [DeepLearning AI: Quantization Fundamentals](https://www.deeplearning.ai/short-courses/quantization-fundamentals-with-hugging-face/)
* [DeepLearning AI: Quantization in Depth](https://www.deeplearning.ai/short-courses/quantization-in-depth/)

#### Open Source Inference Servers:

* [OpenLLM](https://github.com/bentoml/OpenLLM)
* [VLLM](https://github.com/vllm-project/vllm)
* [Cog](https://github.com/replicate/cog)
* [TGI](https://github.com/huggingface/text-generation-inference)
* [TEI](https://github.com/huggingface/text-embeddings-inference)
 
### Monitoring and Maintenance: 

Cover the importance of monitoring the performance of deployed models and updating them as needed. Discuss potential issues that might arise during deployment and how to troubleshoot them.

* [DeepLearning AI: Evaluating and Debugging Generative AI Apps](https://www.deeplearning.ai/short-courses/evaluating-debugging-generative-ai/)
* [DeepLearning AI: LLMOps](https://www.deeplearning.ai/short-courses/llmops/)
* [DeepLearning AI: Automated Testing for LLMOps](https://www.deeplearning.ai/short-courses/automated-testing-llmops/)

## Open-Source Tools of the Generative AI Trade

### LangChain

* [O'Reilly: Generative AI with LangChain](https://www.oreilly.com/library/view/generative-ai-with/9781835083468/)
* [DeepLearning AI: LangChain for LLM App Development](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/)
* [DeepLearning AI: LangChain: Chat with your Data](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/)
* [DeepLearning AI: LangChain: Functions, Tools, and Agents](https://www.deeplearning.ai/short-courses/functions-tools-agents-langchain/)
* [DeepLearning AI: LLM Apps with JavaScript](https://www.deeplearning.ai/short-courses/build-llm-apps-with-langchain-js/)
* [Latent Space Podcast: LangChain](https://www.latent.space/p/langchain)
  
### LlamaIndex

* [DeepLearning AI: Building and Evaluating Advanced RAG Apps](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/)
* [DeepLearning AI: Agentic RAG Apps](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex/)
* [DeepLearning AI: RAG Apps with JavaScript](https://www.deeplearning.ai/short-courses/javascript-rag-web-apps-with-llamaindex/)
* [Deep Learning AI: Multi-Agent Systems](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/)
* [Latent Space Podcast: Llama Index](https://www.latent.space/p/llamaindex)
  
### Vector Databases

* [DeepLearning AI: Vector Databases: Embedding to Applications](https://www.deeplearning.ai/short-courses/vector-databases-embeddings-applications/)
* [DeepLearning AI: Knowledge Graphs for RAG](https://www.deeplearning.ai/short-courses/knowledge-graphs-rag/)
* [DeepLearning AI: Advanced AI Retrieval](https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/)
* [DeepLearning AI: Building Applications with Vector databases](https://www.deeplearning.ai/short-courses/building-applications-vector-databases/)
  
### Other

* [DeepLearning AI: Building Multi-modal RAG Apps](https://www.deeplearning.ai/short-courses/building-multimodal-search-and-rag/)

## Ethical and Responsible Use of Generative AI

### Ethical Considerations: 

Discuss the ethical considerations of deploying LLMs, including issues of bias, fairness, transparency, and accountability. Discuss real-world examples where these issues have arisen and how they were addressed.

* [DeepLearning AI: Red-Teaming LLM Applications](https://www.deeplearning.ai/short-courses/red-teaming-llm-applications/)

### Responsible AI Practices: 

Teach students about best practices for responsible AI, including techniques for auditing and mitigating bias in LLMs, transparency and explainability techniques, and guidelines for human oversight.

* [DeepLearning AI: Quality and Safety of LLM Apps](https://www.deeplearning.ai/short-courses/quality-safety-llm-applications/)
  
### Legal and Regulatory Considerations: 

Provide an overview of the legal and regulatory landscape for AI and LLMs. Discuss issues such as data privacy, intellectual property rights, and liability.

## Practical Evaluation of LLMs

### Introduction to Evaluation Metrics: 

This section would cover the various metrics used to evaluate the performance of LLMs. This could include Perplexity, BLEU, ROUGE, and others. Each metric should be explained in detail, including what it measures and its strengths and weaknesses.

### Evaluating LLMs: 

Teach students how to evaluate the performance of LLMs using the discussed metrics. This should include practical exercises where students can apply these metrics to evaluate LLMs on various tasks.
 
### Hands-on Exercises: 

Provide students with practical exercises where they can practice evaluating LLMs. We will use real-world examples and datasets provided by our partners for these exercises to give students practical experience.

* [DeepLearning AI: LLMs for Semantic Search](https://www.deeplearning.ai/short-courses/large-language-models-semantic-search/)
* [DeepLearning AI: Preprocessing Unstructured Data for LLM Apps](https://www.deeplearning.ai/short-courses/preprocessing-unstructured-data-for-llm-applications/)
* [Latent Space Podcast: Benchmarks 101](https://www.latent.space/p/benchmarks-101)
  
## Practical Prompt Optimization for LLMs

### Introduction to Prompt Optimization: 

Discuss the concept of prompt optimization and its importance in effectively using LLMs. Explain how carefully optimized prompts can guide the model's responses and improve its performance.

### Techniques for Prompt Engineering: 

Teach students various techniques for designing effective prompts. This would include methods for crafting initial prompts, techniques for iterative refinement, and strategies for testing and evaluating prompts.

* [O'Reilly: Prompt Engineering for Generative AI](https://www.oreilly.com/library/view/prompt-engineering-for/9781098153427/) ([GitHub](https://github.com/BrightPool/prompt-engineering-for-generative-ai-examples/), [Udemy](https://www.udemy.com/course/prompt-engineering-for-ai/?couponCode=OF52424))
* [DeepLearning AI: Prompt Engineering with Llama 2/3](https://www.deeplearning.ai/short-courses/prompt-engineering-with-llama-2/)
* [DeepLearning AI: ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/)
* [DeepLearning AI: Building systems with ChatGPT](https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/)
* [DeepLearning AI: Getting Started with Mistral](https://www.deeplearning.ai/short-courses/getting-started-with-mistral/)
* [DeepLearning AI: Prompt Engineering for Vision Models](https://www.deeplearning.ai/short-courses/prompt-engineering-for-vision-models/)
* [Udemy: Prompt Engineering for AI](https://www.udemy.com/course/prompt-engineering-for-ai/?couponCode=ST19MT60324)
* [Open AI: Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
  
### Techniques for Prompt Tuning: 

Prompt-tuning introduces additional parameters into prompts and then optimizes those parameters using supervised samples. 

### Prompt Engineering vs Prompt Tuning: 

Discuss the numerous trade-offs involved when deciding which of prompt engineering or prompt tuning (or both!) is the best approach for tuning an LLM.

## Practical Fine-Tuning of LLMs

### Introduction to Fine-Tuning: 

Discuss the concept of fine-tuning and its importance in effectively using LLMs. Explain how fine-tuning adjusts the pre-trained models to perform specific tasks.

* [Mastering LLMs: End-to-end Fine-tuning and Deployment](https://maven.com/parlance-labs/fine-tuning)
* [Latent Space Podcast: The End of Fine-Tuning](https://www.latent.space/p/fastai)
* [Latent Space Podcast: Axolotl](https://www.latent.space/p/axolotl)

### "Full" Fine-Tuning: 

Cover approaches such as Transfer Learning and Knowledge Distillation.

* [DeepLearning AI: Finetuning LLMs](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/)

### Parameter Efficient Fine-Tuning: 

Cover approaches such as Prefix Tuning and Low Rank Adaptation (LoRA).

* [DeepLearning AI: Efficient Serving LLMs](https://www.deeplearning.ai/short-courses/efficiently-serving-llms/)

### Instruction Tuning: 

Cover the basic ideas of Instruction Tuning as well as extensions such as the popular Reinforcement Learning through Human Feedback (RLHF).

### Alignment: 

Cover approaches to fine-tuning, such as Direct Preference Optimization (DPO), where the goal of the fine-tuning is to improve the alignment of LLMs with human preferences.

* [DeepLearning AI: RLHF](https://www.deeplearning.ai/short-courses/reinforcement-learning-from-human-feedback/)
* [Latent Space Podcast: RLHF 201](https://www.latent.space/p/rlhf-201)
* [Latent Space Podcast: From RLHF to RLHB](https://www.latent.space/p/amplitude)
  
## Practical Training of LLMs from Scratch

### Introduction to LLMs: 

Overview of LLMs and their applications. Understanding the Transformer Model Architecture. The Original LLM Scaling Laws. Discussion on whether to build or use pre-trained LLM models.

* [DeepLearning AI: How Diffusion Models Work](https://www.deeplearning.ai/short-courses/how-diffusion-models-work/)
* [DeepLearning AI: Generative AI with LLMs](https://www.deeplearning.ai/courses/generative-ai-with-llms/)
* [Building LLMs from Scratch](https://www.manning.com/books/build-a-large-language-model-from-scratch) ([GitHub](https://github.com/rasbt/LLMs-from-scratch/))

[Suggested organization](https://x.com/rasbt/status/1790013057659183601?s=12&t=I0v324dBexoJhxivPjEmZw) of materials for this course.

1. Read* Chapters 1 and 2 on implementing the data loading pipeline (https://manning.com/books/build-a-large-language-model-from-scratch… & https://github.com/rasbt/LLMs-from-scratch…).
2. Watch Karpathy's video on training a BPE tokenizer from scratch (https://youtube.com/watch?v=zduSFxRajkE…).
3. Read Chapters 3 and 4 on implementing the model architecture.
4. Watch Karpathy's video on pretraining the LLM.
5. Read Chapter 5 on pretraining the LLM and then loading pretrained weights.
6. Read Appendix E on adding additional bells and whistles to the training loop.
7. Read Chapters 6 and 7 on finetuning the LLM.
8. Read Appendix E on parameter-efficient finetuning with LoRA.
9. Check out Karpathy's repo on coding the LLM in C code (https://github.com/karpathy/llm.c).
10. Check out LitGPT to see how multi-GPU training is implemented and how different LLM architectures compare (https://github.com/Lightning-AI/litgpt…).
11. Build something cool and share it with the world.

(Read = read, run the code, and attempt the exercises 😊)
   
### Hardware and Scaling: 

Understanding the hardware requirements for training LLMs. The role of memory and compute efficiency in LLM training. Techniques for parallelization, including gradient accumulation, asynchronous stochastic gradient descent optimization, and micro-batching.

### Data Collection and Pre-processing: 

The importance of dataset diversity and quality in LLM training. Techniques for dataset collection, including crawling public data, online publication or book repositories, code data from GitHub, Wikipedia, news, social media conversations, etc. Dataset pre-processing. Tokenization.

* [Latent Space Podcast: Datasets 101](https://www.latent.space/p/datasets-101)

### Pre-Training and Model Evaluation: 

Steps involved in pre-training an LLM. Techniques for model evaluation. Understanding bias and toxicity in LLMs. Instruction tuning.

* [Latent Space Podcast: How to Train Your Own Multi-modal LLM](https://www.latent.space/p/idefics)
