conda activate "c:\Continuous Learning\Fundamentals-of-AI-Agents-Using-RAG-and-LangChain\.conda"

conda deactivate

Python interpreter path: .\.conda\python.exe


pip install huggingface_hub[hf_xet]

pip install -r requirements.txt
-------------------------------------------------------------------


Fundamentals of AI Agents Using RAG and LangChain
Module 1
Course Overview 
Course Overview 
Hello, and welcome to “Fundamentals of Building AI Agents using RAG and LangChain.”

In this course, you will explore how RAG is used to generate a response when the model is not pre-trained. You will gain knowledge about the RAG process, including context and question encoders with their tokenizers, as well as the Facebook AI similarity search (Faiss) library. Additionally, you will gain insight into in-context learning, advanced methods of prompt engineering and its key elements, and prompt templates with LangChain. You will then deep dive into the LangChain tools, components, chat models, and document loader. Finally, you will gain a clear understanding of how LangChain chains and agents are used for developing applications.

In hands-on labs, you will use the Jupyter labs environment to practice these concepts and technologies to build a solid foundation for their application in your projects. At the end of this course, you’ll also complete a project based on a real-world scenario.

Prerequisites:
You’ll require basic knowledge of Generative AI, prompt engineering techniques as well as working knowledge of machine learning with Python and PyTorch. This course is suitable for professionals aspiring to build their careers in AI engineering, including training, developing, fine-tuning, and deploying LLMs. 

This course is one in a series of specializations: Generative AI Engineering with LLMs Specialization. Follow the link(s) below to gain knowledge on generative AI and see how these programs can benefit you and advance your career.

Generative AI Engineering Specialization:

Course 1: Generative AI and LLMs: Architecture and Data Preparation

Course 2: Generative AI Foundational Models for NLP & Language Understanding
[CN1]

Course 3: Generative AI Language Modeling with Transformers

Course 4: Generative AI Engineering and Fine-Tuning Transformers

Course 5: Generative AI Advanced Fine-Tuning for LLMs

Course Objectives:
After completing this course, you will be able to:

Describe retrieval-augmented generation (RAG), encoders, and Faiss.

Apply fundamentals of in-context learning and advanced methods of prompt engineering to enhance prompt design.

Explain the LangChain concept, tools, components, chat models, chains, and agents.

Apply RAG, PyTorch, Hugging Face, LLMs, and LangChain technologies to different applications to acquire job-ready skills.

Course Outline:
This course has two modules, which is listed below. We encourage you to set aside several hours to complete this course successfully. Consistency will help you achieve your learning goals!

You will benefit from viewing all videos and readings and strengthening that knowledge by completing all activities.

Module 1: RAG Framework

In this module, you will learn how RAG is used to generate responses for different applications such as chatbots. You’ll then learn about the RAG process, the Dense Passage Retrieval (DPR) context encoder and question encoder with their tokenizers, and the Faiss library developed by Facebook AI Research for searching high-dimensional vectors.

In hands-on labs, you will Use RAG with PyTorch to evaluate content appropriateness and with Hugging Face to retrieve information from the dataset. 

Module 2: Prompt Engineering and LangChain

In this module, you will learn about in-context learning and advanced methods of prompt engineering to design and refine the prompts for generating relevant and accurate responses from AI. You’ll then be introduced to the LangChain framework, which is an open-source interface for simplifying the application development process using LLM. You’ll learn about its tools, components, and chat models. The module also includes concepts such as prompt templates, example selectors, and output parsers. You’ll then explore the LangChain document loader and retriever, LangChain chains and agents for building applications. 

In hands-on labs, you will enhance LLM applications and develop an agent that uses integrated LLM, LangChain, and RAG technologies for interactive and efficient document retrieval.

Tools/Software Used:
In this course, you can view videos and readings of the course using any web-enabled device, including tablets and mobile phones. A modern web browser is required to complete this course. You’ll be able to leverage tools like document loader, text splitter, vector databases, and embeddings to improve the quality of LLM-generated responses.

Congratulations on taking these steps to further your knowledge and career! Enjoy your journey.

Fundamentals of AI Agents Using RAG and LangChain
Module 1
Specialization Overview
Specialization Overview
The knowledge of generative AI architectures and models is integral to launching or advancing careers in AI, machine learning, and data science. This specialization focuses on the principles, techniques, and applications of generative AI and large language models (LLMs) for natural language processing (NLP). The specialization best suits professionals aspiring to be or currently working as AI Engineers, machine learning (ML) engineers, and data scientists.

This specialization requires basic skills in Python, knowledge of PyTorch, and an understanding of machine learning and neural networks.

The program comprises seven online courses covering the most popular frameworks and pretrained LLMs, such as generative pretrained transformers (GPT), bidirectional encoder representations from transformers (BERT), and LLaMA. You will use NLP libraries, such as the Hugging Face Transformers library and the PyTorch deep learning library, to develop, train, and deploy AI applications that use LLMs. 

The hands-on labs in this course will help develop practical skills. For example, you will create an NLP data loader and build and train a simple language model with a neural network. You will also apply transformers for classification, build and evaluate a translation model, and perform prompt engineering.

The last module in the course is a capstone project. In this project, you will develop a question-answering system using generative AI for NLP in three phases: preparation, development, and deployment. 

Upon completing the course in this specialization, you’ll gain a portfolio of projects where you’ve learned to develop generative AI applications using LLMs.

In addition to earning a Professional Certificate from Coursera, you'll receive a digital badge from IBM recognizing your proficiency in using LLMs to build generative AI applications.

Who should take this specialization?
This specialization is designed for professionals interested in AI engineering, including training, developing, fine-tuning, and deploying LLMs. It is also suitable for existing and aspiring data scientists and machine learning engineers with basic knowledge of Python. Knowledge of PyTorch, machine learning, and neural networks is also beneficial but not strictly required.

Specialization content 
The Generative AI Engineering with LLMs Specialization consists of seven courses, each with an average duration of 5–8 hours.

The seven courses are as follows:

Course 1: Generative AI and LLMs: Architecture and Data Preparation

Course 2: Generative AI Model Foundations for NLP and Language Understanding

Course 3: Generative AI Language Modeling with Transformers

Course 4: Generative AI Engineering and Fine-Tuning Transformers

Course 5: Generative AI Advanced Fine-Tuning for LLMs

Course 6: Fundamentals of AI Agents Using RAG and LangChain

Course 7: Project: Generative AI Applications with RAG and LangChain

Course 1: Generative AI and LLMs: Architecture and Data Preparation
In this course, you will explore the significance of generative AI in various domains. You will differentiate between generative AI models and learn how LLMs are used to build NLP-based applications. Additionally, you will learn about the libraries and tools used in developing these applications. Finally, you will learn to prepare data to train LLMs by implementing tokenization and creating NLP data loaders. 

Course content 

The Generative AI and LLMs: Architecture and Data Preparation course includes the following modules:

Module 1: Generative AI Architecture

Module 2: Data Preparation for LLMs 

Course 2: Gen AI Foundational Models for NLP & Language Understanding
This course will introduce you to the various aspects of NLP and AI model development. You will learn about the fundamentals of language understanding, including converting words to features. You will also learn about various models such as N-Gram, Word2Vec, and sequence-to-sequence models and the metrics to evaluate the quality of the generated text. Finally, you will implement document classification, build, and train a simple language model, integrate Word2Vec, and develop a sequence-to-sequence model.

Course content 

The Gen AI Foundational Models for NLP & Language Understanding course includes the following modules: 

Module 1: Fundamentals of Language Understanding

Module 2: Word2Vec and Sequence-to-Sequence Models

Course 3: Generative AI-Language Modeling with Transformers
This course covers the fundamental concepts of transformer-based models for NLP. You’ll explore the significance of positional encoding and word embedding, understand attention mechanisms and their role in capturing context and dependencies, and learn about multi-head attention. You’ll learn how to apply transformer-based models for text classification, specifically focusing on the encoder component. You will also learn about decoder-based models, such as GPT, and encoder-based models, such as BERT, and use them for language translation.

Course content 

The Generative AI-Language Modeling with Transformers course includes the following modules: 

Module 1: Fundamental Concepts of Transformer Architecture

Module 2: Advanced Concepts of Transformer Architecture

Course 4: Generative AI Engineering and Fine-Tuning Transformers
In this course, you’ll explore the concepts of PyTorch and Hugging Face and their differences. You’ll also understand how to use pre-trained transformers for language tasks and fine-tune them for special tasks. Further, you’ll fine-tune generative AI models using PyTorch and Hugging Face. Finally, you’ll learn parameter-efficient fine-tuning (PEFT), low-rank adaptation (LoRA), quantized low-rank adaptation (QloRA), model quantization, and prompting in transformers.

Course content 

The Generative AI Engineering and Fine-Tuning Transformers course includes the following modules: 

Module 1: Transformers and Fine-Tuning

Module 2: Parameter Efficient Fine-Tuning (PEFT)

Course 5: Generative AI Advanced Fine-Tuning for LLMs
In this course, you’ll explore the basics of instruction-tuning with Hugging Face, reward modeling, and how to train a reward model. You’ll also learn proximal policy optimization (PPO) with Hugging Face, LLMs as policies, and reinforcement learning from human feedback (RLHF). This course will further delve into direct performance optimization (DPO) with Hugging Face using the partition function.

Course content 

The Generative AI Advanced Fine-Tuning for LLMs course includes the following modules: 

Module 1: Different Approaches to Fine-Tuning

Module 2: Fine-Tuning Causal LLMs with Human Feedback and Direct Preference

Course 6: Fundamentals of AI Agents Using RAG and LangChain
In this course, you’ll learn to generate responses using retrieval-augmented generation (RAG), and how RAG involves encoding prompts into vectors, sorting them, and retrieving related information. You’ll also explore RAG, encoder, and Facebook AI similarity search (FAISS). Further, you’ll explore prompt engineering and in-context learning, where tasks are provided to the model as a prompt. In advanced methods of prompt engineering, you’ll learn zero-shot prompts, few-shot prompts, chain-of-thought (CoT) prompting, and self-consistency. Finally, you’ll explore LangChain and its components, such as documents, chains, and agents. 

Course content

The Fundamentals of Building AI Agents Using RAG and LangChain course includes the following modules: 

Module 1: RAG Framework

Module 2: Prompt Engineering and LangChain

Course 7: Project: Generative AI Applications with RAG and LangChain
This course will allow you to apply your acquired knowledge and skills to a capstone project. You’ll learn about document loaders from LangChain and then use that knowledge to load your document from various sources. Then, you’ll learn about text-splitting strategies and apply them to enhance model responsiveness. You’ll then use Watsonx to embed documents, a vector database to store document embeddings, and LangChain to develop a retriever to fetch documents. Further, you’ll implement RAG to improve retrieval, create a QA bot, and set up a simple Gradio interface to interact with your models. Finally, you will construct a QA bot to answer questions from loaded documents.

Course content 

The Project: Generative AI Applications with RAG and LangChain course includes the following modules: 

Module 1: Document Loader Using LangChain

Module 2: RAG Using LangChain

Module 3: Create a QA Bot to Read Your Document

Learning resources
The courses in this specialization offer various learning assets, such as videos, readings, practice, graded quizzes, and a Capstone project.

Ungraded quizzes, including multiple-choice questions with single or multiple correct answers, are provided in each lesson.

Graded quizzes, including scenario-based, multiple-choice questions with single or multiple correct answers, are included in each course.

One specialization-wide graded assessment is provided, including multiple-choice questions with single or multiple correct answers.

Hands-on project: A peer-graded project at the end of the specialization that aims to test the learner’s ability to apply most of the knowledge and skills developed in the course.

The specialization uses simple, easy-to-understand language to explain the important concepts. You will find a glossary and cheat sheet to understand the technical jargon.

To derive maximum learning from this specialization, actively participate in and complete the various learning engagements offered across the three courses.

Good luck!

------------------


RAG with Pytorch


!pip install --user numpy
!pip install --user torch==2.2.0
!pip install --user sacremoses==0.1.1
!pip install --user torchtext
!pip install --user transformers==4.40.2
!pip install --user matplotlib==3.8.4
!pip install --user sentencepiece==0.2.0
!pip install --user scikit-learn==1.4.2





--------------------------



Fundamentals of AI Agents Using RAG and LangChain
Module 1
Reading: Summary and Highlights
Reading: Summary and Highlights
Congratulations! You have completed this lesson. At this point in the course, you know that: 


RAG is an AI framework that helps optimize the output of large language models or LLMs.

RAG combines retrieved information and generates natural language to create responses.

RAG consists of two main components: the retriever, the core of RAG, and the generator, which functions as a chatbot.

In RAG process: 

The retriever encodes user-provided prompts and relevant documents into vectors, stores them in a vector database, and retrieves relevant context vectors based on the distance between the encoded prompt and documents. 

The generator then combines the retrieved context with the original prompt to produce a response.  

The Dense Passage Retrieval (or DPR) Context Encoder and its tokenizer focus on encoding potential answer passages or documents. This encoder creates embeddings from extensive texts, allowing the system to compare these with question embeddings to find the best match.

Facebook AI Similarity Search, also known as Faiss, is a library developed by Facebook AI Research that offers efficient algorithms for searching through large collections of high-dimensional vectors.

Faiss is essentially a tool to calculate the distance between the question embedding and the vector database of context vector embeddings.

The DPR question encoder and its tokenizer focus on encoding the input questions into fixed-dimensional vector representations, grasping their meaning and context to facilitate answering them.



----------------------


In Context Learning - Prompt Engineering


In context learning means in the context of prompt engineeringEditIn-context learning in prompt engineering refers to the ability of large language models to learn and adapt to new tasks or patterns directly within a single conversation or prompt, without requiring additional training or fine-tuning of the model's parameters.

Here's how it works:

Basic Mechanism

Instead of updating the model's weights, you provide examples, instructions, or context within the prompt itself. The model uses this information to understand what you want and applies that understanding to new, similar situations within the same conversation.

Common Techniques
Few-shot learning - Providing a few examples of input-output pairs to demonstrate the desired behavior:
Translate these phrases to French:
English: Hello → French: Bonjour
English: Thank you → French: Merci
English: Good morning → French: Bonjour
Zero-shot learning - Giving clear instructions without examples:
Classify the following text as positive, negative, or neutral sentiment: [text]
Chain-of-thought prompting - Teaching the model to show its reasoning process:
Solve this step by step:
What is 15% of 240?
Step 1: Convert percentage to decimal: 15% = 0.15
Step 2: Multiply: 240 × 0.15 = 36
Key Advantages

Immediate adaptation without retraining

Flexible and rapid experimentation
No need for large datasets or computational resources for training
Can combine multiple techniques in a single prompt

Limitations 

Performance depends on prompt quality and examples chosen
Context window limits how much information you can provide
May not be as robust as models specifically trained for certain tasks

In-context learning essentially turns the prompt into a temporary "training environment" where the model learns the specific task or style you want it to perform.

-----------------

Why use top-p and top-k?
Top-p (Nucleus Sampling): Limits the generation to the smallest set of tokens whose cumulative probability exceeds the probability threshold p.
Top-k Sampling: Limits the generation to the top k most probable tokens. 
By using these methods, the model can generate more diverse and interesting text while still maintaining a degree of coherence. 


-----------------


Fundamentals of AI Agents Using RAG and LangChain
Module 2
Guided Project: Summarize Private Documents Using RAG, LangChain, and LLMs
Guided Project: Summarize Private Documents Using RAG, LangChain, and LLMs
Clicking on the Launch App button below will launch the cloud based SN labs virtual labs environment with instructions to complete this lab. Your username and email will be shared with SN Labs to authenticate and provision your lab environment.

Note: In case you are unable to access the Launch App button, instructions to complete this lab are also available 

here
.

This course uses a third-party app, Guided Project: Summarize Private Documents Using RAG, LangChain, and LLMs, to enhance your learning experience. The app will reference basic information like your name, email, and Coursera ID.


---------------------


Fundamentals of AI Agents Using RAG and LangChain
Module 2
Summary and Highlights
Summary and Highlights
Congratulations! You have completed this lesson. At this point in the course, you know that: 


LangChain provides an environment for building and integrating large language model (LLM) applications into external data sets and workflow.

LangChain simplifies the integration of language models like GPT-4 and makes it accessible for developers to build natural language processing or NLP applications. 

The components of LangChain are:

Chains, agents, and retriever

LangChain-Core

LangChain-Community 

Generative models understand and capture the underlying patterns and data distribution to resemble the given data sets. Generative models are applicable in generating images, text, and music, augmenting data, discovering drugs, and detecting anomalies. 

Types of generative models are: 

Gaussian mixture models (GMMs)

Hidden Markov models (HMMs)

Restricted Boltzmann machines (RBMs)

Variational autoencoders (VAEs)

Generative adversarial networks (GANs)

Diffusion models

In-context learning is a method of prompt engineering where task demonstrations are provided to the model as part of the prompt.

Prompts are inputs given to an LLM to guide it toward performing a specific task. They consist of instructions and context.

Prompt engineering is a process where you design and refine the prompts to get relevant and accurate responses from AI.

Prompt engineering has several advantages:

It boosts the effectiveness and accuracy of LLMs.

It ensures relevant responses.

It facilitates meeting user expectations.

It eliminates the need for continual fine-tuning.

A prompt consists of four key elements: instructions, context, input data, and output indicator.

Advanced methods for prompt engineering are: zero-shot prompting, few-shot prompting, chain-of-thought prompting, and self-consistency.

Prompt engineering tools facilitate interactions with LLMs. 

LangChain uses “prompt templates,” which are predefined recipes for generating effective prompts for LLMs 

An agent is a key component in prompt applications that can perform complex tasks across various domains using different prompts.

The language models in LangChain use text input to generate text output. 

The chat model understands the questions or prompts and responds like a human.

The chat model handles various chat messages, such as:

HumanMessage

AIMessage

SystemMessage

FunctionMessage

ToolMessage

The prompt templates in LangChain translate the questions or messages into clear instructions.

An example selector instructs the model for the inserted context and guides the LLM to generate the desired output. 

Output parsers transform the output from an LLM into a suitable format.

LangChain facilitates comprehensive tools for retrieval-augmented generation (RAG) applications, focusing on the retrieval step to ensure sufficient data fetching. 

The “Document object” in LangChain serves as a container for data information, including two key attributes, such as page_content and metadata.

The LangChain document loader handles various document types, such as HTML, PDF, and code, from various locations.

LangChain in document retrieves relevant isolated sections from the documents by splitting them into manageable pieces. 

LangChain embeds documents and facilitates various retrievers. 

LangChain is a platform that embeds APIs for developing applications. 

Chains in the LangChain is a sequence of calls. In chains, the output from one step becomes the input for the next step. 

In LangChain, chains first define the template string for the prompt, then create a PromptTemplate using the defined template and create an LLMChain object name.  

In LangChain, memory storage is important for reading and writing historical data. 

Agents in LangChain are dynamic systems where a language model determines and sequences actions, such as predefined chains. 

Agents integrate with tools such as search engines, databases, and websites to fulfill user requests.


------------------------------------


Fundamentals of AI Agents Using RAG and LangChain
Module 2
Course Conclusion
Course Conclusion
Congratulations! You’ve successfully completed the course and are equipped to apply the key concepts in retrieval augmented generation (or RAG), encoders, and FAISS. You are adept at applying the fundamentals of in-context learning and advanced methods of prompt engineering to enhance prompt design and can also use RAG, PyTorch, Hugging Face, LLMs, and LangChain technologies for different applications to acquire job-ready skills.

At this point, you know that:

The chatbot generates responses based on the questions; however, it is challenging to generate responses for specific domains, such as the company’s mobile policy.

RAG process involves encoding prompts into vectors, storing them, and retrieving the relevant ones to produce a response.

The DPR Context Encoder and its tokenizer encode the potential answer passages or documents.

FAISS is a library developed by Facebook AI Research for searching through large collections of high-dimensional vectors.

In-context learning is a technique that incorporates task-specific examples into the prompt to boost model performance.

Prompt engineering enhances the effectiveness and accuracy of LLMs by designing prompts that ensure relevant responses without continual fine-tuning.

Advanced prompt engineering methods like zero-shot, few-shot, Chain-of-Thought prompting, and self-consistency enhance LLM interactions.

Tools like LangChain and agents can facilitate effective prompt creation and enable complex, multi-domain tasks.

LangChain is an open-source interface that simplifies the application development process using LLMs.

The ‘Document object’ in LangChain serves as a container for data information, including two key attributes, such as page_content and metadata.

The LangChain document loader handles various document types, such as HTML, PDF, and code from various locations.

Chains in LangChain enable sequential processing, where the output from one step becomes the input for the next, streamlining the prompt generation and processing workflow.

Agents in LangChain dynamically sequence actions, integrating with external tools like search engines and databases to fulfill complex user requests.


--------------------------------


Fundamentals of AI Agents Using RAG and LangChain
Module 2
Congratulations and Next Steps
Congratulations and Next Steps
Congratulations on completing this Fundamentals of AI Agents using RAG and LangChain! We hope you enjoyed it and find great satisfaction using your new skills in the workplace or elsewhere.

This course is part of the 
Generative AI Engineering with LLMs Specialization
.

This course will teach you how RAG is used to generate a response when the model is not pre-trained. You will learn about the RAG process, context, question encoders with their tokenizers, and FAISS (Facebook AI similarity search) library. Additionally, you will gain insight into in-context learning, advanced methods of prompt engineering and its key elements, and prompt templates with LangChain. Furthermore, you will deep dive into the LangChain tools, components, chat models, and document loader. The course also covers how LangChain chains and agents are used to develop applications. Hands-on labs have been strategically placed in the course to help learners acquire job-ready skills with these tools and technologies.

Through practical labs, you will use RAG with PyTorch to evaluate content appropriateness in songs and Hugging Face to retrieve relevant information from a large dataset and develop an agent using integrated LLM, LangChain, and RAG technologies.

This course is designed for individuals with a working knowledge of Python, PyTorch, neural networks, and machine learning and is best suited for you to advance your AI career.

Next Steps

To advance your learning in this course, you can explore the following links:

https://pytorch.org/

https://www.ibm.com/topics/natural-language-processing

As the next step, you are encouraged to explore other courses in this specialization. The specialization includes the following courses:

Course 1: Generative AI and LLMs: Architecture and Data Preparation

Course 2: Generative AI Model Foundations for NLP and Language Understanding

Course 3: Generative AI Language Modeling with Transformers

Course 4: Generative AI Engineering and Fine-Tuning Transformers

Course 5: Advanced Fine-Tuning with Generative LLMs

Course 6: Fundamentals AI Agents using RAG and LangChain

Course 7: Project: Generative AI Applications with RAG and LangChain

To learn more and develop your AI expertise, we encourage you to explore other courses, such as:

Introduction to Artificial Intelligence (AI)
 or 

Building AI-Powered Chatbots without Programming

We also encourage you to leave your feedback and rate this course so that we can continue to improve our content. 

Good luck!


-----------------


Fundamentals of AI Agents Using RAG and LangChain
Module 2
Thanks from the Course Team
Thanks from the Course Team
The entire course team thanks you for taking this course. We hope you enjoyed it, and we wish you the best of luck in applying your new knowledge and skills!

We encourage you to rate the course and provide a review. Your feedback is much appreciated.  

Best regards, 

Course Team 

- - - 

This course has been brought to you through the involvement of the following team of contributors: 

Primary Instructor: Joseph Santarcangelo, Ashutosh Sagar, Fateme Akbari

Instructional Designer: Gagandeep Singh Bawa, Bhavika Chhatbar

Lab Author: Ashutosh Sagar, Wojciech 'Victor' Fulmyk, Fateme Akbari, Kang Wang

Other Contributors and Staff
Project Lead: Rav Ahuja

Production Team
Publishing: Kushal Jha, Kunal Merchant

QA: Prashant Juyal, Praveen Thapliyal, Pornima More, Rahul Rawat

Project Manager: Vishali (Karpagam Sangameswaran)

Video Production: Prayas Gupta, Saad Ali, Saad Ali, Vaishali Rani

Narration: Brayden Medlin

Teaching Assistants and Forum Moderators: Anamika Agarwal, Malika Singla, Lavanya T.S.


-----------------