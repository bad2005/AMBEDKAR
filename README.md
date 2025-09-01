# AMBEDKAR


## Abstract
Large Language Models (LLMs) can inadvertently reflect societal biases present in their training data, leading to harmful or prejudiced outputs. In the Indian context, our empirical evaluations across a suite of models reveal that biases around caste and religion are particularly salient. Yet, most existing mitigation strategies are Western-centric and fail to address these local nuances.  

We propose **AMBEDKAR**, a framework inspired by the egalitarian vision of Dr. B. R. Ambedkar, architect of the Indian Constitution, to guide LLM outputs toward fairness, neutrality, and inclusion in line with Articles 14 to 17.  

Our approach introduces a **ConstitutionAware Decoding Layer**, guided by the **AI Constitution of India** and applied only at inference time, without any parameter updates to the base model. We incorporate a **speculative decoding algorithm** that proactively reduces casteist and communal bias during generation. This mitigation layer operates directly within the decoding process, avoiding changes to model internals and lowering the computational and infrastructural costs associated with retraining.  

We reinterpret speculative decoding not merely as an efficiency tool but as a mechanism for fairness. In this framework, a **Small Language Model (SLM)** acts as a potentially biased generator, while a constitutionally guided **Large Language Model (LLM)** serves as the verifier. Rather than accelerating generation, the LLM enforces bias-robust trajectories in the SLM’s outputs. This inversion of roles gives rise to a **fairness-by-speculation paradigm**.  

Our approach yields an **absolute reduction of bias up to 26.41%** compared to baseline methods.  

---

## Technical Report
For a full, comprehensive technical report (50+ pages), please refer to [this link](https://github.com/bad2005/AMBEDKAR/blob/main/A%20Multi%20Level%20Bias%20Elimination%20through%20a%20Decoding%20Approach%20with%20Knowledge%20Augmentation%20for%20Consititutional%20Alignment%20for%20Language%20Models.pdf).  

---

## Dataset
For reproducibility, we have released a **partial dataset and counterfactuals**. The **full dataset** will be made available upon paper acceptance.  

