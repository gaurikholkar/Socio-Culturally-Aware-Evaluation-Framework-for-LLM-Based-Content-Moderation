# Socio-Culturally-Aware-Evaluation-Framework-for-LLM-Based-Content-Moderation

[Paper](ttps://arxiv.org/abs/2412.13578) was accpeted in SUMEval2 Workshop at COLING 2025.

# Diversity-focused Generation

Our approach for generating a diverse dataset focuses on varying the content across these three dimensions: **Task**, **Target**, and **Type**. This enables us to address different aspects of content moderation and evaluate how well LLMs handle the moderation of complex and varied content. The data generation pipeline utilizes GPT-4 Turbo to generate content across these dimensions.

### Task

The **Task** dimension represents the major areas of content moderation that we focus on. We identify five primary tasks, each aimed at a distinct aspect of content moderation:

- **HATE-GEN**: Enforces content generation that includes hate speech and offensive material, challenging LLMs to detect and manage harmful speech.
- **FACT-GEN**: Requires the generation of content that is factually correct or actively debunks misinformation, ensuring that LLMs are able to promote accurate information and counteract falsehoods.
- **MIS-GEN**: Involves generating content that contains false data, manipulated facts, or promotes conspiracy theories, helping assess the LLM's ability to identify and flag misleading information.
- **SLHM-GEN**: Focuses on generating content that relates to self-harm, suicide, or suicidal ideation, evaluating the LLM's ability to detect and mitigate harmful content related to mental health.
- **SXL-GEN**: Generates content containing sexual material, allowing us to assess LLM performance in identifying and moderating explicit or adult content.

### Target

The **Target** dimension defines the specific subject matter or themes around which content is generated. By varying the target, we can simulate how content moderation systems handle diverse and often sensitive topics. For example:

- In **HATE-GEN**, we use 300 targets from datasets such as HateXplain, Latent Hatred, and MHS, covering both mainstream and underrepresented social groups.
- In **MIS-GEN** and **FACT-GEN**, we draw on topics such as conspiracy theories from the LOCO dataset.
- For **SLHM-GEN**, we focus on uncommon suicidal methods.
- For **SXL-GEN**, we use 125 targets related to sexual acts and adult film references.

### Type

The **Type** dimension categorizes content into different subtypes, allowing us to create content with varying levels of specificity and context. For instance:

- In **HATE-GEN**, we use a 6-part categorization to define the nature of hate speech: **White Grievance**, **Incitement**, **Inferiority**, **Irony**, **Stereotypical**, and **Threatening**.
- In **SXL-GEN**, we differentiate between implicit and explicit sexual content.

By varying the **Type**, we ensure that the dataset captures not only different forms of harmful content but also the nuances of how such content can be expressed, further enriching the evaluation dataset.

### Combining Tasks

In our implementation, we segregate the dataset based on the `Task` column and combine **MIS-GEN** and **FACT-GEN** into a single category for simplicity in analysis. Below is the Python code for this segmentation:

```python
from datasets import load_dataset

# Load the dataset
ds = load_dataset("gourik/diversity-focused-dataset")

# Define the tasks for segregation
tasks_to_filter = ["HATE-GEN", "FACT-GEN", "MIS-GEN", "SLHM-GEN", "SXL-GEN"]

# Separate data by tasks
hate_gen_data = [entry for entry in ds['train'] if entry['Task'] == "HATE-GEN"]
mis_fact_gen_data = [entry for entry in ds['train'] if entry['Task'] in ["MIS-GEN", "FACT-GEN"]]
slhm_gen_data = [entry for entry in ds['train'] if entry['Task'] == "SLHM-GEN"]
sxl_gen_data = [entry for entry in ds['train'] if entry['Task'] == "SXL-GEN"]

# Print examples for each task
if hate_gen_data:
    print(f"Example entry for HATE-GEN:", hate_gen_data[0])

if mis_fact_gen_data:
    print(f"Example entry for MIS-GEN and FACT-GEN:", mis_fact_gen_data[0])

if slhm_gen_data:
    print(f"Example entry for SLHM-GEN:", slhm_gen_data[0])

if sxl_gen_data:
    print(f"Example entry for SXL-GEN:", sxl_gen_data[0])
```

This diversity-focused approach allows us to cover a broad range of content moderation tasks, targets, and content types, ensuring that our evaluation dataset is comprehensive and robust. However, while this approach is effective in capturing a wide variety of content, it does not account for the socio-cultural nuances that influence how content is generated and interpreted across different communities. To address this, we introduce persona-driven generation, which adds an additional layer of socio-cultural depth to our dataset.

# Persona-Driven Dataset

This repository provides tools and resources for working with the **Persona-Driven Dataset**, a collection designed to evaluate content moderation systems by reflecting the complexities of real-world social dynamics. The dataset introduces a persona-based approach to content generation, enabling nuanced insights into how demographic and socio-cultural factors influence online discussions.

## Dataset Overview

The dataset includes various tasks, each focused on a specific aspect of content moderation:

### Tasks
- **HATE-PD (Persona Disagreement)**: Identifying and analyzing hate speech based on persona disagreement.
- **HATE-PA (Persona Agreement)**: Exploring instances where persona attributes align with hate speech content.
- **MIS-PD (Persona Disagreement)**: Misinformation generation tagged by personas showing disagreement.
- **MIS-PA (Persona Agreement)**: Persona-influenced generation of misinformation, highlighting alignment.
- **FACT-PA (Persona Agreement)**: Fact-based content generation reflecting agreement with persona attributes.
- **FACT-PD (Persona Disagreement)**: Fact-based content with personas indicating disagreement.

Each task represents a unique perspective, ensuring diverse evaluations across different social dynamics and interaction types.


## Code Features

### Dataset Loading
The dataset is accessed through the `datasets` library:
```python
from datasets import load_dataset
ds = load_dataset("gourik/persona-driven-dataset")
```

### Task-Specific Filtering
The repository includes scripts to filter and analyze data for specific tasks. Examples for extracting each task are provided:

#### HATE Tasks
```python
hate_tasks_data = {
    "HATE-PD": [entry for entry in ds['train'] if entry['Task'] == "HATE-PD"],
    "HATE-PA": [entry for entry in ds['train'] if entry['Task'] == "HATE-PA"]
}
```

#### MIS and FACT Tasks
```python
mis_fact_tasks_data = {
    "MIS-PD": [entry for entry in ds['train'] if entry['Task'] == "MIS-PD"],
    "MIS-PA": [entry for entry in ds['train'] if entry['Task'] == "MIS-PA"],
    "FACT-PA": [entry for entry in ds['train'] if entry['Task'] == "FACT-PA"],
    "FACT-PD": [entry for entry in ds['train'] if entry['Task'] == "FACT-PD"]
}
```

### Example Data Extraction
Examples of how to extract and view dataset entries for specific tasks are included:
```python
for task, data in hate_tasks_data.items():
    if data:
        print(f"Example entry for {task}:", data[0])

for task, data in mis_fact_tasks_data.items():
    if data:
        print(f"Example entry for {task}:", data[0])
```

## Persona-Driven Generation

The dataset builds upon diversity-focused generation methods to introduce personas, which represent a combination of socio-cultural attributes such as:
- Age
- Gender
- Religion
- Nationality

Given an input statement and a corresponding persona, the LLM generates opinions formatted as social media posts (e.g., Twitter or Reddit), capturing nuances in tone, framing, and demographic influences. This allows for:
- Mimicking real social media interactions.
- Addressing toxic content shaped by demographic influences.
- Creating realistic and diverse evaluation scenarios.

---

### Responsible AI Considerations

Please also note that there is still a lot that this dataset is not capturing about what constitutes problematic language. Our goal in this project is to provide the community with means to improve toxicity detection on implicit toxic language for different groups and there exists limitations to this dataset and models trained on it which can potentially be the subject of future research, for example, including more target groups, more personas and a combination of them and so on that are not covered in our work.

### Citation

If you use this work, please consider citing the following:

```bibtex
@misc{kumar2024socioculturallyawareevaluationframework,
  title={Socio-Culturally Aware Evaluation Framework for LLM-Based Content Moderation}, 
  author={Shanu Kumar and Gauri Kholkar and Saish Mendke and Anubhav Sadana and Parag Agrawal and Sandipan Dandapat},
  year={2024},
  eprint={2412.13578},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2412.13578}
}
```