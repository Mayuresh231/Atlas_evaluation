# Atlas Evaluation

Two Python backend scenarios, maintained together in one repository. Each scenario has its own setup instructions, dependencies, and tests.

| Scenario | Project | Documentation |
| --- | --- | --- |
| 1 | **SupportIQ:** customer support classification with TF-IDF, logistic regression, offline training, and a FastAPI API. | [Classification README](Scenario_1_classification/README.md) |
| 2 | **Engineering Chatbot:** document and image retrieval, grounded answers, and validated citations using PostgreSQL/pgvector and OpenAI. | [Chatbot README](scenario-2-chatbot/README.md) |

## Get started

Clone this repository, then follow the README inside the scenario you want to run:

```bash
git clone https://github.com/Mayuresh231/Atlas_evaluation.git
cd Atlas_evaluation
```

- For classification: `cd Scenario_1_classification`.
- For the engineering chatbot: `cd scenario-2-chatbot`.

Run each project's setup, training, migration, and test commands from that project's directory. Both APIs use port 8000 by default; choose a different port for one if running them together.

The classification dataset is synthetic demonstration data. The chatbot requires a configured OpenAI API key and PostgreSQL. Local environment files, virtual environments, generated model artifacts, reports, and uploaded documents are excluded from Git.
