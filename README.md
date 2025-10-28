# DSPy Tutorial

This Repository contains files from learning dspy.  
This includes a document generation for Astriks Hackathon Project.

---

### Execute Document Generation

The Document generation is set up in a single file: `generate_encounter_narrative.py`  

The Generation may be executed via "05_generation_runs.ipynb" or directly via commandline.

#### MLFlow

MlFlow is used for development and evaluation.  
Deactivate mlflow by passing `--no-mlflow` to the execution  

Or run mlflow for tracing lm calls:

```sh
cd mlflow
docker compose up
```

and access  
`localhost:5000`

