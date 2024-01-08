# AI-Earth-Hackathon

This tool streamlines the assessment of proposed solutions tailored to circular economy challenges in different industries. Employing advanced Natural Language Processing and Large Language Models, it evaluates criteria like relevance, novelty, scalability, feasibility, impact, market potential, and adherence to circular economy principles. 

By harnessing state-of-the-art AI technology, it enhances the evaluation process, providing valuable insights to guide decision-making on the effectiveness and viability of diverse solutions in the context of circular economy-related problems.

## How to run the demo locally?

### 1. Install the dependencies

- Use python 3.9
- Install the dependencies with `pip install -r requirements.txt`

### 2. Use your own API key

- Create a file named `secrets.toml` in the `.streamlit` folder
- Write your API key with the following format:
```
GEMINI_API_KEY = "xxx"
```

### 2. Run the demo

- Run the demo with `streamlit run main.py`


## How to run the demo online?

- Simply use the following link: https://ai-earth-hackathon.streamlit.app/


## How to use the demo?

- You can either manually enter a problem statement and the proposed solution, or you can upload a CSV file with the same format as the training data.
- When clicking on the "Evaluate" button, our tool will use the different models and combine their results to provide scores for the different criteria, and an overall score.
- If the user has manually entered one problem and solution, the "Explanations of the scores" section provides interpretability for the scores using the Gemini API.
- If the user has uploaded a CSV file, we instead provide a new CSV file with the scores for each row, that the user can download.
