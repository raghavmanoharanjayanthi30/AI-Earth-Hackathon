import streamlit as st
import pandas as pd
from utils import *
from zsl_model import inference_zeroshot
from nlp_model import compute_overall_rating


def prediction(problem_input, solution_input):
    ''' Format is: {'novelty': 1, 'scalability': 1, 'feasibility': 1, 'impact': 1, 'market potential': 1, 'adherence to circular economy principles': 1} '''
    nlp_solution = compute_overall_rating(problem_input, solution_input)
    print(nlp_solution)
    zsl_solution = inference_zeroshot(solution_input)
    print(zsl_solution)
    zsl_solution['relevance to problem'] = nlp_solution['relevance to problem']
    # put relevanve to problem first
    zsl_solution = {k: zsl_solution[k] for k in ['relevance to problem', 'novelty', 'scalability', 'feasibility', 'impact', 'market potential', 'adherence to circular economy principles']}
    # take the average of the 2 models
    for key in zsl_solution.keys():
        zsl_solution[key] = (zsl_solution[key] + nlp_solution[key]) / 2
    return zsl_solution

# Streamlit interface
def main():
    st.title("AI EarthHack")

    st.info("You can either enter a problem and solution to evaluate, or upload a csv file.")

    # Text input from user
    problem_input = st.text_area("Enter the problem here:")
    user_input = st.text_area("Enter your solution here:")

    # separate the 2 text areas with a delimiter
    st.write("---")
    # or, user can also upload a csv file with 2 columns for problem and solution
    uploaded_file = st.file_uploader("Or upload a csv file", type="csv")

    if st.button("Evaluate"):
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            scores = {}
            for index, row in df.iterrows():
                scores[index] = prediction(row['solution'])
            df_scores = pd.DataFrame.from_dict(scores, orient='index')
            # add total score column
            df_scores['total_score'] = df_scores.sum(axis=1)
            df_scores.columns = ['novelty', 'scalability', 'feasibility', 'impact', 'market potential', 'adherence to circular economy principles', 'total_score']
            df = pd.concat([df, df_scores], axis=1)
            st.dataframe(df)

        else:
            if user_input:
                # Get predictions
                scores = prediction(problem_input, user_input)

                # Display scores
                st.subheader("Evaluation Results:")
                for criterion, score in scores.items():
                    st.write(f"{criterion.capitalize()}: {score}/3")
                    st.progress(score / 3)
            else:
                st.write("Please enter a solution to evaluate.")
        
            total_score = sum(scores.values())
            animation(total_score, "Total Score (from 0 to 21)", "blue")

    # section to explain how scores are calculated
    with st.expander("How are scores calculated?"):
        st.write('''
            We combine the results of the 3 following methods:
            - Zero-shot classification model using BART-large-mnli
            - Gemini model API
            - NLP and text analysis techniques
                 ''')        

if __name__ == "__main__":
    main()
