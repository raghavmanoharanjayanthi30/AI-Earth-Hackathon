import streamlit as st
import pandas as pd
from utils import *
from zsl_model import inference_zeroshot


def prediction(text):
    ''' Format is: {'novelty': 1, 'scalability': 1, 'feasibility': 1, 'impact': 1, 'market potential': 1, 'adherence to circular economy principles': 1} '''
    return inference_zeroshot(text)

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
            st.write(scores)

        else:
            if user_input:
                # Get predictions
                scores = prediction(user_input)

                # Display scores
                st.subheader("Evaluation Results:")
                for criterion, score in scores.items():
                    st.write(f"{criterion.capitalize()} Score: {score}/3")
                    st.progress(score / 3)
            else:
                st.write("Please enter a solution to evaluate.")
        
            total_score = sum(scores.values())
            animation(total_score, "Total Score (from 0 to 18)", "blue")

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
