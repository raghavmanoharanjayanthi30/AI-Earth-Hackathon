import streamlit as st
import random
from utils import *


def prediction(text):
    return {'novelty': random.randint(1, 3), 
            'scalability': random.randint(1, 3), 
            'feasibility': random.randint(1, 3),
            'impact': random.randint(1, 3),
            'market potential': random.randint(1, 3),
            'adherence to circular economy principles': random.randint(1, 3)}

# Streamlit interface
def main():
    st.title("AI EarthHack")

    # Text input from user
    problem_input = st.text_area("Enter your problem text here:")
    user_input = st.text_area("Enter your solution text here:")

    if st.button("Evaluate"):
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


if __name__ == "__main__":
    main()
