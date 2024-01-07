import streamlit as st
import pandas as pd
from utils import *
from zsl_model import inference_zeroshot
from nlp_model import compute_overall_rating
from gemini_model import calculate_gemini_scores, generate_explanations


st.set_page_config(layout="wide")

def prediction(problem_input, solution_input):
    ''' Format is: {'novelty': 1, 'scalability': 1, 'feasibility': 1, 'impact': 1, 'market potential': 1, 'adherence to circular economy principles': 1} '''
    nlp_solution = compute_overall_rating(problem_input, solution_input)
    zsl_solution = inference_zeroshot(solution_input)
    gemini_solution = calculate_gemini_scores(problem_input, solution_input)
    zsl_solution['relevance to problem'] = nlp_solution['relevance to problem']
    gemini_solution['relevance to problem'] = nlp_solution['relevance to problem']
    # put relevanve to problem first
    zsl_solution = {k: zsl_solution[k] for k in ['relevance to problem', 'novelty', 'scalability', 'feasibility', 'impact', 'market potential', 'adherence to circular economy principles']}
    # majority vote between the 3 models
    for k in zsl_solution.keys():
        solution_list = [zsl_solution[k], gemini_solution[k], nlp_solution[k]]
        # if all different
        if len(set(solution_list)) == 3:
            zsl_solution[k] = 2
        else:
            zsl_solution[k] = max(set(solution_list), key = solution_list.count)
    return zsl_solution

# Streamlit interface
def main():
    st.title("AI EarthHack")

    st.info("You can either enter a problem and solution to evaluate, or upload a csv file. Our models will analyze the problem and the solution and give based on different criteria.")

    # Text input from user
    problem_input = st.text_area("Enter the problem here:", value="Majority of cargo containers are abandoned post-usage, leading to extensive metal waste and environmental degradation, while urban areas face a shortage of parking spaces due to rapid urbanization.")
    user_input = st.text_area("Enter your solution here:", value="We can leverage this situation by recycling and refurbishing these unused cargo containers to serve as portable garages. This would not only help to declutter the marine and dockyard spaces but also provide a sustainable, economic, and space-efficient solution for parking in congested urban areas. The containers can be refurbished with minimal energy input and subsequently leased out, creating a new revenue stream while contributing positively to the environment. This approach aligns with the circular economy‚Äôs principles of sharing, leasing and recycling.")

    # separate the 2 text areas with a delimiter
    st.write("---")
    # or, user can also upload a csv file with 2 columns for problem and solution
    uploaded_file = st.file_uploader("Or upload a csv file", type="csv")

    if st.button("Evaluate"):
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            scores = {}
            for index, row in df.iterrows():
                scores[index] = prediction(row['problem'], row['solution'])
            df_scores = pd.DataFrame.from_dict(scores, orient='index')
            # add total score column
            df_scores['total_score'] = df_scores.sum(axis=1)
            df_scores.columns = ['relevance to problem', 'novelty', 'scalability', 'feasibility', 'impact', 'market potential', 'adherence to circular economy principles', 'total_score']
            df = pd.concat([df, df_scores], axis=1)
            st.dataframe(df)
            # button to download the csv file
            st.download_button(
                label="Download CSV",
                data=df.to_csv().encode("utf-8"),
                file_name="ai_earthhack_scores.csv",
                mime="text/csv",
            )

        else:
            if user_input:
                # Get predictions
                scores = prediction(problem_input, user_input)

                # Get explanations
                explanation = generate_explanations(scores, problem_input, user_input)

                # Display scores
                st.subheader("Evaluation Results:")
                for criterion, score in scores.items():
                    st.write(f"{criterion.capitalize()}: {score}/3")
                    st.progress(score / 3)
            else:
                st.write("Please enter a solution to evaluate.")
        
            total_score = sum(scores.values())
            animation(total_score, "Total Score (from 0 to 21)", "green")
            # Display explanation
            with st.expander("See explanations for the scores"):
                for criterion, explanation in explanation.items():
                    # remove " " from beginning and end of string if it exists
                    explanation = explanation.strip()
                    if explanation[0] == '"' and explanation[-1] == '"':
                        explanation = explanation[1:-1]
                    st.write(f"**{criterion.capitalize()}**: {explanation}")

    # section to explain how scores are calculated
    with st.expander("How are scores calculated?"):
        st.write('''
            We combine the results of the 3 following methods:
            - Zero-Shot Classification Model Using BART-Large-MNLI: in our application, this model interprets the user's text input and evaluates it against predefined labels, even if those labels were not present in the model's training data. This offers flexibility and broad applicability in understanding and categorizing diverse solution statements
            - Gemini Model API: this model is good at assessing the solution's tone, context, and underlying assumptions, providing a nuanced understanding that contributes to the overall score
            - NLP and Text Analysis Techniques: we employ a range of Natural Language Processing (NLP) and text analysis techniques to deeply analyze the solution text
        ''')        

if __name__ == "__main__":
    main()
