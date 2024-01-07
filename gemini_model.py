import google.generativeai as genai

import config_gemini_model

#Setting up Gemini model
genai.configure(api_key = config_gemini_model.api_key)
model = genai.GenerativeModel('gemini-pro')

metrics = ["novelty", "scalability", "feasibility", "impact", "market potential", "adherence to circular economy principles"]



"""
:param problem: string containing the given problem
:param solution: string containing the given solution

:returns: a dictionary, formatted as follows - {key: metric, value: score}
"""
def calculate_gemini_scores(problem, solution):

    output_scores = {}

    for m in metrics:
        
        metric = m

        if m == "impact":
            metric = "potential impact"
        
        output_scores[m] = model.generate_content(f'Given the following input problem-solution pair, return an integer on a scale of 1-3 (inclusive) representing the {metric} of the solution given the problem.\n\nProblem: {problem}\n\nSolution: {solution}').text

    return output_scores





"""
:param problem: string containing the given problem
:param solution: string containing the given solution

:returns: a dictionary, formatted as follows - {key: metric, value: [score, gemini explanation]}
"""
def calculate_gemini_scores_and_explanations(problem, solution):


    output_scores = {}

    for m in metrics:
        
        metric = m
        if m == "impact":
            metric = "potential impact"
        
        output_scores[m] = parse_array(model.generate_content(f'Given the following input problem-solution pair, return an integer on a scale of 1-3 (inclusive) representing the {metric} of the solution given the problem. Return the answer as an array in the following format: [score (as an integer), justification for the given score].\n\nProblem: {problem}\n\nSolution: {solution}').text)

    return output_scores


def parse_array(string):
  string = string[1:-1]
  elements = string.split(",", 1)
  return elements




# # Testing

# if __name__ == "__main__":

#     problem = "With an increase in e-waste and the environmental hazards it brings about, there is a need for comprehensive solutions that reduce the total volume of electronics produced, prolong their lifecycle, and ensure their proper disposal.  "
#     solution = "I propose a novel concept - Electronic Device Libraries (EDLs). This model is inspired by the concept of traditional libraries but applied to electronics like laptops, tablets, cameras, and more. EDLs would be community hubs where members can borrow devices according to their needs, thus reducing the need for individual ownership.   This model not only increases the usage lifecycle of each device but also reduces resource extraction for production and cuts down e-waste. Devices would be effectively repurposed and recycled at the end of their lifespan within the EDL system, ensuring proper e-waste management.  The EDL system would generate revenue through membership subscriptions and services such as data security and device maintenance. Additionally, businesses that donate devices to EDLs could benefit from tax deductions or green certifications, offering financial value.  To facilitate feasibility and scalability, partnerships with local governments and non-government organizations could support set up and running of EDLs at community centers, libraries, or schools. Scaling up could involve opening more EDLs in different areas or incorporating more device types. With communities growing more environmentally conscious, favorable reception and quick adoption of EDLs can be predicted. Via public education and engagement, the shift from ownership to shared use can be accelerated."

#     # print(calculate_gemini_scores_and_explanations(problem, solution))

#     output_scores = calculate_gemini_scores_and_explanations(problem, solution)

#     for key in output_scores:
#         print(key, "->", output_scores[key][1])