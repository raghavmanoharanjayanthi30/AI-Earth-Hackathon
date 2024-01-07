from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import spacy
#from google.colab import drive
#drive.mount('/content/drive')
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


nouns_df = pd.read_csv("data/AI_Earth_Hackathon_unique_nouns_processed.csv")
problems_solutions_sorted_df = pd.read_csv("data/AI_Earth_Hackathon_problem_solution_sorted.csv")

from sklearn.metrics.pairwise import cosine_similarity
nlp = spacy.load("en_core_web_sm")

def remove_stop_punct(text):
  #remove stop words and punctuations
  if isinstance(text, str):
    doc = nlp(text)
    return " ".join([token.text for token in doc if not token.is_stop and not token.is_punct])
  else:
    return text

word_count_mapping = {}
columns_nouns_df = nouns_df.columns
for word in columns_nouns_df:
  word_count_mapping[word] = sum(nouns_df[word])

def compute_uniqueness_rating(solution):
  #for all nouns across all solutions, check how many of them appear in this particular solution
  word_in_vector = [0]*len(columns_nouns_df)
  for i, word in enumerate(columns_nouns_df):
    if word in solution:
      word_in_vector[i] = 1

  count = 0
  for i, val in enumerate(word_in_vector):
    #if the word appears in the solution and it appears only in atmost 50 of the 1300 solutions in the dataset, increase count (word counts as unique)
    if word_in_vector[i] == 1 and word_count_mapping[columns_nouns_df[i]] <= 50:
      count += 1

  #rating based on how many indicator words appear
  if count >= 40:
    return 3
  elif count < 40 and count >= 30:
    return 2
  else:
    return 1

def compute_similarity_rating(problem, solution):
  #compute how contextually and semantically similar the problem and solution are to assess the relevance of the solution
  problem_solution_pair = [problem, solution]
  embeddings = model.encode(problem_solution_pair,convert_to_tensor=False)
  cosine_similarity_val = cosine_similarity(embeddings)[0,1]

  #rating based on cosine similarity
  if cosine_similarity_val >= 0.6:
    return 3
  elif cosine_similarity_val >= 0.45:
    return 2
  else:
    return 1

def compute_scalability_rating(solution):
  #see how many of these indicator words for scalability are there in your solution
  scalability_words = ["scale", "expand", "increased demand", "scale-up", "scale up", "scalability"]
  total = 0
  if isinstance(solution, str):
    for word in scalability_words:
      if word in solution.lower():
        total += 1

  #rating based on how many indicator words appear
  if total >= 2:
    return 3
  elif total == 1:
    return 2
  else:
    return 1

def compute_feasibility_rating(solution):
  #see how many of these indicator words for feasibility are there in your solution
  feasibility_words = ["feasible", "feasibility", "viable", "viability","practical", "implement", "implementable", "doable"]
  total = 0
  if isinstance(solution, str):
    for word in feasibility_words:
      if word in solution.lower():
        total += 1

  #rating based on how many indicator words appear
  if total >= 2:
    return 3
  elif total == 1:
    return 2
  else:
    return 1


def compute_impact_rating(solution):
  #see how many of these indicator words for impact are there in your solution
  impact_words = ["environmental impact", "financial impact", "efficiency", "productivity", "cost savings", "save cost", "save time", "time savings", "flexibility"]
  total = 0
  if isinstance(solution, str):
    for word in impact_words:
      if word in solution.lower():
        total += 1

  #rating based on how many indicator words appear
  if total >= 2:
    return 3
  elif total == 1:
    return 2
  else:
    return 1

def circular_economy_relevance(solution):
  #see how many of these indicator words for circular economy are there in your solution
  circular_economy_words = [
    "regenerate", "regeneration", "recycling", "waste reduction", "waste elimination",
    "pollution reduction", "circulation", "remanufacturing", "increased product lifespan",
    "zero waste", "resource recovery", "material reusability", "reusable", "lifespan", "eco-friendly", "renewable", "sustainable",
    "circular sourcing", "circular supply chain", "green", "environmental friendly",
    "sustainable", "sustainability", "reuse"]
  total = 0

  if isinstance(solution, str):
    for word in circular_economy_words:
      if word in solution.lower():
        total += 1

  #rating based on how many indicator words appear
  if total >= 3:
    return 3
  elif total >= 1:
    return 2
  else:
    return 1

def compute_market_potential_rating(solution):
  #see how many of these indicator words for market potential are there in your solution
  market_potential_words = ["competitive advantage", "target audience", "market demand",
                            "market share","revenue", "sales", "profitability", "competitor",
                            "market expansion", "industry growth", "growth rate",
                            "market trends", "expanding market", "market penetration",
                            "unique selling proposition", "USP", "product differentiation",
                            "technological advancements"]
  total = 0
  if isinstance(solution, str):
    for word in market_potential_words:
      if word in solution.lower():
        total += 1

  #rating based on how many indicator words appear
  if total >= 2:
    return 3
  elif total == 1:
    return 2
  else:
    return 1


def compute_overall_rating(problem, solution):
  problem = problem.replace('\n', '')
  solution = solution.replace('\n', '')
  problem_processed = remove_stop_punct(problem)
  solution_processed = remove_stop_punct(solution)

  problem_relevance_rating = compute_similarity_rating(problem_processed, solution_processed)
  scalability_rating = compute_scalability_rating(solution_processed)
  feasibility_rating = compute_feasibility_rating(solution_processed)
  impact_rating = compute_impact_rating(solution_processed)
  market_potential_rating = compute_market_potential_rating(solution_processed)
  ce_rating = circular_economy_relevance(solution_processed)
  uniqueness_rating = compute_uniqueness_rating(solution_processed)

  all_ratings = [problem_relevance_rating, scalability_rating, feasibility_rating,
                 impact_rating, market_potential_rating, ce_rating, uniqueness_rating]
  avg_rating = sum(all_ratings)/len(all_ratings)

  ratings_dict = {
      #"problem": problem,
      #"solution": solution,
      "relevance to problem": problem_relevance_rating,
      "scalability": scalability_rating,
      "feasibility": feasibility_rating,
      "impact": impact_rating,
      "market potential": market_potential_rating,
      "adherence to circular economy principles": ce_rating,
      "novelty": uniqueness_rating,
      #"average_rating": avg_rating
  }

#   new_row_df = pd.DataFrame(ratings_dict, index=[0])

#   # Append the new row to the existing DataFrame
#   #ignore_index=True to reset index of combined dataframe
#   problems_solutions_sorted_df_updated = pd.concat([problems_solutions_sorted_df, new_row_df], ignore_index=True)


#   # Sort the DataFrame by the 'average_rating' column
#   problems_solutions_sorted_df_updated = problems_solutions_sorted_df_updated.sort_values(by=['average_rating'], ascending=False)

#   problems_solutions_sorted_df_updated.reset_index(drop=True, inplace=True)

#   row_index = problems_solutions_sorted_df_updated.index[
#     (problems_solutions_sorted_df_updated['problem'] == problem) &
#     (problems_solutions_sorted_df_updated['solution'] == solution)]

#   rank = row_index.values[0] + 1
#   row_count = problems_solutions_sorted_df_updated.shape[0]

#   summary = f"Your solution for this problem ranked {rank} out of {row_count} problem solution pairs"

#   return ratings_dict, round(avg_rating, 2), summary
  return ratings_dict
