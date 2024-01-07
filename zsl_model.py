import pandas as pd
from tqdm import tqdm
from transformers import pipeline


df = pd.read_csv('data/AI EarthHack Dataset.csv', encoding='latin1')

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

metrics = ["novelty", "scalability", "feasibility", "impact", "market potential", "adherence to circular economy principles"]
candidate_labels_dic = {
    "novelty": ["Not a novel solution", "Novel solution", "Highly novel solution"],
    "scalability": ["Not a scalable solution", "Scalable solution ", "Highly scalable solution"],
    "feasibility": ["Not a feasible solution", "Feasible solution", "Highly feasible solution"],
    "impact": ["Not an impactful solution", "Impactful solution", "Highly impactful solution"],
    "market potential": ["Not a marketable solution", "Marketable solution", "Highly marketable solution"],
    "adherence to circular economy principles": ["Solution does not adhere to circular economy principles", "Solution adheres to circular economy principles", "Solution highly adheres to circular economy principles"],
}

def inference_zeroshot(text_solution):
    ''' output a dictionary with the score for each metric'''
    scores = {}
    for m in metrics:
        candidate_labels = candidate_labels_dic[m]
        classification = classifier(text_solution, candidate_labels)
        best_labels = classification["labels"][0]
        scores[m] = candidate_labels.index(best_labels)
    return scores


if __name__ == "__main__":
    batch_size = 16

    data = df["solution"]
    results = []
    for m in metrics:
        results = []
        for i in tqdm(range(0, len(data), batch_size)):
            batch = list(data[i:i + batch_size])
            candidate_labels = candidate_labels_dic[m]
            classification = classifier(batch, candidate_labels)
            best_labels = [classification[i]["labels"][0] for i in range(len(classification))]
            results += [candidate_labels.index(l) for l in best_labels]
        df[m+"_ZSl"] = results
        df.to_csv('data/AI EarthHack Dataset ZSL.csv', index=False)

    print(df.head())
