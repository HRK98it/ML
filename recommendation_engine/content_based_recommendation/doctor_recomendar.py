import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df=pd.read_excel('doctor_review.xlsx')

# Combine useful text features for recommendation
df['profile'] = (
    df['Specialist'].astype(str) + " " +
    df['Location'].astype(str) + " " +
    df['Hospital_Name'].astype(str) + " " +
    df['Reviews'].astype(str)
)

# Vectorize using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_mat = tfidf.fit_transform(df['profile'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_mat, tfidf_mat)

# Recommendation function based on speciality
def get_recommendations_by_speciality(speciality, top_n=5):
    # Filter doctors of that speciality
    spec_indices = df[df['Specialist'].str.contains(speciality, case=False, na=False)].index
    
    if len(spec_indices) == 0:
        return f"No doctors found for speciality: {speciality}"
    
    # Take first doctor of that speciality as reference profile
    idx = spec_indices[0]
    
    # Get similarity scores for this doctor with all others
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    recommended = []
    for i, score in sim_scores:
        if i != idx and df.loc[i, 'Specialist'].lower() == speciality.lower():
            recommended.append((
                df.loc[i, 'Name_of_dr'],
                df.loc[i, 'Specialist'],
                df.loc[i, 'Ratings'],
                df.loc[i, 'Location']
            ))
        if len(recommended) >= top_n:
            break
    return recommended

a=input('Enter the specilisty of doctor you want: ')
get_recommendations_by_speciality(a, top_n=5)
