import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Step 1: Load datasets
print("Loading datasets...")
anime_data = pd.read_csv('anime.csv')  # Ensure anime.csv is in your working directory
ratings_data = pd.read_csv('rating.csv')  # Optional dataset for collaborative filtering

# Step 2: Preprocess data
print("Preprocessing data...")
anime_data = anime_data.dropna()  # Drop rows with missing values

# Split genres into lists for multi-genre handling
anime_data['genre'] = anime_data['genre'].str.split(', ')

# Flatten the genres using one-hot encoding
anime_data = anime_data.explode('genre').reset_index(drop=True)

# Encode genres numerically
genre_encoder = LabelEncoder()
anime_data['genre_encoded'] = genre_encoder.fit_transform(anime_data['genre'])

# Scale ratings and members
scaler = StandardScaler()
anime_data[['rating', 'members']] = scaler.fit_transform(anime_data[['rating', 'members']])

# Step 3: Build KNN model
print("Building KNN model...")
features = anime_data[['genre_encoded', 'rating', 'members']]
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(features)

# Step 4: Define recommendation function
def recommend_anime(genre, user_rating, user_members):
    user_genre = genre_encoder.transform([genre])[0]  # Encode user input genre
    user_features = [[user_genre, user_rating, user_members]]
    
    # Get nearest neighbors
    distances, indices = knn.kneighbors(user_features)
    
    # Fetch recommendations
    recommendations = anime_data.iloc[indices[0]]
    return recommendations[['name', 'genre', 'rating']].drop_duplicates()

# Step 5: Example user input
print("\nEnter your preferences:")
user_genre = input("Preferred genre (e.g., Action, Comedy): ")
user_rating = float(input("Minimum acceptable rating (1-10): "))
user_members = int(input("Estimated popularity (number of members): "))

# Normalize user inputs
user_rating_scaled = scaler.transform([[user_rating]])[0][0]
user_members_scaled = scaler.transform([[user_members]])[0][1]

# Step 6: Get recommendations
recommended_animes = recommend_anime(user_genre, user_rating_scaled, user_members_scaled)

# Step 7: Print results
print("\nRecommended Animes for You:")
print(recommended_animes)