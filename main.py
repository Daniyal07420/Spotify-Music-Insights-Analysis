import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Data Load
data = pd.read_csv("c:\\Users\\Lenovo\\Desktop\\spotify_data clean.csv")
df = pd.DataFrame(data)
print(df.head())
# Data Cleaning
df = df.dropna()
df = df.drop_duplicates()
df = df.reset_index(drop=True)
print(df.info())
print(df.describe())
# Most popular tracks
most_popular_tracks = df.nlargest(10, "track_popularity")[["track_name", "track_popularity"]]
print("Most Popular Tracks:\n", most_popular_tracks)
# Compare popularity vs duration
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="track_duration_min", y="track_popularity", color="red")
axes = plt.gca()
axes.set_facecolor("lightgray")   # axes ka background
plt.gcf().patch.set_facecolor("lightblue")  # figure ka background
plt.title("Track Popularity vs Duration")
plt.xlabel("Duration (min)")
plt.ylabel("Track Popularity")
plt.show()
# Popularity vs explicit songs
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="explicit", y="track_popularity", color="red")
axes = plt.gca()
axes.set_facecolor("lightgray")   # axes ka background
plt.gcf().patch.set_facecolor("lightblue")  # figure ka background
plt.title("Track Popularity vs Explicit Songs")
plt.xlabel("Explicit")
plt.ylabel("Track Popularity")
plt.show()
# Album position vs popularity
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="album_type", y="track_popularity", color="red")
axes = plt.gca()
axes.set_facecolor("lightgray")
plt.gcf().patch.set_facecolor("lightblue")
plt.title("Album Position vs Track Popularity")
plt.xlabel("Album Position")
plt.ylabel("Track Popularity")
plt.show()
# Top 10 popular artists
top_popular_artists = df.groupby("artist_name")["artist_popularity"].mean().nlargest(10).reset_index()
plt.figure(figsize=(12, 6))
axes = sns.barplot(data=top_popular_artists, x="artist_name", y="artist_popularity", palette="viridis")
axes.set_xticklabels(axes.get_xticklabels(), rotation=45)
axes.set_facecolor("lightgray")      
plt.gcf().patch.set_facecolor("lightblue") 
plt.title("Top 10 Popular Artists")
plt.xlabel("Artist Name")
plt.ylabel("Artist Popularity")
plt.tight_layout()
plt.show()
# Realtionship between artist popularity & followers
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="artist_followers", y="artist_popularity", color="red")
axes = plt.gca()
axes.set_facecolor("lightgray")
plt.gcf().patch.set_facecolor("lightblue")
plt.title("Artist Popularity vs Followers")
plt.xlabel("Artist Followers")
plt.ylabel("Artist Popularity")
plt.show()
# Correlation Heatmap
plt.figure(figsize=(10, 6))
correlation_matrix = df[["track_popularity", "track_duration_min", "artist_popularity", "artist_followers"]].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
axes = plt.gca()
axes.set_facecolor("lightgray")
plt.gcf().patch.set_facecolor("lightblue")
plt.title("Correlation Heatmap")
plt.show()
# Top genres by number of tracks
top_genres = df["artist_genres"].value_counts().nlargest(10).reset_index()
top_genres.columns = ["track_genre", "count"]
plt.figure(figsize=(10, 6))
axes = sns.barplot(data=top_genres, x="track_genre", y="count", palette="viridis")
axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
axes.set_facecolor("lightgray")
plt.gcf().patch.set_facecolor("lightblue")
plt.title("Top 10 Genres by Number of Tracks")
plt.xlabel("Track Genre")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
# Average Popularity by Genre
avg_popularity_genre = df.groupby("artist_genres")["track_popularity"].mean().nlargest(10).reset_index()
plt.figure(figsize=(10, 6))
axes = sns.barplot(data=avg_popularity_genre, x="artist_genres", y="track_popularity", palette="viridis")
axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
axes.set_facecolor("lightgray")
plt.gcf().patch.set_facecolor("lightblue")
plt.title("Average Track Popularity by Genre")
plt.xlabel("Artist Genres")
plt.ylabel("Average Track Popularity")
plt.tight_layout()
plt.show()
# Artists with longest and shortest tracks
Longest_tracks = df.nlargest(10, "track_duration_min")[["artist_name", "track_name", "track_duration_min"]]
Shortest_tracks = df.nsmallest(10, "track_duration_min")[["artist_name", "track_name", "track_duration_min"]]
print("Artists with longest tracks:\n", Longest_tracks)
print("Artists with shortest tracks:\n", Shortest_tracks)
# Duration distribution of all songs
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="track_duration_min", bins=30, color="red", kde=True)
axes = plt.gca()
axes.set_facecolor("lightgray")
plt.gcf().patch.set_facecolor("lightblue")
plt.title("Duration Distribution of All Songs")
plt.xlabel("Track Duration (min)")
plt.ylabel("Frequency")
plt.show()
# Albums with the most tracks
album_track_counts = df["album_name"].value_counts().nlargest(10).reset_index()
album_track_counts.columns = ["album_name", "track_count"]
plt.figure(figsize=(10, 6))
axes = sns.barplot(data=album_track_counts, x="album_name", y="track_count", palette="viridis")
axes.set_xticklabels(axes.get_xticklabels(), rotation=45)
axes.set_facecolor("lightgray")
plt.gcf().patch.set_facecolor("lightblue")
plt.title("Top 10 Albums with Most Tracks")
plt.xlabel("Album Name")
plt.ylabel("Track Count")
plt.tight_layout()
plt.show()
# Yearly album release trend
df["album_release year"] = pd.to_datetime(df["album_release_date"])
df["release year"] = df["album_release year"].dt.year
yearly_release_trend = df["release year"].value_counts().sort_index().reset_index()
yearly_release_trend.columns = ["release_year", "album_count"]
plt.figure(figsize=(10, 6))
axes = sns.lineplot(data=yearly_release_trend, x="release_year", y="album_count", markers=True)
axes.set_facecolor("lightgray")
plt.gcf().patch.set_facecolor("lightblue")
plt.title("Yearly Album Release Trend")
plt.xlabel("Release Year")
plt.ylabel("Album Count")
plt.show()
# Popularity trend across albums
album_popularity_trend = df.groupby("album_name")["track_popularity"].mean().nlargest(10).reset_index()
plt.figure(figsize=(10, 6))
axes = sns.barplot(data=album_popularity_trend, x="album_name", y="track_popularity", palette="viridis")
axes.set_xticklabels(axes.get_xticklabels(), rotation=70)
axes.set_facecolor("lightgray")
plt.gcf().patch.set_facecolor("lightblue")
plt.title("Album Popularity Trend")
plt.xlabel("Album Name")
plt.ylabel("Average Track Popularity")
plt.tight_layout()
plt.show()
# Album Type comparison: singles vs albums
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="album_type", y="track_popularity", color="red")
axes = plt.gca()
axes.set_facecolor("lightgray")
plt.gcf().patch.set_facecolor("lightblue")
plt.title("Album Type vs Track Popularity")
plt.xlabel("Album Type")
plt.ylabel("Track Popularity")
plt.tight_layout()
plt.show()
# Artist followers vs track popularity
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="artist_followers", y="track_popularity", size="artist_popularity", hue="artist_popularity", palette="viridis", sizes=(20,200))
axes = plt.gca()
axes.set_facecolor("lightgray")
plt.gcf().patch.set_facecolor("lightblue")
plt.title("Artist Followers vs Track Popularity")
plt.xlabel("Artist Followers")
plt.ylabel("Track Popularity")
plt.show()
# Do artists with more followers have more popular songs?
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x="artist_followers", y="track_popularity", color="red")
axes = plt.gca()
axes.set_facecolor("lightgray")
plt.gcf().patch.set_facecolor("lightblue")
plt.title("Artist Followers vs Track Popularity")
plt.xlabel("Artist Followers")
plt.ylabel("Track Popularity")
plt.show()
# Artist-level summary dashboard
artist_summary = df.groupby("artist_name").agg({
    "track_name": "count",
    "track_popularity": "mean",
    "artist_followers": "mean",
    "artist_popularity": "mean",
}).reset_index().rename(columns={
    "track_name": "total_tracks",
    "track_popularity": "avg_track_popularity",
    "artist_followers": "avg_artist_followers",
    "artist_popularity": "avg_artist_popularity"
})
print("Artist-level Summary Dashboard:\n", artist_summary.head())
# Explicit vs non-explicit popularity
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="explicit", y="track_popularity", color="red")
axes = plt.gca()
axes.set_facecolor("lightgray")
plt.gcf().patch.set_facecolor("lightblue")
plt.title("Explicit vs Non-Explicit Track Populaarity")
plt.xlabel("Explicit")
plt.ylabel("Track Popularity")
plt.show()
# Which genres have more explicit tracks?
explicit_genres = df[df["explicit"] == True]["artist_genres"].value_counts().nlargest(10).reset_index()
explicit_genres.columns = ["track_genre", "explicit_count"]
plt.figure(figsize=(10, 6))
axes = sns.barplot(data=explicit_genres, x="track_genre", y="explicit_count", palette="viridis")
axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
axes.set_facecolor("lightgray")
plt.gcf().patch.set_facecolor("lightblue")
plt.title("Top 10 Genres with Explicit Tracks")
plt.xlabel("Track Genre")
plt.ylabel("Explicit Track Count")
plt.tight_layout()
plt.show()
# Which artists release explicit tracks most often?
explicit_artists = df[df["explicit"] == True]["artist_name"].value_counts().nlargest(10).reset_index()
explicit_artists.columns = ["artist_name", "explicit_count"]
plt.figure(figsize=(10, 6))
axes = sns.barplot(data=explicit_artists, x="artist_name", y="explicit_count", palette="viridis")
axes.set_xticklabels(axes.get_xticklabels(), rotation=45)
axes.set_facecolor("lightgray")
plt.gcf().patch.set_facecolor("lightblue")
plt.title("Top 10 Artists Releasing Explicit Tracks")
plt.xlabel("Artsit Name")
plt.ylabel("Explicit Track Count")
plt.tight_layout()
plt.show()
# Donut chart of explicit vs non-explicit tracks
explicit_artists_count = df["explicit"].value_counts()
labels = ["Non-Explicit", "Explicit"]
sizes = explicit_artists_count.values
colors = ["lightblue", "salmon"]
plt.figure(figsize=(10, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140, wedgeprops={"edgecolor": "black"})
# Draw a circle at the center to make it a donut chart
centre_circle = plt.Circle((0, 0), 0.70, fc="White")
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title("Explicit vs Non-Explicit Tracks")
plt.show()
# Saved cleaned Dataset
df.to_excel("Spotify_Data_Cleaned.xlsx", index=False)
print("Cleaned dataset saved as 'Spotify Data Cleaned.xlsx'")