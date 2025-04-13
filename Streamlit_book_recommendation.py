import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
from streamlit_lottie import st_lottie
from PIL import Image
import requests
import re



def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


animation = load_lottieurl("https://gist.githubusercontent.com/Chandhini06/cf1d55ddd7acb9913f18495a2536c915/raw/38f43aa3d0c1196da9e48a9bfc4f8b631345207b/animation.json")

df = pd.read_csv("C:/Users/Admin/OneDrive/Documents/Book Recommendation/clustered_data.csv")
load_cosine_sim = pickle.load(open("C:/Users/Admin/OneDrive/Documents/Book Recommendation/cosine_similarity_matrix.pkl", "rb"))
load_kmeans = pickle.load(open("C:/Users/Admin/OneDrive/Documents/Book Recommendation/kmeans_clustering_model.pkl", "rb"))


st.set_page_config(page_title = "Book Recommendation System")


# Custom CSS for background and text styling
st.markdown("""
    <style>
        .main {
            background-color: #f4f4f9;
            color: #333333;
        }
        .css-18e3th9 {
            background-color: #2E3B4E;
        }
        .sidebar .sidebar-content {
            background-color: #2E3B4E;
            color: white;
        }
        .css-1v0mbd3 {
            background-color: #2E3B4E;
            color: white;
        }
        .css-2trqyj {
            background-color: #2E3B4E;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)


def recommend_books_by_content_in_genre(df, cosine_sim, genre, num_recs=5):
    genre_df = df[df['Genre'] == genre]
    if genre_df.empty:
        return []

    genre_idx = genre_df.index.to_list()
    cosine_sim_genre = cosine_sim[genre_idx][:, genre_idx]

    similar_books = list(enumerate(cosine_sim_genre[0]))
    sorted_books = sorted(similar_books, key=lambda x: x[1], reverse=True)[1:num_recs+1]
    recommendations = [
        (genre_df.iloc[i[0]]['Book Name'], genre_df.iloc[i[0]]['Rating'], genre_df.iloc[i[0]]['Number of Reviews'], genre_df.iloc[i[0]]['cleaned_description'], genre_df.iloc[i[0]]['Price'])
        for i in sorted_books
    ]
    return recommendations

# Clustering-based recommendations for books within the selected genre
def recommend_books_by_cluster_in_genre(df, genre, kmeans_model, num_recs=5):
    genre_df = df[df['Genre'] == genre]
    if genre_df.empty:
        return []

    book_cluster = genre_df['cluster'].mode()[0]
    cluster_books = genre_df[genre_df['cluster'] == book_cluster][['Book Name', 'Rating', 'Number of Reviews', 'cleaned_description', 'Price']].head(num_recs).values.tolist()
    return cluster_books



# Truncate long descriptions
def truncate_description(description, max_length=200):
    return description[:max_length] + '...' if len(description) > max_length else description



def rec_system_page():

    # st.title(" 📚Book Recommendation System")
    genre = st.selectbox("Choose a Genre", df['Genre'].unique())
    num_recs = st.slider("Number of Recommendations", 1, 10, 5)
    st.header(f"📚 Content-Based Recommendations for {genre}")

    recs = recommend_books_by_content_in_genre(df, load_cosine_sim, genre, num_recs)
    if recs:
        for rec in recs:
            st.markdown(f"**Book Name:** {rec[0]}\n\n"
                        f"**Rating:** {rec[1]}\n\n"
                        f"**Price:** {rec[4]}\n\n"
                        "---")
    else:
        st.warning(f"No recommendations found for {genre}.")

def cluster_rec():

    genre = st.selectbox("Choose a Genre", df['Genre'].unique())
    num_recs = st.slider("Number of Recommendations", 1, 10, 5)
    st.header(f"📚Clustering-Based Recommendations for {genre}")


    recs = recommend_books_by_cluster_in_genre(df, genre, load_kmeans, num_recs)
    if recs:
        for rec in recs:
            st.markdown(f"**Book Name:** {rec[0]}\n\n"
                        f"**Rating:** {rec[1]}\n\n"
                        f"**Price:** {rec[4]}\n\n"
                        "---")
    else:
        st.warning(f"No recommendations found for {genre}.")


st.sidebar.title("Navigate through the sections")
choice = st.sidebar.radio(label = "Select a section", options=["Home", "Content based recommendation", "Cluster based recommendation", "EDA"])


if choice == "Home" :

    st_lottie(animation, height=250, width = 500, key="title_anim")

    st.title("📚 Welcome to the Book Recommendation System!")
    st.markdown("""
        ### 📖 *Discover your next favorite read with smart recommendations.*
        Welcome to the **Book Recommendation System**, where you can explore and discover new books tailored to your interests. 
        Whether you're a fan of fiction, non-fiction, or self-improvement, our system helps you find the perfect book for every mood and need.
    """)

elif choice == "Content based recommendation" :

    rec_system_page()

elif choice == "Cluster based recommendation":

    cluster_rec()    
    
elif choice == "EDA":

   
    st.title("Exploratory Data Analysis (EDA) - FAQs")

    faq_questions = {
        "What are the most popular genres in the dataset?": "📚 Most Popular Genres",
        "Which authors have the highest-rated books?": "👨‍💻 Highest-Rated Authors",
        "What is the average rating distribution across books?": "📈 Average Rating Distribution",
        "How do ratings vary between books with different review counts?": "⭐ Ratings vs. Review Counts",
        "Which books are frequently clustered together based on descriptions?": "📚 Frequently Clustered Books",
        "How does genre similarity affect book recommendations?": "📊 Genre Similarity and Recommendations",
        "What is the effect of author popularity on book ratings?": "👨‍💻 Effect of Author Popularity on Book Ratings",
        "Which combination of features provides the most accurate recommendations?": "📊 Feature Combinations and Recommendations",
        "Identify books that are highly rated but have low popularity to recommend hidden gems.": "💎 Hidden Gems"
    }

    for question, header in faq_questions.items():
        with st.expander(question):
            if header == "📚 Most Popular Genres":
                genre_count = df['Genre'].value_counts().head(10)
                plt.figure(figsize=(12, 6))
                sns.barplot(x=genre_count.values, y=genre_count.index, palette="magma")
                plt.xlabel("Number of Books")
                plt.ylabel("Genre")
                plt.title("Top 10 Book Genres")
                plt.show()
                fig = plt.gcf()
                st.pyplot(fig)

            elif header == "👨‍💻 Highest-Rated Authors":
                top_authors = df.groupby('Author').agg({'Rating': 'mean', 'Book Name': 'count'}).reset_index()
                top_authors = top_authors[top_authors['Book Name'] > 1].sort_values('Rating', ascending=False).head(10)
                fig = px.bar(top_authors, x='Author', y='Rating', color='Rating', title="Top 10 Authors by Average Rating",
                             labels={'Rating': 'Average Rating'}, color_continuous_scale='Viridis')
                st.plotly_chart(fig)

            elif header == "📈 Average Rating Distribution":
                fig, ax = plt.subplots()
                sns.histplot(df['Rating'], kde=True, ax=ax)
                ax.set_title('Distribution of Ratings')
                st.pyplot(fig)

            elif header == "⭐ Ratings vs. Review Counts":
                fig, ax = plt.subplots()
                sns.scatterplot(data=df, x='Number of Reviews', y='Rating', ax=ax)
                ax.set_title('Ratings vs. Review Counts')
                st.pyplot(fig)

            elif header == "📚 Frequently Clustered Books":
                
                top_clusters = df['cluster'].value_counts().head(5).index

                st.write("Top 5 Clusters by Book Count:")
                st.dataframe(df['cluster'].value_counts().head(5))

                # Display books in each of the top clusters
                for cluster_num in top_clusters:
                    st.write(f"\n📚 **Books in Cluster {cluster_num}:**")
                    st.dataframe(df[df["cluster"] == cluster_num][["Book Name", "Author"]].head(3))

            elif header == "📊 Genre Similarity and Recommendations":
                st.markdown("""
                Genre similarity plays a crucial role in content-based recommendations by grouping books with similar themes, topics, and styles. Books from the same genre tend to have higher cosine similarity scores, leading to stronger recommendations within the genre.
                """)

            elif header == "👨‍💻 Effect of Author Popularity on Book Ratings":
                author_popularity = df.groupby('Author').agg({'Rating': 'mean', 'Number of Reviews': 'sum'}).reset_index()
                fig = px.scatter(author_popularity, x='Number of Reviews', y='Rating', hover_data=['Author'],
                                 title='Author Popularity vs. Ratings')
                st.plotly_chart(fig)

            elif header == "📊 Feature Combinations and Recommendations":
                st.markdown("""
                Feature combinations such as Genre, Ratings, Review Counts, and Author Popularity can help fine-tune recommendations. By combining these features, we can better understand user preferences and optimize recommendations.
                """)

            elif header == "💎 Hidden Gems":
                hidden_gems = df[(df['Rating'] >= 4.5) & (df['Number of Reviews'] < 100)]
                st.dataframe(hidden_gems[['Book Name', 'Rating', 'Number of Reviews']])


