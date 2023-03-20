# By David Maimoun
# deployed the 13.03.23
import streamlit as st
import pandas as  pd
import json
from streamlit_lottie import st_lottie
import math
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tmdbv3api import Movie
from tmdbv3api import TMDb


st.write("""
<style>
   :root {
      --main-color: rgb(239, 179, 80);
      --grey: #bebebe;
   }

   .main_title {
      color: var(--main-color);
      letter-spacing: 10px;

   }

   h1 {
      font-size: 55px;
      text-shadow: 2px 2px 2px rgb(224, 145, 95);
      margin-bottom: -8px;
   }

   h3 {
      font-weight: 300;

   }

   img *{
      background-color: lightgrey;
   }
   
   .subtitle {
      color: grey;
      font-size: 1.1rem;
      margin-bottom: 32px;

   }
  
   .title {
      font-size: 1.3rem;
      color: var(--main-color);
      lettre-spacing: 12px;

   }
  
   .description {
      margin-top: 8px;
      margin-bottom: 24px;
      font-size: .85rem;
   }
   
   .genres {
      color: gray;
      margin-right: 8px;
   }
  
   .time {
      color: red;
   }
   .crew {
      font-style: italic;
      color: gray;
   }
   .stars {
      color: #ffdf00;
   }
   img {
      border-radius: 5px;
      width:100%;
      object-fit: cover;
      margin-bottom: 40px;

   }

   .filter_result {
      font-style: italic;
      color: var(--grey);
      font-weight: 200;
   }

   .cast_title {
      text-transform: uppercase;
      text-decoration: none;
      background-color: var(--main-color) !important;
      color: white !important;
      padding: 4px 8px;
      border-radius: 5px;
   }
   .cast_title:hover {
      text-decoration: none;
      background-color: white !important;
      color: var(--main-color) !important;
      border: 1px solid var(--main-color) ;
   }
   .cast_films {
      font-size: .8rem;
      margin-left: 8px;
      line-height: .9rem;
   }

   div[data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
      box-shadow: 0 0 12px rgb(240,240,240);
      border-radius: 22px;
      padding: 18px;
   }


</style>
""", unsafe_allow_html=True)

API_KEY='15d2ea6d0dc1d476efbca3eba2b9bbfb&que'
MOVIES_DB_URL = 'http://image.tmdb.org/t/p/w500'
URL_WIKI = 'https://en.wikipedia.org/wiki/'
TITLE = 'title'
DIRECTOR = 'director'
GENRES = 'genres'
SYNOPSIS = 'overview'
CAST = 'cast'
RUNTIME = 'runtime'
RELEASE_DATE = 'release_date'
FILTER_ONE = 'Suggestion based on movie genre'
FILTER_TWO = 'Filter your search'
tmdb = TMDb()
tmdb.api_key = API_KEY
movie_search = Movie()

genres_selection = ['Action', 'Adventure', 'Animation',
                 'Biography', 'Comedy', 'Crime', 'Drama' , 
                 'Family', 'Fantasy', 'Foreign', 'History', 'Horror', 'Mystery',
                  'Romance','Science-Fiction', 'Thriller', 'TV Movie', 
                  'War','Western']

def load_lottiefile(filepath: str):
   with open(filepath, "r") as f:
      return json.load(f)

def returnLottie(path):
   return st_lottie(
      load_lottiefile(path),
      speed=1,
      reverse=False,
      loop=False,
      quality="medium", # medium ; high
      height=180,
      width=180,
      key=None,
   )

def getRatingScore(rating):
   return math.ceil(rating/2)

def populateRatingStars(getRatingScore, data):
   rating = getRatingScore(data)
   rating_stars = "<span class=stars>"
   for i in range(0, rating):
      rating_stars +='★'
   for i in range(0, (5-rating)):
         rating_stars +='☆'
   rating_stars += "</span>"
   return rating_stars

def populateTitle(index, title, release_date, genres, time):
   return st.markdown(f"""
            <div class=title>
               {index}. {title} ({release_date.split('-')[0]})
            </div>
            <div class='description'>
               📽️ <span class='genres'>{genres}</span> 
               ⏳️ <span class=time>{time} min</span>
            </div>
            """, unsafe_allow_html=True)

def populateOverview(overview): 
   return st.markdown(f"""
            <div class="container">
               <b>Synopsis</b><div class=SYNOPSIS>{overview} </div>
            </div>
            """, unsafe_allow_html=True)

def populateCrew(director, cast, rating):
   st.markdown(f"""
            <div class="container">
               <b>Director</b> <div class='crew director'>{director}</div>
               <b>Cast</b> <div class='crew cast'>{cast}</div>
               <b>Rating:</b> {rating}

            </div>
            """, unsafe_allow_html=True)

def populateImage(search):
   return st.markdown(f"""
         <img src='{MOVIES_DB_URL}/{search[0].poster_path}' alt='movie'>
         """,unsafe_allow_html=True)

def populateImageDefault(url):
   return st.image(url)

def getDfSimilarities(genres):
   df = pd.DataFrame(index=df_movies['title'])
   count_list = []
   for g in genres:
      for index, row in df_movies.iterrows():
         count_list.append(1 if g in row[GENRES] else 0)
      df[g] = count_list
      count_list = []
   print(df.columns)
   return df

df_movies = pd.read_csv('data/movies.csv')
df_movies[DIRECTOR] = df_movies[DIRECTOR].fillna('')
df_movies[GENRES] = df_movies[GENRES].fillna('')
df_movies[CAST] = df_movies[CAST].fillna(' ')
df_movies['release_year'] = df_movies[RELEASE_DATE].apply(lambda x: str(x).split('-')[0])
df_movies['release_year'] = df_movies['release_year'].apply(lambda x: int(x))
# Remove unicode
df_movies[GENRES] = df_movies[GENRES].apply(lambda x: str(x).replace(u'\xa0', u' '))

list_of_titles = df_movies[TITLE].tolist()

# features = [GENRES, CAST, 'tagline', 'keywords', DIRECTOR, 'original_title']
features = [GENRES]

# Replace null values by null string
for feature in features:
   df_movies[feature] = df_movies[feature].fillna('')


col1, col2 = st.columns([.5,1], gap='small')
with col1:
   returnLottie('assets/movie.json')
with col2:
   st.markdown("""
      <h1 class='main_title'>CLAPP!</h1>
      <h3 class='main_title'>THE MOVIE FINDER</h3>
      <p class=subtitle>Because you deserve a nice evening !</p>
   """, unsafe_allow_html=True)

container = st.container()

search_filter = container.radio(
    "Search choice: ",
    (FILTER_ONE, FILTER_TWO), horizontal=True)

st.markdown("<br>", unsafe_allow_html=True)

movies_match = []
FILTER_ONE = 'Suggestion based on movie genre'
radio_one = FILTER_ONE
radio_two = FILTER_TWO
is_match = True
input = ''

if search_filter == radio_one:
   # Get the movie title input
   col1, col2 = container.columns([1, 1], gap="medium")
   with col1:
      title_taped = st.text_input("1- Enter a title",
         help="""
         If empty, the search will be done based on field 2.\n
         If the both field filled, the field 1 will be take in priority.""").strip() 
      with st.expander("2- Otherwise select from the list"):
         title_selected = st.selectbox('Select',pd.unique(df_movies[TITLE])) 
   with col2:
      results_to_display = st.slider("Number of Results:", 1, 500, 30)

   # Get the exact match
   if len(title_taped) > 0:
      movies_match = df_movies.loc[df_movies[TITLE].str.contains(title_taped, case=False)]
      movies_match = movies_match.reindex(movies_match[TITLE].str.len().sort_values().index)
      input = title_taped
   else: 
      movies_match = df_movies.loc[df_movies[TITLE].str.contains(title_selected, case=False)]
      input = title_selected
   
   if len(movies_match) == 0:
      is_match = False
   
   if is_match == True:
         genres_list = movies_match[GENRES].iloc[0]

         if not genres_list:
            genres = pd.unique(genres_list.split(' '))
         else:
            genres = pd.unique(genres_list)
   else:
      st.warning('Please enter a valid title', icon="⚠️")

    
if search_filter == radio_two:
   col1, col2, col3 = container.columns([1,1,1], gap="medium")
   with col1:
      director = st.selectbox('Director Name', sorted(pd.unique(df_movies[DIRECTOR]))) 
   with col2:
      genres = st.selectbox('Movie genres wanted', genres_selection)
   with col3:  
      min_year = min(df_movies['release_year'])
      max_year = max(df_movies['release_year'])
      released = st.slider('Released Date', min_year, max_year, [min_year, max_year])
  
   movies_match = df_movies.copy()
   if len(director) > 0 :
      movies_match = movies_match.loc[df_movies[DIRECTOR].str.contains(director, case=False)]
      input += f"{director}, "
   if len(genres) > 0 :
      movies_match = movies_match.loc[df_movies[GENRES].str.contains(genres, case=False)]
      input += f"genres: {genres}, "
   if len(released) > 0 :
      movies_match = movies_match.loc[df_movies['release_year'].between(*released)]
      input += f"released in: {released[0]} - {released[1]}"


if container.button('Search Movies !'):
   with st.spinner(f'Fetching the data for {input}...'):

      id = 1
      if len(movies_match) > 0:
         st.markdown(f"""
               <hr>
               <h5>Found {len(movies_match)} results for <span class='filter_result'>{input}</span></h5>
               """, unsafe_allow_html=True
         )
         genres_wanted = ''
         for index, row in movies_match.iterrows():
            rating_stars = populateRatingStars(getRatingScore, row['vote_average'])
            search = movie_search.search(row[TITLE])
            genres_wanted += row[GENRES] + ' '

            with st.container():
               populateTitle(id, row[TITLE], row[RELEASE_DATE], row[GENRES], row[RUNTIME])

               id += 1
               col1, col2, col3 = st.columns([1,2,1],gap="medium")
               
               with col1:
                  if len(search) > 0:
                     populateImage(search)
                  else:
                     populateImageDefault('assets/default.jpg')
               with col2:
                  populateOverview(row[SYNOPSIS])
               with col3:
                  populateCrew(row[DIRECTOR], row[CAST], rating_stars)

     
         if (search_filter == radio_one) & (is_match):
            df_similarities = getDfSimilarities(pd.unique(genres_wanted.split()))
            df_similarities['total'] = df_similarities.sum(axis=1)
            df_similarities.sort_values(by=['total'], ascending=False, inplace=True)

            i = 0
            # Reset index to pass title from index to column
            df_similarities = df_similarities.reset_index(TITLE)
            similar_movies = pd.DataFrame()

            for index, row in df_similarities.iterrows():
               if i < (results_to_display-len(movies_match)):
                  # I want to remove the exact match from the result to not get
                  # them twice (in the exact result, and in the recommended)
                  if row[TITLE] not in movies_match[TITLE].values:
                     similar_movies = pd.concat([similar_movies, df_movies[df_movies[TITLE] == row[TITLE]]])
                     i += 1
               else:
                  break
            
            ######################################################
            # Get the cast filmogragraphy for the first match
            with st.sidebar:
               st.subheader('🎬 With the same actors ')

               cast = movies_match[CAST].iloc[0]
               if '.' in cast:
                  cast = cast.replace('Jr.','').replace('Sr.', '')
               cast = cast.split()
               cast_exact_match = pd.DataFrame()
               for c in range(0, (len(cast))): 
                  if (c%2 != 0):
                     name = cast[c-1] + ' ' + cast[c]
                     name_wiki = cast[c-1] + '_' + cast[c]
                     
                     cast_exact_match = pd.concat(
                        [cast_exact_match, df_movies.loc[df_movies[CAST].str.contains(name)]])
                     st.markdown(f"""
                        <hr>
                        <h4><a class='cast_title' href={URL_WIKI}{name_wiki}>{name}</a></h4>
                     """, unsafe_allow_html=True)
                     for index, row in cast_exact_match.iterrows():
                        st.markdown(f"<p class=cast_films>{row[TITLE]}<p>", unsafe_allow_html=True)
                     
                     cast_exact_match = pd.DataFrame()
            
            st.markdown(f"""
                        <hr>
                        <h5>You also might like...</h5>
                        """, unsafe_allow_html=True
            ) 

            for index, row in similar_movies.iterrows():
               rating_stars = populateRatingStars(getRatingScore, row['vote_average'])
               search = movie_search.search(row[TITLE])
            
               with st.container():
                  populateTitle(id, row[TITLE], 
                              row[RELEASE_DATE], 
                              row[GENRES],
                              row[RUNTIME])

                  id += 1
                  col1, col2, col3 = st.columns([1,2,1])
                  with col1:
                     if len(search) > 0:
                        populateImage(search)
                     else:
                        populateImageDefault('assets/default.jpg')
                  with col2:
                     populateOverview(row[SYNOPSIS])
                  with col3:
                     populateCrew(row[DIRECTOR], row[CAST], rating_stars)

                  
st.markdown('<br><br><p><i>By David Maimoun</p></i>',unsafe_allow_html=True)  
      

     
