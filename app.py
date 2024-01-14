import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Importing requried libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import linear_kernel

# Streamlit app header
st.title("Food Recommendation App")

# Sidebar with user input
# user_input = st.sidebar.text_input("Enter some text:")
# Display user input
# st.write("User Input:", user_input)

st.text("Welcome to Group-19 Food recommendation App, Please enter your name to continue: ")
name_input = st.text_input("Your name", key="name")

st.text("What are your dietary preferences?")
# Checkbox for vegetarian diet
vegetarian = st.checkbox("Vegetarian")
# Checkbox for non-vegetarian diet
non_vegetarian = st.checkbox("Non Vegetarian")
# Checkbox for eggetarian diet
eggetarian = st.checkbox("Eggetarian")

# Add selected diet preferences
selected_diets = []
if vegetarian:
    selected_diets.append("Vegetarian")
if non_vegetarian:
    selected_diets.append("Non Vegetarian")
if eggetarian:
    selected_diets.append("Eggetarian")
    
imp_options1 = ['1', '2', '3', '4', '5']
st.text("Rate importance of this selection:")
diet_imp = st.selectbox("Rating 1", imp_options1)
    
st.text("Please select if you have any dietary restrictions")
# Options for diet restrictions
diet_restrictions_options = ['Diabetic Friendly', 'High Protein', 'No Onion No Garlic', 'Vegan', 'Gluten Free']
# Multi-select dropdown for diet restrictions
selected_restrictions = st.multiselect("Select Diet Restrictions", diet_restrictions_options)

imp_options2 = ['1', '2', '3', '4', '5']
st.text("Rate importance of this selection:")
rest_imp = st.selectbox("Rating 2", imp_options2)

st.text("What is your preferred course for preparation")
# Options for course
course_options = ['Breakfast', 'Lunch', 'Dinner', 'Main Course', 'Snack', 'Appetizer']
# Multi-select dropdown for course
selected_course = st.multiselect("Select Course Preference", course_options)

st.text("Rate importance of this selection:")
imp_options3 = ['1', '2', '3', '4', '5']
course_imp = st.selectbox("Rating 3", imp_options3)

st.text("What is your preferred cuisine for this preparation")
# Options for course
cuisine_options = ['', 'Indian', 'Continental', 'Italian', 'Local']
# Multi-select dropdown for course
selected_cuisine = st.selectbox("Select Cuisine Preference", cuisine_options)

st.text("Rate importance of this selection:")
imp_options4 = ['1', '2', '3', '4', '5']
cuisine_imp = st.selectbox("Rating 4", imp_options4)

st.text("What are the list of ingredients available with you")
# Multi-text input for ingredients
ingredients = st.text_area("Enter ingredients (upto 5, separate with commas or new lines):", key="ingredients")

st.text("Rate importance of this selection:")
imp_options5 = ['1', '2', '3', '4', '5']
ingredients_imp = st.selectbox("Rating 5", imp_options5)

st.text("What is your preferred time of preparation for this dish?")
# Slider input for preparation time in minutes
preparation_time = st.slider("Select Preparation Time (minutes):", min_value=0, max_value=120, value=0, step=5)

st.text("Rate importance of this selection:")
imp_options6 = ['1', '2', '3', '4', '5']
prep_imp = st.selectbox("Rating 6", imp_options6)



# Display selected diet preferences
if name_input:
    st.markdown("**Hi,** ")
    st.write(name_input)

# Display selected diet
if selected_diets:
    st.markdown("**Your Selected Diet Preferences:**")
    st.write(", ".join(selected_diets))
    st.write("Diet Preference with importance:", diet_imp)

# Display selected diet restrictions
if selected_restrictions:
    st.markdown("**Your Selected Diet Restrictions:**")
    st.write(", ".join(selected_restrictions))
    st.write("Diet Restrictions with importance:", rest_imp)

# Display selected course preference
if selected_course:
    st.markdown("**Your Course Preference:**")
    st.write(", ".join(selected_course))
    st.write("Course with importance:", course_imp)

# Display selected cuisine preference
if selected_cuisine:
    st.markdown("**Your Cusine for preparation:**")
    st.write(selected_cuisine)
    st.write("Cuisine with importance:", cuisine_imp)

# Display the entered ingredients
if ingredients:
    ingredients_list = [ingredient.strip() for ingredient in ingredients.split(',')]
    st.markdown("**You entered the following ingredients:**")
    st.write('\n'.join(ingredients_list))
    st.write("Ingredients with importance:", ingredients_imp)

# Display the selected preparation time
if preparation_time > 0:
    st.markdown("**Selected Preparation Time is:**")
    st.write(preparation_time, "minutes")
    st.write("Preparation time with importance:", prep_imp)
  

# Function that takes user input as an argument
def process_user_input(user_input):
    # Perform some operation with the selected options
    result = user_input
    return result

#function which performs food recommendation
def food_recommendation(user_input):
    
    #read cleaned up model recipe data
    # model_recipe_data = pd.read_csv("D:/Study/Python/FP1_Project/recipe_data_clean.csv")
    model_recipe_data = pd.read_csv("recipe_data_clean.csv")
    
    # Create a score for each dish
    scores = np.zeros(len(model_recipe_data))
    
    #diet_pref calculation - veg, non-veg. egg
    scores[model_recipe_data["Diet_Pref"].isin(user_input['diet_pref'])] += float(diet_imp)
    
    #diet_restriction calculation if not none - diabetic, high protien etc
    if user_input['diet_restriction']:
        scores[model_recipe_data["Diet_Restrict"].isin(user_input['diet_restriction'])] += float(rest_imp)
        
    #cuisine_type calculation
    scores[model_recipe_data["Cuisine"].str.contains(user_input['cuisine_type'])] += float(cuisine_imp)
    
    #course_type calculation
    scores[model_recipe_data["Course_Type"].isin(user_input['course_type'])] += float(course_imp)
    
    # Use TF-IDF to vectorize the ingredients
    tfidf = TfidfVectorizer(stop_words='english')
    model_recipe_data['Ingredients_list'] = model_recipe_data['Ingredients_list'].fillna('')
    tfidf_matrix = tfidf.fit_transform(model_recipe_data['Ingredients_list'])
    
    # Vectorize available ingredients
    for ingredient in user_input['available_ingredients']:
        ingredient = ingredient.lower()
        user_pref_vector = tfidf.transform([ingredient])

        # Calculate cosine pairwise similarity and add to scores
        cosine_similarities = linear_kernel(user_pref_vector, tfidf_matrix).flatten()

        # Introduce a threshold for cosine similarity (for example, 0.2)
        threshold = 0.2
        cosine_similarities[cosine_similarities < threshold] = 0
        
        scores += cosine_similarities
        
    # Get top 5 highest scoring dishes
    top_5_indices = scores.argsort()[:-6:-1]
    
    # Get corresponding cosine similarity scores
    top_5_scores = scores[top_5_indices]

    # Return names of the dishes and their scores
    return model_recipe_data.iloc[top_5_indices]['Recipe Name'], top_5_scores

# Button to trigger the function
if st.button("Recommend Food"):
    
    user_input = {
    'diet_pref': selected_diets,
    'diet_restriction': selected_restrictions,
    'cuisine_type': selected_cuisine,
    'course_type': selected_course,
    'available_ingredients': ingredients_list,
    'prep_time': preparation_time,
    'diet_imp': diet_imp,
    'rest_imp': rest_imp,
    'cuisine_imp': cuisine_imp,
    'course_imp': course_imp,
    'ingredients_imp': ingredients_imp,
    'prep_imp': prep_imp
    }
    
    # st.write(type(user_input['diet_pref']))
    # st.write(user_input['diet_restriction'])
    # st.write(user_input['cuisine_type'])
    # st.write(type(user_input['cuisine_type']))
    # st.write(user_input['course_type'])
    
    # Call the function with the selected options
    # output = process_user_input(user_input)
    
    recommendations, scores = food_recommendation(user_input)
    
    food_list = recommendations.tolist()

    # Display the output
    # st.write(type(food_list))
    # st.write(food_list)
    # st.write(scores)
    
    st.write("\nTop five food recommendations for you:")
    for i in range(len(food_list)):
        st.write(f"{i+1}. {food_list[i]} with a score of {scores[i]}")

    

    

