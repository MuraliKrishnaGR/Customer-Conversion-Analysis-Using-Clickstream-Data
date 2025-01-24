import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow.pyfunc

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load models (cached for performance)
@st.cache_resource
def load_models():
    clustering_model = mlflow.pyfunc.load_model("models:/KMeans Clustering/9")
    classification_model = mlflow.pyfunc.load_model("models:/Random Forest Classifier/10")
    regression_model = mlflow.pyfunc.load_model("models:/Random Forest Regressor/10")
    return clustering_model, classification_model, regression_model

clustering_model, classification_model, regression_model = load_models()

# Sidebar for task selection
st.sidebar.title("Choose Task")
task = st.sidebar.selectbox(
    "Select a Machine Learning Task:",
    ["Clustering", "Regression", "Classification"]
)

# Task: Clustering
if task == "Clustering":
    st.title("Clustering - Recommendations Based on Session Data")
    st.markdown("""
    This section uses a KMeans clustering model to recommend groups based on session data.
    """)

    # Input fields for clustering
    session_length = st.number_input("Session Length", value=4.0)
    page1_main_category_sale = st.checkbox("Page1 Main Category: Sale", value=False)
    page1_main_category_skirts = st.checkbox("Page1 Main Category: Skirts", value=False)
    page1_main_category_trousers = st.checkbox("Page1 Main Category: Trousers", value=False)
    continent_north_america = st.checkbox("Continent: North America", value=False)
    continent_oceania = st.checkbox("Continent: Oceania", value=False)
    continent_europe = st.checkbox("Continent: Europe", value=True)

    # Prepare input DataFrame for clustering
    clustering_input = pd.DataFrame({
        'session_length': [session_length],
        'page1_main_category_sale': [page1_main_category_sale],
        'page1_main_category_skirts': [page1_main_category_skirts],
        'page1_main_category_trousers': [page1_main_category_trousers],
        'continent_North America': [continent_north_america],
        'continent_Oceania': [continent_oceania],
        'continent_Europe': [continent_europe]
    })
    clustering_input = clustering_input.astype(int)
    
    # Predict cluster
    if st.button("Predict Cluster"):
        cluster_label = clustering_model.predict(clustering_input)[0]
        st.success(f"Predicted Cluster: {cluster_label}")
        
        # Recommendations based on predicted cluster
        st.subheader("Cluster-Based Recommendations")

        if cluster_label == 0:
            st.markdown("""
            **Cluster 0: High-Engagement European Shoppers**
            - Long session lengths with no specific category preference.
            - Promote curated collections or trending items to capture interest.
            - Highlight exclusive deals to keep them engaged.
            """)
        elif cluster_label == 1:
            st.markdown("""
            **Cluster 1: Quick Browsers in Europe**
            - Short sessions with no specific focus.
            - Simplify navigation and feature popular products prominently.
            - Use personalized recommendations to encourage deeper exploration.
            """)
        elif cluster_label == 2:
            st.markdown("""
            **Cluster 2: Focused Trousers Shoppers**
            - Preference for trousers and short session lengths.
            - Showcase more options within the trousers category.
            - Highlight complementary products, such as matching tops or accessories.
            """)
        elif cluster_label == 3:
            st.markdown("""
            **Cluster 3: Multi-Category North American Users**
            - Moderate engagement with multiple categories.
            - Suggest bundles or "shop the look" collections.
            - Use targeted ads for trending products in their region.
            """)
        elif cluster_label == 4:
            st.markdown("""
            **Cluster 4: Skirts Lovers in Europe**
            - Preference for skirts and shorter session lengths.
            - Promote seasonal skirt collections.
            - Highlight complementary items, such as shoes or tops.
            """)
        elif cluster_label == 5:
            st.markdown("""
            **Cluster 5: Oceania Shoppers with Mixed Preferences**
            - Short sessions and engagement across sale and skirts.
            - Focus on discounts and promotions for skirts.
            - Use email campaigns to re-engage them with sales alerts.
            """)
        elif cluster_label == 6:
            st.markdown("""
            **Cluster 6: Sale-Focused European Shoppers**
            - Long session lengths and strong preference for sale items.
            - Emphasize ongoing sales and exclusive offers.
            - Create urgency with "limited time deals."
            """)
        elif cluster_label == 7:
            st.markdown("""
            **Cluster 7: Trousers Enthusiasts with High Engagement**
            - Long sessions focused on trousers.
            - Offer premium trousers collections and customization options.
            - Provide loyalty rewards or incentives for frequent buyers.
            """)
        elif cluster_label == 8:
            st.markdown("""
            **Cluster 8: High-Engagement Skirts Shoppers**
            - Long sessions with exclusive focus on skirts.
            - Promote premium skirt collections or limited-edition designs.
            - Highlight related products to build complete outfits.
            """)
        elif cluster_label == 9:
            st.markdown("""
            **Cluster 9: High-Engagement Generalists in Europe**
            - Very long sessions with no specific category focus.
            - Offer curated collections, such as "Top Picks for You."
            - Focus on cross-category bundling to maximize value.
            """)
        else:
            st.markdown("""
            **Unidentified Cluster**
            - No specific behavior identified. Continue collecting data for further insights.
            """)
            # Visualizations
        st.subheader("Visualizations")


        # Visualization: Bar chart of input features
        st.subheader("Input Features Visualization")
        fig, ax = plt.subplots(figsize=(8, 4))
        clustering_input.iloc[0].plot(kind="bar", color="skyblue", ax=ax)
        plt.title(f"Input Features for Cluster {cluster_label}")
        plt.xlabel("Features")
        plt.ylabel("Values")
        st.pyplot(fig)


# Task: Regression
elif task == "Regression":
    st.title("Regression - Price Prediction")
    st.markdown("""
    This section uses a regression model to predict the price of a clothing item.
    colours: 1-beige 2-black 3-blue 4-brown 5-burgundy 6-gray 7-green 8-navy blue 9-of many colors 
    10-olive 11-pink 12-red 13-violet 14-white
    """)

    # Input fields for regression
    clothing_model_number = st.number_input("Clothing Model Number", value=23)
    page1_main_category_sale = st.checkbox("Page1 Main Category: Sale", value=False)
    page1_main_category_skirts = st.checkbox("Page1 Main Category: Skirts", value=False)
    page1_main_category_trousers = st.checkbox("Page1 Main Category: Trousers", value=False)
    colour = st.number_input("Colour", value =2 )
    continent_europe = st.checkbox("Continent: Europe", value=True)
    continent_north_america = st.checkbox("Continent: North America", value=False)
    continent_oceania = st.checkbox("Continent: Oceania", value=False)

    # Prepare input DataFrame for regression
    regression_input = pd.DataFrame({
        'clothing_model_number': [clothing_model_number],
        'page1_main_category_sale': [page1_main_category_sale],
        'page1_main_category_skirts': [page1_main_category_skirts],
        'page1_main_category_trousers': [page1_main_category_trousers],
        'colour':[colour],
        'continent_Europe': [continent_europe],
        'continent_North America': [continent_north_america],
        'continent_Oceania': [continent_oceania]
    })

    regression_input = regression_input.astype(int)
    # Predict price
    if st.button("Predict Price"):
        predicted_price = regression_model.predict(regression_input)[0]
        st.success(f"Predicted Price: ${predicted_price:.2f}")

        # Visualization: Bar chart of input features
        st.subheader("Input Features Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        regression_input.iloc[0].plot(kind="bar", color="skyblue", ax=ax)
        plt.title("Input Features for Regression")
        plt.xlabel("Features")
        plt.ylabel("Values")
        st.pyplot(fig)

# Task: Classification
elif task == "Classification":
    st.title("Classification - Price Category Prediction")
    st.markdown("""
    This section uses a classification model to predict the price category.
    """)

    # Input fields for manual classification prediction
    continent_europe = st.checkbox("Continent: Europe", value=True)
    continent_north_america = st.checkbox("Continent: North America", value=False)
    continent_oceania = st.checkbox("Continent: Oceania", value=False)
    session_length = st.number_input("Session Length", value=4.0)
    clothing_model_number = st.number_input("Clothing Model Number", value=23)
    page1_main_category_sale = st.checkbox("Page1 Main Category: Sale", value=False)
    page1_main_category_skirts = st.checkbox("Page1 Main Category: Skirts", value=False)
    page1_main_category_trousers = st.checkbox("Page1 Main Category: Trousers", value=False)
    clicks_sale = st.number_input("Clicks on Sale", value=2.0)
    clicks_skirts = st.number_input("Clicks on Skirts", value=0.0)
    clicks_trousers = st.number_input("Clicks on Trousers", value=0.0)

    
    # Prepare input DataFrame for classification
    classification_input = pd.DataFrame({
        'continent_Europe': [continent_europe],
        'continent_North America': [continent_north_america],
        'continent_Oceania': [continent_oceania],
        'session_length': [session_length],
        'clothing_model_number': [clothing_model_number],
        'page1_main_category_sale': [page1_main_category_sale],
        'page1_main_category_skirts': [page1_main_category_skirts],
        'page1_main_category_trousers': [page1_main_category_trousers],
        'clicks_sale': [clicks_sale],
        'clicks_skirts': [clicks_skirts],
        'clicks_trousers': [clicks_trousers]

    })
    
    classification_input = classification_input.astype(int)
    # Predict price category
    if st.button("Predict Price Category"):
        predicted_category = classification_model.predict(classification_input)[0]
        category_result = "Yes" if predicted_category == 1 else "No"
        st.success(f"Predicted Price Category (Purchased): {category_result}")
        
        # Visualization: Bar chart for feature importance
        st.subheader("Input Features Visualization")
        fig, ax = plt.subplots(figsize=(8, 4))
        classification_input.iloc[0].plot(kind="bar", color="orange", ax=ax)
        plt.title(f"Input Features for Predicted Category: {predicted_category}")
        plt.xlabel("Features")
        plt.ylabel("Values")
        st.pyplot(fig)
