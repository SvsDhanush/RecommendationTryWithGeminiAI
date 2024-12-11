# # # # import streamlit as st
# # # # from data_processing import load_data, preprocess_data, display_data_analysis
# # # # from recommendation import display_product_recommendation

# # # # def main():
# # # #     """
# # # #     Main function to run the Streamlit app.
# # # #     """
# # # #     st.title("E-commerce Product Recommendation")

# # # #     dataset_path = 'flipkart_com-ecommerce_sample.csv'
# # # #     df = load_data(dataset_path)
    
# # # #     if df is not None:
# # # #         refined_df = preprocess_data(df)

# # # #         option = st.sidebar.selectbox("Select an option", ("Data Analysis", "Product Recommendation"))

# # # #         if option == "Data Analysis":
# # # #             display_data_analysis(refined_df)
# # # #         elif option == "Product Recommendation":
# # # #             display_product_recommendation(refined_df)

# # # # if __name__ == '__main__':
# # # #     main()
# # # # ----------------------------------------------------------------------

# # # import os
# # # import streamlit as st
# # # import pandas as pd
# # # import google.generativeai as genai
# # # from dotenv import load_dotenv
# # # from langchain_community.vectorstores import FAISS
# # # from langchain_community.embeddings import HuggingFaceEmbeddings
# # # from langchain_community.vectorstores.faiss import FAISS

# # # # Load environment variables
# # # load_dotenv()

# # # # Configure Google AI Studio API
# # # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# # # def process_data(refined_df):
# # #     """
# # #     Process the refined dataset and create the vector store.
    
# # #     Args:
# # #         refined_df (pd.DataFrame): Preprocessed dataset DataFrame.
        
# # #     Returns:
# # #         vectorstore (FAISS): Vector store containing the processed data.
# # #     """
# # #     # Combine product information
# # #     refined_df['combined_info'] = refined_df.apply(
# # #         lambda row: f"Product ID: {row['pid']}. "
# # #                     f"Product Name: {row['product_name']}. "
# # #                     f"Primary Category: {row['primary_category']}. "
# # #                     f"Retail Price: ${row['retail_price']}. "
# # #                     f"Brand: {row['brand']}. "
# # #                     f"Gender: {row['gender']}", 
# # #         axis=1
# # #     )

# # #     # Use HuggingFace embeddings as an alternative to OpenAI
# # #     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
# # #     # Create vector store
# # #     vectorstore = FAISS.from_texts(
# # #         refined_df['combined_info'].tolist(), 
# # #         embeddings
# # #     )
# # #     print("Vector store created successfully.")
# # #     print(f"Number of vectors: {vectorstore.num_vectors}")
# # #     print(vectorstore)

# # #     return vectorstore

# # # # ----------------------------------------------------------------------
# # # # def get_gemini_recommendations(department, category, brand, price):
# # # #     try:
# # # #         prompt = f"""
# # # #         Act as a product recommendation assistant. 
# # # #         Suggest three similar products with the following constraints:
        
# # # #         Product Department: {department}
# # # #         Product Category: {category}
# # # #         Product Brand: {brand}
# # # #         Maximum Price Range: {price}
        
# # # #         For each product, provide:
# # # #         1. Full Product Name
# # # #         2. Category
# # # #         3. Exact Price
# # # #         4. Brief Description
# # # #         5. Availability Status
# # # #         """
# # # #         response = genai.generate_text(
# # # #             model="gemini-pro",
# # # #             prompt=prompt
# # # #         )
# # # #         return response.text
# # # #     except AttributeError as e:
# # # #         return f"API attribute error: {e}"
# # # #     except Exception as e:
# # # #         return f"An error occurred: {e}"
# # # # ----------------------------------------------------------------------

# # # def get_gemini_recommendations(department, category, brand, price):
# # #     """
# # #     Generate product recommendations using Gemini API.
    
# # #     Args:
# # #         department (str): Product department
# # #         category (str): Product category
# # #         brand (str): Product brand
# # #         price (str): Maximum price range
    
# # #     Returns:
# # #         str: Recommendation response
# # #     """
# # #     # Configure Gemini model
# # #     model = genai.GenerativeModel('gemini-pro')
    
# # #     # Create prompt for recommendations
# # #     prompt = f"""
# # #     Act as a product recommendation assistant. 
# # #     Suggest three similar products with the following constraints:
    
# # #     Product Department: {department}
# # #     Product Category: {category}
# # #     Product Brand: {brand}
# # #     Maximum Price Range: {price}
    
# # #     For each product, provide:
# # #     1. Full Product Name
# # #     2. Category
# # #     3. Exact Price
# # #     4. Brief Description
# # #     5. Availability Status
    
# # #     Recommend products that closely match the input criteria.
# # #     """
    
# # #     # Generate recommendations
# # #     response = model.generate_content(prompt)
# # #     return response.text


# # # def display_product_recommendation(refined_df):
# # #     """
# # #     Streamlit interface for product recommendations.
    
# # #     Args:
# # #         refined_df (pd.DataFrame): Preprocessed dataset DataFrame.
# # #     """
# # #     st.title("🛍 Smart Product Recommender")
    
# # #     # Input fields
# # #     col1, col2 = st.columns(2)
    
# # #     with col1:
# # #         department = st.text_input("Product Department", placeholder="e.g., Electronics")
# # #         category = st.text_input("Product Category", placeholder="e.g., Smartphones")
    
# # #     with col2:
# # #         brand = st.text_input("Product Brand", placeholder="e.g., Apple")
# # #         price = st.text_input("Maximum Price Range", placeholder="e.g., $1000")
    
# # #     # Recommendation button
# # #     if st.button("Get Recommendations", type="primary"):
# # #         with st.spinner('Generating personalized recommendations...'):
# # #             # Get recommendations
# # #             recommendations = get_gemini_recommendations(
# # #                 department, category, brand, price
# # #             )
            
# # #             # Display recommendations
# # #             st.subheader("🌟 Recommended Products")
# # #             st.write(recommendations)

# # # # Main Streamlit app
# # # def main():
# # #     # Load your refined dataframe here
# # #     refined_df = pd.read_csv('flipkart_com-ecommerce_sample.csv')  # Replace with your actual data loading method
# # #     display_product_recommendation(refined_df)

# # # if __name__ == "__main__":
# # #     main()




# # # ----------------------------------------------------------------------------------------------------------------
# # # ----------------------------------------------------------------------------------------------------------------

# # import streamlit as st
# # import pandas as pd
# # from langchain_community.vectorstores import FAISS
# # from langchain_community.embeddings import HuggingFaceEmbeddings


# # def process_data(refined_df):
# #     """
# #     Process the refined dataset and create the vector store.
    
# #     Args:
# #         refined_df (pd.DataFrame): Preprocessed dataset DataFrame.
        
# #     Returns:
# #         vectorstore (FAISS): Vector store containing the processed data.
# #     """
# #     # Combine product information
# #     refined_df['combined_info'] = refined_df.apply(
# #         lambda row: f"Name: {row['name']}. "
# #                     f"Price: ${row['price']}. "
# #                     f"Rating: {row['rating']} stars. "
# #                     f"Reviews: {row['review_count']}. "
# #                     f"Features: {row['features']}. "
# #                     f"Sub-category: {row['sub_category']}. "
# #                     f"Category: {row['category']}.",
# #         axis=1
# #     )

# #     # Use HuggingFace embeddings
# #     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
# #     # Create vector store
# #     vectorstore = FAISS.from_texts(
# #         refined_df['combined_info'].tolist(), 
# #         embeddings
# #     )
# #     refined_df['vector_index'] = range(len(refined_df))  # Store indices for retrieval
# #     vectorstore.add_metadata(refined_df[['combined_info', 'name', 'price', 'category', 'sub_category']].to_dict(orient='records'))

# #     return vectorstore

# # def find_similar_products(vectorstore, query, num_results=3):
# #     """
# #     Find similar products using the FAISS vector store.
    
# #     Args:
# #         vectorstore (FAISS): Vector store of product embeddings.
# #         query (str): Query string describing the product.
# #         num_results (int): Number of similar products to retrieve.
    
# #     Returns:
# #         List[dict]: List of similar product metadata.
# #     """
# #     # Search for similar products in the vector store
# #     results = vectorstore.similarity_search(query, k=num_results)
# #     return results

# # def display_product_recommendation(refined_df):
# #     """
# #     Streamlit interface for product recommendations.
    
# #     Args:
# #         refined_df (pd.DataFrame): Preprocessed dataset DataFrame.
# #     """
# #     st.title("🛍 Product Recommender (Local Dataset)")
    
# #     # Input fields
# #     category = st.selectbox("Category", refined_df['category'].unique())
# #     sub_category = st.selectbox("Sub-category", refined_df[refined_df['category'] == category]['sub_category'].unique())
# #     price = st.number_input("Maximum Price", min_value=0.0, step=1.0)
    
# #     # Process the data and create a vector store
# #     vectorstore = process_data(refined_df)
    
# #     if st.button("Get Recommendations", type="primary"):
# #         with st.spinner('Finding similar products...'):
# #             # Create a query from the user's inputs
# #             query = f"Category: {category}. Sub-category: {sub_category}. Maximum Price: ${price}."
            
# #             # Get similar products from the vector store
# #             results = find_similar_products(vectorstore, query, num_results=3)
            
# #             # Display results
# #             st.subheader("🌟 Recommended Products")
# #             if results:
# #                 for i, result in enumerate(results, 1):
# #                     metadata = result['metadata']
# #                     st.markdown(
# #                         f"**{i}. {metadata['name']}**  \n"
# #                         f"- **Price**: ${metadata['price']}  \n"
# #                         f"- **Category**: {metadata['category']}  \n"
# #                         f"- **Sub-category**: {metadata['sub_category']}  \n"
# #                         f"- **Description**: {metadata['combined_info']}"
# #                     )
# #             else:
# #                 st.write("No similar products found. Try adjusting your input criteria.")

# # # Main Streamlit app
# # def main():
# #     # Load your refined dataframe here
# #     refined_df = pd.read_csv('sony_audio_all_cleandata_newFileFormat.csv')  # Replace with your actual data file

# #     # # Load the .ods file
# #     # refined_df = pd.read_excel('sony_audio_all_cleandata_newFileFormat.csv', engine='odf')

# #     # # Display the first few rows to verify
# #     # print(refined_df.head())

# #     display_product_recommendation(refined_df)

# # if __name__ == "__main__":
# #     main()

# # #----------------------------------------------------------------------------------------------------------------

# import os
# import streamlit as st
# import pandas as pd
# import google.generativeai as genai
# from dotenv import load_dotenv
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# import re

# # Load environment variables
# load_dotenv()

# # Configure Google AI Studio API
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Preprocess and build the FAISS vector store
# def preprocess_and_build_vectorstore(df, max_price=None):
#     """
#     Preprocess the dataset and create a FAISS vector store.

#     Args:
#         df (pd.DataFrame): Raw product dataset.
#         max_price (float, optional): Maximum price filter.

#     Returns:
#         vectorstore (FAISS): Vector store containing product embeddings.
#         processed_df (pd.DataFrame): Processed dataset with combined descriptions.
#     """
#     # Clean column names
#     df.columns = df.columns.str.strip()  # Strip leading/trailing spaces from column names

#     # Check if 'price' column exists
#     if 'price' not in df.columns:
#         raise ValueError("The 'price' column is missing from the dataset")

#     # Clean and standardize the price column
#     df['price'] = df['price'].apply(lambda x: re.sub(r'[^\d,]', '', str(x)))  # Remove non-numeric characters except for commas
#     df['price'] = df['price'].apply(lambda x: x.replace(',', ''))  # Remove commas
#     df['price'] = pd.to_numeric(df['price'], errors='coerce')  # Convert to numeric

#     # Drop rows with invalid price values
#     df = df.dropna(subset=['price'])

#     # Apply price filter if specified
#     if max_price is not None:
#         df = df[df['price'] <= max_price]

#     # Combine relevant fields for description
#     df['description'] = df.apply(
#         lambda row: f"Product Name: {row['name']}. "
#                     f"Price: ₹{row['price']}. "
#                     f"Rating: {row['rating']}/5. "
#                     f"Reviews: {row['review_count']} reviews. "
#                     f"Features: {row['features']}. "
#                     f"Category: {row['category']}. "
#                     f"Sub-category: {row['sub_category']}.",
#         axis=1
#     )

#     # Create a list of tuples with (description, metadata)
#     documents = [(row['description'], {'index': idx}) for idx, row in df.iterrows()]

#     # Use HuggingFace embeddings
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#     # Create FAISS vector store with metadata
#     vectorstore = FAISS.from_texts(
#         [doc[0] for doc in documents],
#         embeddings,
#         metadatas=[doc[1] for doc in documents]
#     )

#     return vectorstore, df

# def find_similar_products(vectorstore, processed_df, query):
#     """
#     Find similar products from the vector store based on the query.

#     Args:
#         vectorstore (FAISS): The FAISS vector store containing product embeddings.
#         processed_df (pd.DataFrame): Processed product DataFrame.
#         query (str): The query text to find similar products.

#     Returns:
#         pd.DataFrame: A DataFrame containing recommended products.
#     """
#     # Search for the most similar products in the vector store
#     docs_and_scores = vectorstore.similarity_search_with_score(query, k=5)

#     # Extract valid indices from metadata
#     indices = []
#     for doc, _ in docs_and_scores:
#         if 'index' in doc.metadata:  # Check if 'index' exists
#             idx = doc.metadata['index']
#             if idx in processed_df.index:  # Ensure it's a valid index
#                 indices.append(idx)

#     # Handle the case where no valid indices are found
#     if not indices:
#         return pd.DataFrame()  # Return an empty DataFrame

#     # Get the actual product data
#     recommended_products = processed_df.loc[indices]

#     return recommended_products


# # Display the product recommendation interface
# def display_recommendation_interface(processed_df, vectorstore):
#     """
#     Streamlit interface for product recommendations.

#     Args:
#         processed_df (pd.DataFrame): Processed dataset DataFrame.
#         vectorstore (FAISS): Vector store containing product embeddings.
#     """
#     # st.title("🛒 Smart Product Recommender")

#     # Input fields
#     col1, col2 = st.columns(2)

#     with col1:
#         category = st.text_input("Product Category", placeholder="e.g., Smartphones")
#         sub_category = st.text_input("Product Sub-category", placeholder="e.g., Mobile Phones")

#     with col2:
#         product = st.text_input("Product Name", placeholder="e.g., iPhone 13")

#     # Recommendation button
#     if st.button("Get Recommendations", type="primary"):
#         with st.spinner('Generating personalized recommendations...'):
#             query = f"Category: {category}, Sub-category: {sub_category}, Product: {product}"
#             recommendations = find_similar_products(vectorstore, processed_df, query)

#             # Display recommendations
#             if recommendations.empty:
#                 st.warning("No products found within the specified price range. Try adjusting the price.")
#             else:
#                 st.subheader("🌟 Recommended Products")
#                 st.write(recommendations)

# # Main Streamlit app
# def main():
#     # Load and preprocess the dataset
#     df = pd.read_csv('sony_audio_all_cleandata_newFileFormat.csv')  # Replace with your actual dataset path

#     st.title("🛒 Smart Product Recommender")
#     max_price = st.number_input("Set Maximum Price (₹):", value=10000, step=100)

#     # Preprocess the dataset and build the vector store using max_price
#     vectorstore, processed_df = preprocess_and_build_vectorstore(df, max_price=max_price)

#     if processed_df.empty:
#         st.warning("No products found within the specified price range. Try adjusting the price.")
#     else:
#         # Display the recommendation interface
#         display_recommendation_interface(processed_df, vectorstore)

# if __name__ == "__main__":
#     main()


import os
import streamlit as st
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import re
from PIL import Image
import requests
from io import BytesIO

# Load environment variables
load_dotenv()

# Configure Google AI Studio API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Preprocess and build the FAISS vector store
def preprocess_and_build_vectorstore(df, max_price=None):
    """
    Preprocess the dataset and create a FAISS vector store.

    Args:
        df (pd.DataFrame): Raw product dataset.
        max_price (float, optional): Maximum price filter.

    Returns:
        vectorstore (FAISS): Vector store containing product embeddings.
        processed_df (pd.DataFrame): Processed dataset with combined descriptions.
    """
    # Clean column names
    df.columns = df.columns.str.strip()  # Strip leading/trailing spaces from column names

    # Check if 'price' column exists
    if 'price' not in df.columns:
        raise ValueError("The 'price' column is missing from the dataset")

    # Clean and standardize the price column
    df['price'] = df['price'].apply(lambda x: re.sub(r'[^\d,]', '', str(x)))  # Remove non-numeric characters except for commas
    df['price'] = df['price'].apply(lambda x: x.replace(',', ''))  # Remove commas
    df['price'] = pd.to_numeric(df['price'], errors='coerce')  # Convert to numeric

    # Drop rows with invalid price values
    df = df.dropna(subset=['price'])

    # Apply price filter if specified
    if max_price is not None:
        df = df[df['price'] <= max_price]

    # Combine relevant fields for description
    df['description'] = df.apply(
        lambda row: f"Product Name: {row['name']}. "
                    f"Price: ₹{row['price']}. "
                    f"Rating: {row['rating']}/5. "
                    f"Reviews: {row['review_count']} reviews. "
                    f"Features: {row['features']}. "
                    f"Category: {row['category']}. "
                    f"Sub-category: {row['sub_category']}.",
        axis=1
    )

    # Create a list of tuples with (description, metadata)
    documents = [(row['description'], {'index': idx}) for idx, row in df.iterrows()]

    # Use HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create FAISS vector store with metadata
    vectorstore = FAISS.from_texts(
        [doc[0] for doc in documents],
        embeddings,
        metadatas=[doc[1] for doc in documents]
    )

    return vectorstore, df

def find_similar_products(vectorstore, processed_df, query):
    """
    Find similar products from the vector store based on the query.

    Args:
        vectorstore (FAISS): The FAISS vector store containing product embeddings.
        processed_df (pd.DataFrame): Processed product DataFrame.
        query (str): The query text to find similar products.

    Returns:
        pd.DataFrame: A DataFrame containing recommended products.
    """
    # Search for the most similar products in the vector store
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=5)

    # Extract valid indices from metadata
    indices = []
    for doc, _ in docs_and_scores:
        if 'index' in doc.metadata:  # Check if 'index' exists
            idx = doc.metadata['index']
            if idx in processed_df.index:  # Ensure it's a valid index
                indices.append(idx)

    # Handle the case where no valid indices are found
    if not indices:
        return pd.DataFrame()  # Return an empty DataFrame

    # Get the actual product data
    recommended_products = processed_df.loc[indices]

    return recommended_products

# Display recommended products as product cards
def display_product_cards(recommendations):
    """
    Display recommended products in a card format with images, names, and prices.

    Args:
        recommendations (pd.DataFrame): DataFrame containing recommended products.
    """
    st.subheader("🌟 Recommended Products")

    for _, product in recommendations.iterrows():
        col1, col2 = st.columns([1, 3])

        with col1:
            try:
                # Load the first image from the image links
                image_links = eval(product['images'])  # Convert stringified list to Python list
                if image_links:
                    response = requests.get(image_links[0])
                    img = Image.open(BytesIO(response.content))
                    st.image(img, use_container_width=True)
                else:
                    st.text("No Image Available")
            except Exception as e:
                st.text("Error loading image")

        with col2:
            st.markdown(f"**Name:** {product['name']}")
            st.markdown(f"**Price:** ₹{product['price']}")
            st.markdown(f"**Category:** {product['category']} | Sub-category: {product['sub_category']}")
            st.markdown(f"[View Details]({image_links[0]})")  # Link to the first image
        st.markdown("---")

# Display the product recommendation interface
def display_recommendation_interface(processed_df, vectorstore):
    """
    Streamlit interface for product recommendations.

    Args:
        processed_df (pd.DataFrame): Processed dataset DataFrame.
        vectorstore (FAISS): Vector store containing product embeddings.
    """
    col1, col2 = st.columns(2)

    with col1:
        category = st.text_input("Product Category", placeholder="e.g., Smartphones")
        sub_category = st.text_input("Product Sub-category", placeholder="e.g., Mobile Phones")

    with col2:
        product = st.text_input("Product Name", placeholder="e.g., iPhone 13")

    if st.button("Get Recommendations", type="primary"):
        with st.spinner('Generating personalized recommendations...'):
            query = f"Category: {category}, Sub-category: {sub_category}, Product: {product}"
            recommendations = find_similar_products(vectorstore, processed_df, query)

            if recommendations.empty:
                st.warning("No products found within the specified price range. Try adjusting the price.")
            else:
                display_product_cards(recommendations)

# Main Streamlit app
def main():
    # Load and preprocess the dataset
    df = pd.read_csv('sony_audio_all_cleandata_newFileFormat.csv')  # Replace with your actual dataset path

    st.title("🛒 Product Recommender")
    max_price = st.number_input("Set Maximum Price (₹):", value=10000, step=100)

    # Preprocess the dataset and build the vector store using max_price
    vectorstore, processed_df = preprocess_and_build_vectorstore(df, max_price=max_price)

    if processed_df.empty:
        st.warning("No products found within the specified price range. Try adjusting the price.")
    else:
        # Display the recommendation interface
        display_recommendation_interface(processed_df, vectorstore)

if __name__ == "__main__":
    main()