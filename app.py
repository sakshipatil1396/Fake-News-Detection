import streamlit as st
import joblib

vectorizer=joblib.load("vectorizer.jb")
model=joblib.load("model.jb")
st.title("Fake News Detection")
st.write("Enter the news article below:")

news_input=st.text_area("News Article")
if st.button("Predict"):
    if news_input.strip():
        transform_input=vectorizer.transform([news_input])
        prediction=model.predict(transform_input)

        if prediction[0]==1:
            st.success("The news article is Real.")
        else:
            st.error("The news article is Fake.")
    else:
        st.warning("Please enter a news article to get a prediction.")