import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from datetime import datetime
import os
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="💸 EarnersHub — Review Analyzer",
    page_icon="💸",
    layout="centered"
)

# Title
st.title("💸 EarnersHub — Earning App Reviews Platform")

# CSV file path
csv_file = 'data/raw/review_data.csv'

# Load or initialize review data
if os.path.exists(csv_file):
    data = pd.read_csv(csv_file)
else:
    data = pd.DataFrame(columns=["Date", "Review", "Sentiment"])

# Train model if data exists
if not data.empty:
    with st.spinner("🔍 Training ML model..."):
        X = data['Review']
        y = data['Sentiment']
        vectorizer = CountVectorizer()
        X_vectorized = vectorizer.fit_transform(X)
        model = MultinomialNB()
        model.fit(X_vectorized, y)

# Sidebar menu
with st.sidebar:
    st.image("assets/logo.png", width=150)
    st.title("📖 Menu")
    option = st.radio("Choose an option:", (
        "📝 Submit a Review",
        "📋 View Reviews",
        "📊 Check App Credibility",
        "ℹ️ About"
    ))

# 📝 Submit a Review
if option == "📝 Submit a Review":
    st.header("📝 Submit a New Review")
    user_review = st.text_area("✍️ Write your review here:")

    if st.button("🚀 Analyze & Save"):
        if user_review.strip() == "":
            st.warning("⚠️ Please write a review first.")
        elif data.empty:
            st.error("❌ Not enough data to train model. Add a few demo reviews first.")
        else:
            review_vector = vectorizer.transform([user_review])
            prediction = model.predict(review_vector)[0]
            st.success(f"✅ Sentiment: **{prediction.upper()}**")

            today = datetime.now().strftime("%Y-%m-%d")
            new_entry = pd.DataFrame([[today, user_review, prediction]], columns=["Date", "Review", "Sentiment"])
            data = pd.concat([data, new_entry], ignore_index=True)
            data.to_csv(csv_file, index=False)
            st.balloons()

# 📋 View Reviews
elif option == "📋 View Reviews":
    st.header("📜 All Submitted Reviews")
    if not data.empty:
        st.dataframe(data, use_container_width=True)
    else:
        st.info("💡 No reviews yet. Submit some in 'Submit a Review'!")

# 📊 Check Credibility
elif option == "📊 Check App Credibility":
    st.header("📊 App Credibility Score")
    if not data.empty:
        positive_count = len(data[data['Sentiment'] == 'positive'])
        negative_count = len(data[data['Sentiment'] == 'negative'])
        total = positive_count + negative_count

        if total > 0:
            pos_percent = (positive_count / total) * 100

            # Status card
            st.subheader("📈 Current Credibility Status")
            if pos_percent >= 70:
                st.success("✅ TRUSTED App 📈")
            elif 40 <= pos_percent < 70:
                st.warning("⚠️ RISKY App ⚠️")
            else:
                st.error("❌ SCAM / Untrustworthy App ❌")

            st.markdown("---")

            # Pie chart with Plotly
            chart_df = pd.DataFrame({
                'Sentiment': ['Positive', 'Negative'],
                'Count': [positive_count, negative_count]
            })

            fig = px.pie(
                chart_df,
                names='Sentiment',
                values='Count',
                color='Sentiment',
                color_discrete_map={'Positive': 'green', 'Negative': 'red'},
                title='📊 Review Sentiment Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Not enough reviews yet to assess credibility.")
    else:
        st.info("ℹ️ No reviews available yet. Submit some first.")

# ℹ️ About
elif option == "ℹ️ About":
    st.header("ℹ️ About EarnersHub")
    st.write("""
        💸 **EarnersHub** lets users share and view reviews of online earning apps.  
        It uses a **Naive Bayes ML model** trained on submitted reviews to classify them as **positive** or **negative**,  
        and generates a credibility score for apps based on sentiment distribution.

        Built with ❤️ using **Streamlit**, **Scikit-learn**, and **Plotly**.
    """)
    st.markdown("**Developer:** Gubbala Adi Shankar ✌️")

# Footer
st.markdown("---")
st.caption("© 2024 EarnersHub | AI-powered Review Platform")
