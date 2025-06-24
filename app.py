import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from datetime import datetime
import os
import plotly.express as px

# Set page config
st.set_page_config(page_title="💸 EarnersHub — Review & Source Directory", page_icon="💸", layout="centered")

# Title and Logo
st.title("💸 EarnersHub — Reviews & Trusted Earning Sources")

# 📂 Load or initialize review data
review_file = 'data/raw/review_data.csv'
if os.path.exists(review_file):
    review_data = pd.read_csv(review_file)
else:
    review_data = pd.DataFrame(columns=["Date", "Review", "Sentiment"])

# 📂 Load or initialize earning sources data
sources_file = 'data/raw/earning_sources.csv'
if os.path.exists(sources_file):
    sources_data = pd.read_csv(sources_file)
else:
    sources_data = pd.DataFrame(columns=["Date", "Name", "Type", "Link", "Submitted_By", "Trust_Status"])

# 📊 Train model if data exists
if not review_data.empty:
    X = review_data['Review']
    y = review_data['Sentiment']
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X)
    model = MultinomialNB()
    model.fit(X_vectorized, y)

# 📖 Sidebar Menu
with st.sidebar:
    st.title("📖 Menu")
    option = st.radio("Choose an option:", (
        "📝 Submit a Review",
        "📋 View Reviews",
        "📊 Check App Credibility",
        "📖 Browse Earning Sources",
        "➕ Add a New Earning Source",
        "ℹ️ About"
    ))

# 📝 Submit a Review
if option == "📝 Submit a Review":
    st.header("📝 Submit a New Review")
    user_review = st.text_area("✍️ Write your review here:")

    if st.button("🚀 Analyze & Save"):
        if user_review.strip() == "":
            st.warning("⚠️ Please write a review first.")
        elif review_data.empty:
            st.error("❌ Not enough data to train model. Add a few demo reviews first.")
        else:
            review_vector = vectorizer.transform([user_review])
            prediction = model.predict(review_vector)[0]

            st.success(f"✅ Sentiment: **{prediction.upper()}**")

            today = datetime.now().strftime("%Y-%m-%d")
            new_entry = pd.DataFrame([[today, user_review, prediction]], columns=["Date", "Review", "Sentiment"])
            review_data = pd.concat([review_data, new_entry], ignore_index=True)
            review_data.to_csv(review_file, index=False)
            st.balloons()

# 📋 View Reviews
elif option == "📋 View Reviews":
    st.header("📜 All Submitted Reviews")
    if not review_data.empty:
        st.dataframe(review_data, use_container_width=True)
    else:
        st.info("💡 No reviews yet. Submit some in 'Submit a Review'!")

# 📊 Check Credibility
elif option == "📊 Check App Credibility":
    st.header("📊 App Credibility Score")
    if not review_data.empty:
        positive_count = len(review_data[review_data['Sentiment'] == 'positive'])
        negative_count = len(review_data[review_data['Sentiment'] == 'negative'])
        total = positive_count + negative_count

        if total > 0:
            pos_percent = (positive_count / total) * 100

            st.subheader("📈 Current Credibility Status")
            if pos_percent >= 70:
                st.success("✅ TRUSTED App 📈")
            elif 40 <= pos_percent < 70:
                st.warning("⚠️ RISKY App ⚠️")
            else:
                st.error("❌ SCAM/Untrustworthy App ❌")

            st.markdown("---")

            chart_df = pd.DataFrame({
                'Sentiment': ['Positive', 'Negative'],
                'Count': [positive_count, negative_count]
            })

            fig = px.pie(chart_df, names='Sentiment', values='Count', color='Sentiment',
                         color_discrete_map={'Positive': 'green', 'Negative': 'red'},
                         title='📊 Review Sentiment Distribution')
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("ℹ️ Not enough reviews yet to assess credibility.")
    else:
        st.info("ℹ️ No reviews available yet. Submit some first.")

# 📖 Browse Earning Sources
elif option == "📖 Browse Earning Sources":
    st.header("📖 Browse Earning Sources")
    if sources_data.empty:
        st.info("💡 No earning sources added yet.")
    else:
        type_filter = st.selectbox("🔍 Filter by Type", ["All"] + sorted(sources_data["Type"].unique()))
        if type_filter != "All":
            filtered_data = sources_data[sources_data["Type"] == type_filter]
        else:
            filtered_data = sources_data

        st.dataframe(filtered_data, use_container_width=True)

# ➕ Add a New Earning Source
elif option == "➕ Add a New Earning Source":
    st.header("➕ Add a New Earning Source")
    name = st.text_input("📛 Name of App/Channel/Website")
    source_type = st.selectbox("📂 Type", ["App", "YouTube Channel", "Website", "Telegram Group"])
    link = st.text_input("🔗 Link")
    submitted_by = st.text_input("🖊️ Your Name")
    trust_status = st.selectbox("🔍 Initial Trust Status", ["Trusted", "Risky", "Scam"])

    if st.button("✅ Submit Source"):
        if name.strip() == "" or link.strip() == "" or submitted_by.strip() == "":
            st.warning("⚠️ Please fill all fields.")
        else:
            today = datetime.now().strftime("%Y-%m-%d")
            new_source = pd.DataFrame([[today, name, source_type, link, submitted_by, trust_status]],
                                      columns=["Date", "Name", "Type", "Link", "Submitted_By", "Trust_Status"])
            sources_data = pd.concat([sources_data, new_source], ignore_index=True)
            sources_data.to_csv(sources_file, index=False)
            st.success("🎉 New source added successfully!")
            st.balloons()

# ℹ️ About Section
elif option == "ℹ️ About":
    st.header("ℹ️ About EarnersHub")
    st.write("""
        💸 **EarnersHub** lets users share and view reviews of online earning apps  
        and maintain a curated directory of trusted or flagged money-making sources.

        Built with ❤️ using **Streamlit**, **Scikit-learn**, and **Plotly**.
    """)
    st.markdown("**Developer:** Gubbala Adi Shankar ✌️")

# Footer
st.markdown("---")
st.caption("© 2024 EarnersHub | AI-powered Review & Source Platform")
