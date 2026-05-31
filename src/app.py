import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

tfidf    = joblib.load('tfidf.pkl')
lr_model = joblib.load('lr_model.pkl')

with open('nlp_data.pkl', 'rb') as f:
    data = pickle.load(f)

le = data['le']
df = data['df']

st.set_page_config(page_title='Complaint Classifier', layout='wide')
st.title('Banking Complaint Classifier')
st.markdown('Powered by TF-IDF and DistilBERT')
st.markdown('---')

st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Classify Complaint', 'Analytics', 'About'])

if page == 'Classify Complaint':
    st.header('Classify a Customer Complaint')
    complaint = st.text_area('Enter complaint:', height=180)
    if st.button('Classify'):
        if complaint.strip():
            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()
            text = complaint.lower()
            text = re.sub('[^a-zA-Z ]', '', text)
            words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
            clean = ' '.join(words)
            vec       = tfidf.transform([clean])
            pred      = lr_model.predict(vec)[0]
            proba     = lr_model.predict_proba(vec)[0]
            category  = le.inverse_transform([pred])[0]
            conf      = proba.max() * 100
            col1, col2 = st.columns(2)
            col1.metric('Category',   category)
            col2.metric('Confidence', f'{conf:.1f}%')
            proba_df = pd.DataFrame({'Category': le.classes_, 'Score': proba * 100})
            proba_df = proba_df.sort_values('Score', ascending=True)
            fig = px.bar(proba_df, x='Score', y='Category', orientation='h',
                         title='Confidence by Category', color='Score',
                         color_continuous_scale='reds')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning('Please enter a complaint first.')

elif page == 'Analytics':
    st.header('Complaint Analytics')
    col1, col2, col3 = st.columns(3)
    col1.metric('Total Complaints', f'{len(df):,}')
    col2.metric('Categories', df['category'].nunique())
    col3.metric('Avg Length', f"{int(df['text'].str.len().mean())} chars")
    st.markdown('---')
    cat_counts = df['category'].value_counts().reset_index()
    cat_counts.columns = ['category', 'count']
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(cat_counts, x='category', y='count',
                      title='Complaints by Category',
                      color='count', color_continuous_scale='reds')
        fig1.update_xaxes(tickangle=45)
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.pie(cat_counts, names='category', values='count',
                      title='Category Share', hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)
    df['text_length'] = df['text'].str.len()
    fig3 = px.histogram(df, x='text_length', nbins=50,
                        title='Complaint Length Distribution',
                        color_discrete_sequence=['#e74c3c'])
    fig3.update_xaxes(range=[0, 5000])
    st.plotly_chart(fig3, use_container_width=True)

elif page == 'About':
    st.header('About This Project')
    st.markdown('---')
    st.markdown('### NLP Customer Complaint Classifier')
    st.markdown('End-to-end NLP pipeline on 160K+ CFPB banking complaints.')
    st.markdown('---')
    st.markdown('### Models')
    st.markdown('- TF-IDF + Logistic Regression (baseline)')
    st.markdown('- DistilBERT (advanced)')
    st.markdown('- DistilBERT SST-2 (sentiment)')
    st.markdown('---')
    st.markdown('### Tech Stack')
    st.markdown('Python | HuggingFace | Scikit-learn | Streamlit | Plotly | NLTK')
