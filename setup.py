from setuptools import setup, find_packages

setup(
    name='nlp-complaint-classifier',
    version='1.0.0',
    description='NLP Banking Complaint Classifier',
    author='Navya',
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'pandas',
        'plotly',
        'scikit-learn',
        'nltk',
        'joblib',
        'transformers',
        'torch'
    ]
)
