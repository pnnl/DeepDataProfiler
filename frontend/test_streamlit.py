import streamlit as st
import pandas as pd
import boto3


# Retrieve file contents.
# Uses st.cache to only rerun when the query changes or after 10 min.
@st.cache(ttl=600)
def read_file(filename="re.csv"):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket= 'ddp-streamlit-data', Key= filename)
#     s3 = boto3.resource('s3', region_name='us-west-2')
#     bucket = s3.Bucket('ddp-streamlit-data')
#     obj = bucket.Object(filename)
    initial_df = pd.read_csv(obj["Body"])  # 'Body' is a key word
    return initial_df


content = read_file()

st.write(content)
