import streamlit as st
import pandas as pd
import numpy as np

st.title("📊 Streamlit Local App")

df = pd.DataFrame({
    "x": np.arange(1, 101),
    "y": np.random.randn(100).cumsum()
})

chart_type = st.radio("Choose chart type", ["Line", "Bar", "Scatter"])

if chart_type == "Line":
    st.line_chart(df, x="x", y="y")
elif chart_type == "Bar":
    st.bar_chart(df, x="x", y="y")
else:
    st.scatter_chart(df, x="x", y="y")
