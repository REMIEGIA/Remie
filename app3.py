import streamlit as st
import pandas as pd
import numpy as np

st.title("📊 資料視覺化範例 (Visualization Example)")

df = pd.DataFrame({
    "x": np.arange(1, 101),
    "y": np.random.randn(100).cumsum()
})

chart_type = st.radio("選擇圖表類型", ["折線圖", "長條圖", "散點圖"])

if chart_type == "折線圖":
    st.line_chart(df, x="x", y="y")
elif chart_type == "長條圖":
    st.bar_chart(df, x="x", y="y")
else:
    st.scatter_chart(df, x="x", y="y")

st.success("✅ 圖表生成完成！")