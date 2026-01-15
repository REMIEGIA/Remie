import streamlit as st
import pandas as pd
import numpy as np

# 標題
st.title("📊 Streamlit 入門範例")

# 子標題
st.subheader("互動式資料展示")

# 建立假資料
data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['A', 'B', 'C']
)

# 顯示表格
st.dataframe(data)

# 畫折線圖
st.line_chart(data)

# 互動元件
number = st.slider("選擇顯示的列數", 1, 20, 5)
st.write("你選擇顯示前", number, "筆資料")
st.dataframe(data.head(number))