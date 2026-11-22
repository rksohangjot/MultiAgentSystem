"""Entry point: uv run streamlit run app/streamlit_app.py"""
import streamlit as st

st.set_page_config(page_title="Shop Support", page_icon="🛍️")
st.navigation([st.Page("customer.py", title="Customer chat", default=True)]).run()
