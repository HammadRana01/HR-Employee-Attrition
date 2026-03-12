# src/visualization.py

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def display_visuals(df):
    st.subheader("📊 Data Visualizations")

    # Attrition Count Plot
    st.markdown("### Attrition Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='Attrition', palette='pastel', ax=ax1)
    ax1.set_title("Attrition Count")
    st.pyplot(fig1)

    # Age vs MonthlyIncome
    st.markdown("### Age vs Monthly Income")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x='Age', y='MonthlyIncome', hue='Attrition', palette='Set2', ax=ax2)
    ax2.set_title("Age vs Monthly Income by Attrition")
    st.pyplot(fig2)

    # Boxplot: Years at Company
    st.markdown("### Years at Company vs Attrition")
    fig3, ax3 = plt.subplots()
    sns.boxplot(data=df, x='Attrition', y='YearsAtCompany', palette='coolwarm', ax=ax3)
    ax3.set_title("Years at Company by Attrition")
    st.pyplot(fig3)
