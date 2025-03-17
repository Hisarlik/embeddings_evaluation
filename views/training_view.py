import streamlit as st
import pandas as pd
import plotly.express as px

class TrainingView:
    def display_results(self, results):
        """Display evaluation results in Streamlit"""
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate metrics
        hit_rate = results_df["is_hit"].mean()
        total_queries = len(results_df)
        correct_predictions = results_df["is_hit"].sum()
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Hit Rate", f"{hit_rate:.2%}")
        with col2:
            st.metric("Total Queries", total_queries)
        with col3:
            st.metric("Correct Predictions", correct_predictions)
        
        # Create confusion matrix
        st.subheader("Results Distribution")
        fig = px.pie(
            results_df,
            names="is_hit",
            title="Correct vs Incorrect Predictions",
            labels={"is_hit": "Prediction Status"},
            values="is_hit",
            color_discrete_map={True: "green", False: "red"}
        )
        st.plotly_chart(fig)
        
        # Display detailed results table
        st.subheader("Detailed Results")
        st.dataframe(results_df) 