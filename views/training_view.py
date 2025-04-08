import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class TrainingView:
    def display_results(self, results, model=None):
        """Display evaluation results in Streamlit
        
        Args:
            results: Evaluation results to display
            model: TrainingModel instance for model download
        """
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate metrics
        hit_rate = results_df["is_hit"].mean()
        total_queries = len(results_df)
        correct_predictions = results_df["is_hit"].sum()
        incorrect_predictions = total_queries - correct_predictions
        
        # Display metrics in a nice grid
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h3>Model Performance Summary</h3>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Hit Rate", f"{hit_rate:.2%}")
        with col2:
            st.metric("Total Queries", total_queries)
        with col3:
            st.metric("Correct Predictions", int(correct_predictions))
        with col4:
            st.metric("Incorrect Predictions", int(incorrect_predictions))
            
        # Add model download section
        if model:
            st.markdown("""
                <div style='text-align: center; margin: 2rem 0;'>
                    <h3>Download Trained Model</h3>
                </div>
            """, unsafe_allow_html=True)
            
            try:
                zip_data, filename = model.get_model_download()
                st.download_button(
                    label="📥 Download Model as ZIP",
                    data=zip_data,
                    file_name=filename,
                    mime="application/zip",
                    help="Download the trained model files as a ZIP archive"
                )
            except Exception as e:
                st.error(f"Error preparing model download: {str(e)}")
        
        # Create a subplot with two visualizations
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Prediction Distribution", "Hit Rate Over Queries"),
            specs=[[{"type": "pie"}, {"type": "scatter"}]]
        )
        
        # Add pie chart
        fig.add_trace(
            go.Pie(
                labels=["Correct", "Incorrect"],
                values=[correct_predictions, incorrect_predictions],
                hole=0.4,
                marker_colors=['#2ecc71', '#e74c3c']
            ),
            row=1, col=1
        )
        
        # Add cumulative accuracy line plot
        cumulative_accuracy = results_df['is_hit'].cumsum() / (pd.Series(range(1, len(results_df) + 1)))
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(results_df) + 1)),
                y=cumulative_accuracy,
                mode='lines',
                name='Cumulative Hit Rate',
                line=dict(color='#3498db', width=2)
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=400,
            showlegend=False,
            title_text="Evaluation Results Visualization",
            title_x=0.5,
            title_font_size=20,
        )
        
        # Update axes
        fig.update_xaxes(title_text="Number of Queries", row=1, col=2)
        fig.update_yaxes(title_text="Cumulative Hit Rate", row=1, col=2)
        
        # Display the figure
        st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed results table with improved formatting
        st.markdown("### Detailed Query Results")
        
        # Format the DataFrame for display
        display_df = results_df.copy()
        display_df['Status'] = display_df['is_hit'].map({True: '✅ Correct', False: '❌ Incorrect'})
        display_df = display_df[['question', 'expected_id', 'Status']]
        display_df.columns = ['Question', 'Expected Document ID', 'Status']
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        ) 