import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add root folder to sys path so we can import src models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.anomaly_detection import AnomalyDetector

def main():
    st.set_page_config(
        page_title="Anomaly Intelligence Dashboard",
        page_icon="📈",
        layout="wide"
    )

    st.title("📈 Anomaly Intelligence Dashboard")
    st.markdown("Explore multivariate anomaly detection with interactive visualizations.")
    st.write("---")
    
    with st.sidebar:
        st.header("Control Panel")
        uploaded_file = st.file_uploader("Upload Telemetry CSV", type=['csv'])
        st.write("---")
        
        model_options = ['dense', 'sparse', 'vae', 'denoising']
        selected_model = st.selectbox("Intelligence Core:", model_options)
        
        # Hardcoded for simplicity as per user request
        window_size = 24
        st.write("---")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Timestamp management
            if 'datetime' in df.columns:
                df['timestamp'] = pd.to_datetime(df['datetime'])
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            st.subheader("1. Data Exploration")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ["name", "datetime", "DATE", "location", "timestamp"]
            numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
            
            e_col1, e_col2 = st.columns([1, 2])
            
            with e_col1:
                st.write("**Feature Selection**")
                selected_columns = st.multiselect(
                    "Dimensions to analyze:",
                    options=numeric_cols,
                    default=numeric_cols[:3] if len(numeric_cols) > 3 else numeric_cols
                )
                
                if selected_columns:
                    st.write("**Summary Statistics**")
                    st.dataframe(df[selected_columns].describe().T)

            with e_col2:
                if selected_columns and len(selected_columns) > 1:
                    st.write("**Feature Correlations**")
                    corr = df[selected_columns].corr()
                    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", 
                                        color_continuous_scale='RdBu_r', 
                                        labels=dict(color="Correlation"))
                    fig_corr.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Select multiple features to view correlations.")

            st.write("---")
            st.subheader("2. Intelligence Analysis")
            
            if st.button("EXECUTE ANOMALY SCAN"):
                if not selected_columns:
                    st.error("Select at least one signal dimension.")
                    return
                
                with st.spinner("Analyzing signal patterns..."):
                    detector = AnomalyDetector(model_type=selected_model, window_size=window_size, num_features=len(selected_columns))
                    
                    series_data = df[selected_columns].values
                    errors, anomalies, threshold = detector.detect(series_data)
                    
                    if len(errors) == 0:
                        st.error("Signal duration insufficient for window size.")
                        return

                    # Result Summary
                    st.info(f"Scan complete using **{selected_model.upper()}** model.")
                    r1, r2, r3, r4 = st.columns(4)
                    anomaly_count = int(np.sum(anomalies))
                    r1.metric("Anomalies Detected", anomaly_count)
                    r2.metric("Mean Deviation", f"{np.mean(errors):.4f}")
                    r3.metric("Scan Threshold", f"{threshold:.4f}")
                    r4.metric("Total Data Points", len(df))

                    # Process results for viz
                    min_len = min(len(df), len(errors))
                    df_viz = df.iloc[:min_len].copy()
                    df_viz['Reconstruction_Error'] = errors[:min_len]
                    df_viz['Is_Anomaly'] = anomalies[:min_len]

                    # Visuals Grid
                    st.write("---")
                    st.subheader("3. Visualization Results")
                    
                    # Main Temporal Plot
                    fig_main = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                           vertical_spacing=0.1, 
                                           subplot_titles=("Multivariate Signal Stream", "Intelligence Deviation (Rec. Error)"))

                    plot_x = df_viz['timestamp'] if 'timestamp' in df_viz.columns else df_viz.index
                    
                    # Add all features
                    for i, col in enumerate(selected_columns):
                        fig_main.add_trace(go.Scatter(x=plot_x, y=df_viz[col], name=col, 
                                                    opacity=0.5), row=1, col=1)
                    
                    # Anomaly points (overlay on primary feature)
                    anomaly_df = df_viz[df_viz['Is_Anomaly'] == 1]
                    anomaly_x = anomaly_df['timestamp'] if 'timestamp' in anomaly_df.columns else anomaly_df.index
                    
                    if not anomaly_df.empty:
                        fig_main.add_trace(go.Scatter(x=anomaly_x, y=anomaly_df[selected_columns[0]], 
                                                mode='markers', name='Anomaly Flag',
                                                marker=dict(color='red', size=8, symbol='x')), 
                                     row=1, col=1)

                    # Error stream
                    fig_main.add_trace(go.Scatter(x=plot_x, y=df_viz['Reconstruction_Error'], name="Rec. Error",
                                            line=dict(color='purple', width=1.5)), row=2, col=1)
                    fig_main.add_trace(go.Scatter(x=[plot_x.min(), plot_x.max()], y=[threshold, threshold], 
                                            name="Threshold", line=dict(color='black', dash='dash')), row=2, col=1)

                    fig_main.update_layout(height=700, hovermode="x unified",
                                         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    
                    st.plotly_chart(fig_main, use_container_width=True)

                    # Error distribution hist
                    st.write("**Error Distribution Analysis**")
                    hist_fig = px.histogram(df_viz, x="Reconstruction_Error", nbins=50, 
                                           color_discrete_sequence=['#636EFA'], opacity=0.8)
                    hist_fig.add_vline(x=threshold, line_dash="dash", line_color="red", 
                                      annotation_text="Detection Threshold")
                    hist_fig.update_layout(height=400)
                    st.plotly_chart(hist_fig, use_container_width=True)

                    # Export Logic
                    st.write("---")
                    csv = df_viz.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📄 DOWNLOAD INTELLIGENCE REPORT",
                        data=csv,
                        file_name=f'anomaly_scan_report.csv',
                        mime='text/csv',
                    )

        except Exception as e:
            st.error(f"Analysis Failure: {e}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.info("Upload a CSV file to begin the anomaly intelligence scan.")
        st.write("Please ensure your CSV has a temporal component and numeric telemetry dimensions.")

if __name__ == "__main__":
    main()
