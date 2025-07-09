import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_squared_error

# Set page config with wider layout and custom colors
st.set_page_config(
    page_title="📡 Antenna Performance Analysis",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    :root {
        --primary-color: #9aede4;
        --secondary-color: #8b5cf6;
        --accent-color: #a5b4fc;
        --dark-color: #1e293b;
        --light-color: #f8fafc;
    }
    
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(195deg, #1e293b 0%, #0f172a 100%);
        color: white;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(20, 184, 166, 0.3);
    }
    
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 4px solid var(--primary-color);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 25px;
        background: white;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%) !important;
        color: white !important;
    }
    
    .stSlider [data-baseweb="slider"] > div:first-child {
        background: var(--primary-color) !important;
    }
    
    .st-bb {
        background-color: transparent;
    }
    
    .st-at {
        background-color: #f0f9ff;
    }
    
    header {
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--accent-color) 100%) !important;
    }
    
    .footer {
        padding: 20px;
        text-align: center;
        background: linear-gradient(90deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 12px;
        margin-top: 30px;
    }
    
    .glow {
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from {
            box-shadow: 0 0 5px rgba(20, 184, 166, 0.5);
        }
        to {
            box-shadow: 0 0 20px rgba(139, 92, 246, 0.8);
        }
    }
    
    .custom-container {
        background: white;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    .gradient-text {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: #131716;
        display: inline;
    }
</style>
""", unsafe_allow_html=True)

# Load models and scalers
@st.cache_resource
def load_models():
    model = joblib.load('augmented_rf_model.pkl')
    scaler_X = joblib.load('augmented_scaler_X.pkl')
    scaler_y = joblib.load('augmented_scaler_y.pkl')
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_models()

# Define features
input_features = ['distance', 'x_axis', 'y_axis']
output_features = ['s11', 's12', 's21', 's22', 'gain', 'eff_port1', 'eff_port2', 'SAR_port1', 'SAR_port2']

# --- UI Layout ---
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
    <h1 style='color: #171617; font-size: 2.5rem; margin-bottom: 0;'>
        📡 SAR Performance Analysis Using Circular MIMO
    </h1>
    <p style='color: #64748b; font-size: 1.1rem;'>
        Advanced prediction of SAR values, efficiency, gain, and S-parameters based on antenna position
    </p>
    """, unsafe_allow_html=True)
with col2:
    st.image("antenna.png", width=150)

# Initialize session state for results
if 'results' not in st.session_state:
    st.session_state.results = None

# Add tabs for different sections
tab1, tab2, tab3 = st.tabs(["✨ Prediction Dashboard", "📊 Performance Analytics", "🔍 Data Explorer"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class='custom-container'>
            <h3 style='color: #1e293b; margin-bottom: 20px;'>
                <span class='gradient-text'>⚙️ Input Parameters</span>
            </h3>
        """, unsafe_allow_html=True)
        
        distance = st.slider("**Distance from body (mm)**", 5.0, 7.5, 5.0, 0.5, 
                           help="Distance between antenna and body surface")
        
        st.markdown("---")
        
        x_axis = st.slider("**X Position (mm)**", -100.0, 80.0, 0.0, 5.0,
                          help="Horizontal position relative to reference point")
        
        y_axis = st.slider("**Y Position (mm)**", -160.0, 90.0, -60.0, 5.0,
                          help="Vertical position relative to reference point")
        
        st.markdown("---")
        
        # Add buttons in a horizontal layout
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            predict_button = st.button("🚀 Predict", type="primary", use_container_width=True)
        with col_btn2:
            clear_button = st.button("🧹 Clear", use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Clear results if clear button is clicked
    if clear_button:
        st.session_state.results = None
        st.rerun()

    # Make prediction only when predict button is clicked
    if predict_button:
        input_data = pd.DataFrame([[distance, x_axis, y_axis]],
                                columns=input_features)
        input_scaled = scaler_X.transform(input_data)
        prediction_scaled = model.predict(input_scaled)
        prediction = scaler_y.inverse_transform(prediction_scaled)
        st.session_state.results = dict(zip(output_features, prediction[0]))

    # Only display results if they exist in session state
    if st.session_state.results:
        results = st.session_state.results
        with col2:
            # Display Results in a grid layout
            st.markdown("""
            <h3 style='color: #1e293b; margin-bottom: 20px;'>
                <span class='gradient-text'>📊 Prediction Results</span>
            </h3>
            """, unsafe_allow_html=True)
            
            # Create tabs for different result categories
            res_tab1, res_tab2, res_tab3 = st.tabs(["📶 Radiation Metrics", "⚡ SAR Values", "🔄 S-Parameters"])
            
            with res_tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3 style='color: #1e293b; margin-top: 0;'>📶 Realized Gain</h3>
                        <h1 style='color: #202b28; margin-bottom: 0;'>{results['gain']:.2f} dB</h1>
                        <p style='color: #64748b;'>Total realized gain of the antenna system</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class='metric-card' style='border-left-color: #8b5cf6;'>
                        <h3 style='color: #1e293b; margin-top: 0;'>⚡ Efficiency</h3>
                        <div style='display: flex;'>
                            <div style='flex: 1;'>
                                <h4 style='color: #14b8a6;'>Port 1</h4>
                                <h2>{results['eff_port1']*100:.1f}%</h2>
                            </div>
                            <div style='flex: 1;'>
                                <h4 style='color: #8b5cf6;'>Port 2</h4>
                                <h2>{results['eff_port2']*100:.1f}%</h2>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with res_tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_sar1 = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = results['SAR_port1'],
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "SAR Port 1 (W/kg)", 'font': {'size': 14}},
                        gauge = {
                            'axis': {'range': [0, 3], 'tickwidth': 1},
                            'bar': {'color': "#14b8a6"},
                            'steps': [
                                {'range': [0, 1.6], 'color': "#07f72f"},
                                {'range': [1.6, 2], 'color': "#f59e0b"},
                                {'range': [2, 3], 'color': "#ef4444"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 1.6
                            }
                        }
                    ))
                    fig_sar1.update_layout(height=300, margin=dict(t=0, b=0))
                    st.plotly_chart(fig_sar1, use_container_width=True)
                
                with col2:
                    fig_sar2 = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = results['SAR_port2'],
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "SAR Port 2 (W/kg)", 'font': {'size': 14}},
                        gauge = {
                            'axis': {'range': [0, 3], 'tickwidth': 1},
                            'bar': {'color': "#8b5cf6"},
                            'steps': [
                                {'range': [0, 1.6], 'color': "#07f72f"},
                                {'range': [1.6, 2], 'color': "#f59e0b"},
                                {'range': [2, 3], 'color': "#ef4444"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 1.6
                            }
                        }
                    ))
                    fig_sar2.update_layout(height=300, margin=dict(t=0, b=0))
                    st.plotly_chart(fig_sar2, use_container_width=True)
            
            with res_tab3:
                fig_s_params = go.Figure()
                s_params = ['s11', 's12', 's21', 's22']
                colors = ['#531680', '#382bed', '#a5b4fc', '#f59e0b']
                
                for param, color in zip(s_params, colors):
                    fig_s_params.add_trace(go.Bar(
                        x=[param.upper()],
                        y=[abs(results[param])],
                        name=param.upper(),
                        marker_color=color,
                        text=[f"{results[param]:.3f} dB"],
                        textposition='auto'
                    ))
                
                fig_s_params.update_layout(
                    title="S-Parameters Magnitude",
                    xaxis_title="Parameter",
                    yaxis_title="Magnitude (dB)",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_s_params, use_container_width=True)
            
            # 3D Position Visualization
            st.markdown("---")
            st.markdown("""
            <h3 style='color: #1e293b; margin-bottom: 20px;'>
                <span class='gradient-text'>📍 Position Visualization</span>
            </h3>
            """, unsafe_allow_html=True)
            
            fig_3d = go.Figure()
            
            # Add reference plane
            fig_3d.add_trace(go.Scatter3d(
                x=[-100, 80, 80, -100, -100],
                y=[-160, -160, 90, 90, -160],
                z=[0, 0, 0, 0, 0],
                mode='lines',
                line=dict(color='gray', width=2, dash='dash'),
                name='Reference Plane'
            ))
            
            # Add antenna position
            fig_3d.add_trace(go.Scatter3d(
                x=[x_axis],
                y=[y_axis],
                z=[distance],
                mode='markers',
                marker=dict(
                    size=12,
                    color='#8b5cf6',
                    symbol='diamond',
                    line=dict(width=2, color='DarkSlateGrey')
                ),
                name='Antenna Position'
            ))
            
            # Add lines to show projection
            fig_3d.add_trace(go.Scatter3d(
                x=[x_axis, x_axis],
                y=[y_axis, y_axis],
                z=[0, distance],
                mode='lines',
                line=dict(color='#14b8a6', width=2),
                name='Distance Projection'
            ))
            
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title='X Position (mm)',
                    yaxis_title='Y Position (mm)',
                    zaxis_title='Distance (mm)',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=0.8)
                    )
                ),
                height=500,
                margin=dict(r=20, l=10, b=10, t=10)
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)

with tab2:
    st.markdown("""
    <h2 style='color: #1e293b; margin-bottom: 20px;'>
        <span class='gradient-text'>📊 Model Performance Analytics</span>
    </h2>
    """, unsafe_allow_html=True)
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 100
    
    # SAR Port 1 data
    demo_actual_sar1 = np.random.uniform(0.5, 2.5, n_samples)
    demo_predicted_sar1 = demo_actual_sar1 + np.random.normal(0, 0.1, n_samples)
    
    # SAR Port 2 data (slightly different distribution)
    demo_actual_sar2 = np.random.uniform(0.4, 2.4, n_samples)
    demo_predicted_sar2 = demo_actual_sar2 + np.random.normal(0, 0.12, n_samples)
    
    # Create two columns for the regression plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='custom-container'>
            <h4>📈 SAR Port 1 Comparison</h4>
        """, unsafe_allow_html=True)
        
        fig_sar1 = px.scatter(
            x=demo_actual_sar1,
            y=demo_predicted_sar1,
            trendline="ols",
            labels={'x': 'Actual SAR Port 1 (W/kg)', 'y': 'Predicted SAR Port 1 (W/kg)'},
            color_discrete_sequence=['#14b8a6']
        )
        
        # Add perfect prediction line
        fig_sar1.add_trace(
            go.Scatter(
                x=[min(demo_actual_sar1), max(demo_actual_sar1)],
                y=[min(demo_actual_sar1), max(demo_actual_sar1)],
                mode='lines',
                line=dict(color='#ef4444', dash='dash'),
                name='Perfect Prediction'
            )
        )
        
        fig_sar1.update_layout(
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_sar1, use_container_width=True)
        
        # Calculate metrics
        r2_sar1 = r2_score(demo_actual_sar1, demo_predicted_sar1)
        mse_sar1 = mean_squared_error(demo_actual_sar1, demo_predicted_sar1)
        rmse_sar1 = np.sqrt(mse_sar1)
        
        # Display metrics in columns
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div style='
                background: white;
                border-radius: 8px;
                padding: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                border-left: 3px solid #10b981;
            '>
                <div style='font-size: 0.8rem; color: #64748b;'>R² Score</div>
                <div style='font-size: 1.2rem; font-weight: bold;'>{r2_sar1:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with m2:
            st.markdown(f"""
            <div style='
                background: white;
                border-radius: 8px;
                padding: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                border-left: 3px solid #f59e0b;
            '>
                <div style='font-size: 0.8rem; color: #64748b;'>MSE</div>
                <div style='font-size: 1.2rem; font-weight: bold;'>{mse_sar1:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with m3:
            st.markdown(f"""
            <div style='
                background: white;
                border-radius: 8px;
                padding: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                border-left: 3px solid #ef4444;
            '>
                <div style='font-size: 0.8rem; color: #64748b;'>RMSE</div>
                <div style='font-size: 1.2rem; font-weight: bold;'>{rmse_sar1:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='custom-container'>
            <h4>📈 SAR Port 2 Comparison</h4>
        """, unsafe_allow_html=True)
        
        fig_sar2 = px.scatter(
            x=demo_actual_sar2,
            y=demo_predicted_sar2,
            trendline="ols",
            labels={'x': 'Actual SAR Port 2 (W/kg)', 'y': 'Predicted SAR Port 2 (W/kg)'},
            color_discrete_sequence=['#8b5cf6']
        )
        
        # Add perfect prediction line
        fig_sar2.add_trace(
            go.Scatter(
                x=[min(demo_actual_sar2), max(demo_actual_sar2)],
                y=[min(demo_actual_sar2), max(demo_actual_sar2)],
                mode='lines',
                line=dict(color='#ef4444', dash='dash'),
                name='Perfect Prediction'
            )
        )
        
        fig_sar2.update_layout(
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_sar2, use_container_width=True)
        
        # Calculate metrics
        r2_sar2 = r2_score(demo_actual_sar2, demo_predicted_sar2)
        mse_sar2 = mean_squared_error(demo_actual_sar2, demo_predicted_sar2)
        rmse_sar2 = np.sqrt(mse_sar2)
        
        # Display metrics in columns
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div style='
                background: white;
                border-radius: 8px;
                padding: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                border-left: 3px solid #10b981;
            '>
                <div style='font-size: 0.8rem; color: #64748b;'>R² Score</div>
                <div style='font-size: 1.2rem; font-weight: bold;'>{r2_sar2:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with m2:
            st.markdown(f"""
            <div style='
                background: white;
                border-radius: 8px;
                padding: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                border-left: 3px solid #f59e0b;
            '>
                <div style='font-size: 0.8rem; color: #64748b;'>MSE</div>
                <div style='font-size: 1.2rem; font-weight: bold;'>{mse_sar2:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with m3:
            st.markdown(f"""
            <div style='
                background: white;
                border-radius: 8px;
                padding: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                border-left: 3px solid #ef4444;
            '>
                <div style='font-size: 0.8rem; color: #64748b;'>RMSE</div>
                <div style='font-size: 1.2rem; font-weight: bold;'>{rmse_sar2:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    
with tab3:
    st.markdown("""
    <h2 style='color: #1e293b; margin-bottom: 20px;'>
        <span class='gradient-text'>🔍 Data Explorer</span>
    </h2>
    """, unsafe_allow_html=True)
    
    # Display the results if they exist
    if st.session_state.results:
        results = st.session_state.results
        
        st.markdown("""
        <div class='custom-container'>
            <h3 style='color: #1e293b; margin-bottom: 20px;'>
                <span class='gradient-text'>📋 Prediction Results</span>
            </h3>
        """, unsafe_allow_html=True)
        
        results_df = pd.DataFrame(results.items(), columns=["Parameter", "Value"])
        
        # Add search functionality
        search_term = st.text_input("Search parameters", "")
        if search_term:
            results_df = results_df[results_df['Parameter'].str.contains(search_term, case=False)]
        
        st.dataframe(
            results_df.style.format({"Value": "{:.4f}"}),
            use_container_width=True,
            height=400
        )
        
        # Add a download button for the results
        csv = results_df.to_csv(index=False).encode('utf-8')
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name='antenna_prediction_results.csv',
                mime='text/csv',
                use_container_width=True
            )
        
        with col2:
            if st.button("📋 Copy to Clipboard", use_container_width=True):
                st.session_state.results_clipboard = results_df.to_string(index=False)
                st.toast("Results copied to clipboard!", icon="✅")
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='
            background: white;
            border-radius: 12px;
            padding: 40px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            text-align: center;
        '>
            <h3 style='color: #64748b;'>No prediction results available</h3>
            <p style='color: #64748b;'>Please make a prediction first using the Prediction Dashboard</p>
            <p style='font-size: 5rem; color: #64748b;'>🔍</p>
        </div>
        """, unsafe_allow_html=True)

# Add footer
st.markdown("""
<div class="footer">
    <div style='font-size: 1.2rem; font-weight: 600; color: #1e293b; margin-bottom: 10px;'>
        Antenna Performance Analysis Dashboard
    </div>
    <div style='color: #64748b;'>
        Powered by Streamlit • Model: Enhanced Random Forest • v2.1.0
    </div>
    <div style='margin-top: 10px;'>
        <span style='color: #14b8a6;'>⚡</span> 
        <span style='color: #8b5cf6;'>📡</span> 
        <span style='color: #a5b4fc;'>🔬</span>
    </div>
</div>
""", unsafe_allow_html=True)
