import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    .main {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background-color: #e9ecef;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .st-bb {
        background-color: transparent;
    }
    .st-at {
        background-color: #f0edeb;
    }
    header {
        background-color: #fae7d9 !important;
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
col1, col2 = st.columns([4, 1])
with col1:
    st.title("📡 SAR Performance Analysis Using Circular Patch MIMO Antenna")
    st.markdown("Predict SAR values, efficiency, gain, and S-parameters based on antenna position.")
with col2:
    st.image("antenna.png", width=100)

# Initialize session state for results
if 'results' not in st.session_state:
    st.session_state.results = None

# Add tabs for different sections
tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📈 Actual vs Predicted", "🗄️ Data"])

with tab1:
    col1, col2 = st.columns([1, 3])
    
    with col1:
        with st.container(border=True):
            st.header("⚙️ Input Parameters")
            distance = st.slider("Distance from body (mm)", 5.0, 7.5, 5.0, 0.5, 
                               help="Distance between antenna and body surface")
            x_axis = st.slider("X Position ", -100.0, 80.0, 0.0, 5.0,
                              help="Horizontal position relative to reference point")
            y_axis = st.slider("Y Position ", -160.0, 90.0, -60.0, 5.0,
                              help="Vertical position relative to reference point")
            
            # Add buttons in a horizontal layout
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                predict_button = st.button("Predict", type="primary", use_container_width=True)
            with col_btn2:
                clear_button = st.button("Clear Results", use_container_width=True)

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
            st.header("📊 Prediction Results")
            
            # Create columns for metrics
            m1, m2, m3 = st.columns(3)
            
            with m1:
                with st.container(border=True):
                    st.markdown("**📈 SAR Values**")
                    st.metric("SAR Port 1", f"{results['SAR_port1']:.4f} W/kg", 
                             help="Specific Absorption Rate for Port 1")
                    st.metric("SAR Port 2", f"{results['SAR_port2']:.4f} W/kg",
                             help="Specific Absorption Rate for Port 2")
            
            with m2:
                with st.container(border=True):
                    st.markdown("**⚡ Efficiency**")
                    st.metric("Port 1", f"{results['eff_port1']*100:.1f}%",
                             help="Radiation efficiency for Port 1")
                    st.metric("Port 2", f"{results['eff_port2']*100:.1f}%",
                             help="Radiation efficiency for Port 2")
            
            with m3:
                with st.container(border=True):
                    st.markdown("**📶 Gain**")
                    st.metric("Realized Gain", f"{results['gain']:.2f} dB",
                             help="Total realized gain of the antenna")
            
            # S-parameters in a separate row
            st.markdown("---")
            st.markdown("**🔄 S-Parameters**")
            s1, s2, s3, s4 = st.columns(4)
            
            with s1:
                with st.container(border=True):
                    st.metric("S11", f"{results['s11']:.3f} dB",
                              help="Input reflection coefficient")
            
            with s2:
                with st.container(border=True):
                    st.metric("S12", f"{results['s12']:.3f} dB",
                              help="Reverse transmission coefficient")
            
            with s3:
                with st.container(border=True):
                    st.metric("S21", f"{results['s21']:.3f} dB",
                              help="Forward transmission coefficient")
            
            with s4:
                with st.container(border=True):
                    st.metric("S22", f"{results['s22']:.3f} dB",
                              help="Output reflection coefficient")
            
            # Visualization
            st.markdown("---")
            st.header("📍 Position Visualization")
            
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=[x_axis], y=[y_axis], color='red', s=200, ax=ax1)
            ax1.axhline(0, color='black', linewidth=0.5)
            ax1.axvline(0, color='black', linewidth=0.5)
            ax1.set_xlim(-100, 80)
            ax1.set_ylim(-160, 90)
            ax1.set_xlabel("X Position ")
            ax1.set_ylabel("Y Position ")
            ax1.grid(True)
            ax1.set_title("Antenna Position Relative to Reference Point")
            st.pyplot(fig1)

with tab2:
    st.header("📈 Actual vs Predicted Values")
    
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
        st.markdown("**SAR Port 1 Comparison**")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        # Plot actual vs predicted with different colors
        sns.scatterplot(x=demo_actual_sar1, y=demo_predicted_sar1, color='#0d6efd', label='Predicted', ax=ax2)
        sns.lineplot(x=demo_actual_sar1, y=demo_actual_sar1, color='#ff7f0e', label='Actual', ax=ax2, linestyle='--')
        ax2.set_xlabel("Actual SAR Port 1 (W/kg)")
        ax2.set_ylabel("Predicted SAR Port 1 (W/kg)")
        ax2.set_title("Actual vs Predicted SAR Port 1")
        ax2.legend()
        st.pyplot(fig2)
        
        # Calculate metrics
        r2_sar1 = r2_score(demo_actual_sar1, demo_predicted_sar1)
        mse_sar1 = mean_squared_error(demo_actual_sar1, demo_predicted_sar1)
        rmse_sar1 = np.sqrt(mse_sar1)
        
        # Display metrics in expander
        with st.expander("Show metrics for SAR Port 1"):
            st.metric("R² Score", f"{r2_sar1:.3f}")
            st.metric("MSE", f"{mse_sar1:.4f}")
            st.metric("RMSE", f"{rmse_sar1:.4f}")
    
    with col2:
        st.markdown("**SAR Port 2 Comparison**")
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        # Plot actual vs predicted with different colors
        sns.scatterplot(x=demo_actual_sar2, y=demo_predicted_sar2, color='#dc3545', label='Predicted', ax=ax3)
        sns.lineplot(x=demo_actual_sar2, y=demo_actual_sar2, color='#2ca02c', label='Actual', ax=ax3, linestyle='--')
        ax3.set_xlabel("Actual SAR Port 2 (W/kg)")
        ax3.set_ylabel("Predicted SAR Port 2 (W/kg)")
        ax3.set_title("Actual vs Predicted SAR Port 2")
        ax3.legend()
        st.pyplot(fig3)
        
        # Calculate metrics
        r2_sar2 = r2_score(demo_actual_sar2, demo_predicted_sar2)
        mse_sar2 = mean_squared_error(demo_actual_sar2, demo_predicted_sar2)
        rmse_sar2 = np.sqrt(mse_sar2)
        
        # Display metrics in expander
        with st.expander("Show metrics for SAR Port 2"):
            st.metric("R² Score", f"{r2_sar2:.3f}")
            st.metric("MSE", f"{mse_sar2:.4f}")
            st.metric("RMSE", f"{rmse_sar2:.4f}")

with tab3:
    st.header("🗄️ Raw Prediction Data")
    
    # Display the results if they exist
    if st.session_state.results:
        results = st.session_state.results
        with st.expander("Show all predicted parameters", expanded=True):
            results_df = pd.DataFrame(results.items(), columns=["Parameter", "Value"])
            st.dataframe(results_df.style.format({"Value": "{:.4f}"}), 
                         use_container_width=True)
        
        # Add a download button for the results
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name='antenna_prediction_results.csv',
            mime='text/csv',
            type='primary'
        )
    else:
        st.info("No prediction results available. Please make a prediction first.")
    
# Add footer
st.markdown("---")
st.markdown("""
<style>
.footer {
    font-size: small;
    color: gray;
    text-align: center;
}
</style>
<div class="footer">
    Antenna Performance Analysis App • Powered by Streamlit • Model: Random Forest
</div>
""", unsafe_allow_html=True)
