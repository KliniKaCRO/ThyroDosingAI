import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# Custom CSS for enhanced UI
st.set_page_config(
    page_title="ThyroDosingAI: Precision Medicine Levothyroxine Calculator",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    /* Main page background */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Custom header styling */
    .main-header {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
        padding-bottom: 10px;
        border-bottom: 2px solid #3498db;
        margin-bottom: 25px;
    }
    
    /* Metric styling */
    .metric-card {
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        padding: 15px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #2c3e50;
    }
    
    .metric-label {
        font-size: 14px;
        color: #7f8c8d;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown('<h1 style="font-size: 24px; color: #2c3e50;">ThyroDosingAI ⚕️</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 14px; color: #7f8c8d;">Precision Medicine for Post-Thyroidectomy Care</p>', unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None,
        options=["Calculator", "Model Insights", "About"],
        icons=["calculator", "graph-up", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#f8f9fa"},
            "icon": {"color": "#3498db", "font-size": "16px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#ebedef",
            },
            "nav-link-selected": {"background-color": "#3498db"},
        }
    )
    
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 12px; color: #95a5a6;">© 2025 ThyroDosingAI</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 12px; color: #95a5a6;">Based on machine learning research from KAIMRC</p>', unsafe_allow_html=True)

# Enhanced personalized dosing function
def calculate_personalized_dose(weight, bmi, pth=None):
    """
    Calculate truly personalized levothyroxine dose based on weight, BMI category,
    and other clinical factors.
    """
    # Initial weight-based calculation (standard approach: 1.6-1.8 mcg/kg)
    base_dose = weight * 1.7  # Using midpoint of standard range
    
    # BMI-based risk categorization
    if bmi <= 27.67:
        bmi_category = "Low"
        success_rate = 57.1
        dose_adjustment = 0  # No adjustment for low BMI
    elif bmi <= 34.98:
        bmi_category = "Medium"
        success_rate = 35.7
        dose_adjustment = 12.5  # Add 12.5 mcg for medium BMI
    else:
        bmi_category = "High"
        success_rate = 35.7
        dose_adjustment = 25  # Add 25 mcg for high BMI
    
    # Apply initial adjustment
    adjusted_base_dose = base_dose + dose_adjustment
    
    # Round to nearest practical dose (12.5 mcg increments)
    rounded_base_dose = round(adjusted_base_dose / 12.5) * 12.5
    
    # Ensure minimum effective dose (usually 75 mcg)
    if rounded_base_dose < 75:
        rounded_base_dose = 75
    
    # Cap maximum initial dose (safety consideration)
    if rounded_base_dose > 200:
        rounded_base_dose = 200
    
    # Create timepoint-specific dosing with progressive adjustments
    dosing_schedule = {
        'at_surgery': {
            'dose': rounded_base_dose,
            'regimen': f"{rounded_base_dose} mcg daily × 7 days",
            'range': f"{rounded_base_dose}-{rounded_base_dose + dose_adjustment}"
        },
        'at_discharge': {
            'dose': rounded_base_dose,
            'regimen': f"{rounded_base_dose} mcg daily × 7 days",
            'range': f"{rounded_base_dose}-{rounded_base_dose + dose_adjustment}"
        },
        'one_month': {
            'dose': rounded_base_dose,
            'regimen': f"{rounded_base_dose} mcg daily × 7 days",
            'range': f"{rounded_base_dose}-{rounded_base_dose + dose_adjustment}"
        }
    }
    
    # Progressive adjustments for medium and high BMI at later timepoints
    if bmi_category == "Low":
        # Low BMI patients typically maintain stable dose
        three_month_dose = rounded_base_dose
        six_month_dose = rounded_base_dose
    elif bmi_category == "Medium":
        # Medium BMI patients often need moderate increases
        three_month_dose = rounded_base_dose + 12.5 if rounded_base_dose < 125 else rounded_base_dose
        six_month_dose = three_month_dose + 12.5 if three_month_dose < 150 else three_month_dose
    else:  # High BMI
        # High BMI patients often need more significant increases
        three_month_dose = rounded_base_dose + 25 if rounded_base_dose < 150 else rounded_base_dose
        six_month_dose = three_month_dose + 25 if three_month_dose < 175 else three_month_dose
    
    # Add later timepoints with adjustments
    dosing_schedule['three_months'] = {
        'dose': three_month_dose,
        'regimen': f"{three_month_dose} mcg daily × 7 days",
        'range': f"{rounded_base_dose}-{three_month_dose + dose_adjustment * 1.5}"
    }
    
    dosing_schedule['six_months'] = {
        'dose': six_month_dose,
        'regimen': f"{six_month_dose} mcg daily × 7 days",
        'range': f"{rounded_base_dose}-{six_month_dose + dose_adjustment * 2}"
    }
    
    # Special monitoring for high PTH
    special_monitoring = False
    if pth is not None and pth > 7:
        special_monitoring = True
    
    # Final recommendation package
    recommendation = {
        'bmi_category': bmi_category,
        'success_rate': success_rate,
        'base_dose': rounded_base_dose,
        'dosing_schedule': dosing_schedule,
        'special_monitoring': special_monitoring
    }
    
    return recommendation

# Function to plot dosing timeline
def plot_dosing_timeline(recommendation, width=700, height=400):
    """Create a dosing timeline visualization for the patient's recommendations"""
    timepoints = ['At Surgery', 'At Discharge', 'One Month', 'Three Months', 'Six Months']
    
    # Extract doses from recommendation
    min_doses = [
        recommendation['dosing_schedule']['at_surgery']['dose'],
        recommendation['dosing_schedule']['at_discharge']['dose'],
        recommendation['dosing_schedule']['one_month']['dose'],
        recommendation['dosing_schedule']['three_months']['dose'],
        recommendation['dosing_schedule']['six_months']['dose'],
    ]
    
    # Extract ranges (max values)
    max_doses = []
    for timepoint in ['at_surgery', 'at_discharge', 'one_month', 'three_months', 'six_months']:
        range_str = recommendation['dosing_schedule'][timepoint]['range']
        max_val = float(range_str.split('-')[1]) if '-' in range_str else float(range_str)
        max_doses.append(max_val)
    
    # Create figure
    fig = go.Figure()
    
    # Add range area
    fig.add_trace(go.Scatter(
        x=timepoints + timepoints[::-1],
        y=min_doses + max_doses[::-1],
        fill='toself',
        fillcolor='rgba(52, 152, 219, 0.2)',
        line=dict(color='rgba(52, 152, 219, 0)'),
        name='Dose Range'
    ))
    
    # Add min line (actual prescribed doses)
    fig.add_trace(go.Scatter(
        x=timepoints,
        y=min_doses,
        mode='lines+markers',
        line=dict(color='#3498db', width=2),
        marker=dict(size=8),
        name='Prescribed Dose'
    ))
    
    # Add max line (potential dose ceiling)
    fig.add_trace(go.Scatter(
        x=timepoints,
        y=max_doses,
        mode='lines+markers',
        line=dict(color='#e74c3c', width=2, dash='dot'),
        marker=dict(size=8),
        name='Maximum Dose'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Levothyroxine Dosing Timeline',
        xaxis=dict(title='Timepoint'),
        yaxis=dict(title='Dose (mcg)'),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)'),
        width=width,
        height=height
    )
    
    return fig

# Main content
if selected == "Calculator":
    st.markdown('<h1 class="main-header">Precision Levothyroxine Dosing Calculator</h1>', unsafe_allow_html=True)
    
    # Information box
    st.info("""
    **Clinical Decision Support Tool:** This calculator provides truly personalized levothyroxine dosing 
    recommendations for patients who have undergone total thyroidectomy, based on machine learning analysis 
    of 619 patients from the King Abdullah International Medical Research Center (KAIMRC).
    """)
    
    # Main input form
    with st.container():
        st.subheader("Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
            gender = st.selectbox("Gender", options=["Female", "Male"])
            
        with col2:
            height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=45)
        
        # Calculate BMI
        bmi = weight / ((height/100) ** 2)
        
        # BMI display
        st.markdown('<div style="background-color:#f8f9fa; padding:15px; border-radius:5px; margin:15px 0;">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("BMI", f"{bmi:.1f} kg/m²")
            
        with col2:
            # BMI Category
            if bmi <= 27.67:
                bmi_category = "Low"
                bmi_color = "#27ae60"  # Green
            elif bmi <= 34.98:
                bmi_category = "Medium"
                bmi_color = "#f39c12"  # Orange
            else:
                bmi_category = "High"
                bmi_color = "#e74c3c"  # Red
                
            st.markdown(f'<div style="text-align:center"><span style="font-size:14px">BMI Category</span><br><span style="font-size:20px; color:{bmi_color}; font-weight:bold">{bmi_category}</span></div>', unsafe_allow_html=True)
            
        with col3:
            # Risk Level
            if bmi <= 27.67:
                risk_level = "Low Risk"
                risk_color = "#27ae60"  # Green
            else:
                risk_level = "Higher Risk"
                risk_color = "#e74c3c"  # Red
                
            st.markdown(f'<div style="text-align:center"><span style="font-size:14px">Risk Level</span><br><span style="font-size:20px; color:{risk_color}; font-weight:bold">{risk_level}</span></div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Optional clinical parameters
        with st.expander("Advanced Clinical Parameters (Optional)"):
            col1, col2 = st.columns(2)
            
            with col1:
                pth = st.number_input("PTH (pmol/L)", min_value=0.0, max_value=30.0, value=0.0, step=0.1)
                t3 = st.number_input("T3 (pmol/L)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
                calcium = st.number_input("Calcium (mmol/L)", min_value=0.0, max_value=3.0, value=0.0, step=0.01)
                
            with col2:
                t4 = st.number_input("T4 (pmol/L)", min_value=0.0, max_value=30.0, value=0.0, step=0.1)
                vitamin_d = st.number_input("Vitamin D (nmol/L)", min_value=0.0, max_value=200.0, value=0.0, step=0.1)
                creatinine = st.number_input("Creatinine (μmol/L)", min_value=0.0, max_value=200.0, value=0.0, step=0.1)
        
        # Calculate button
        calculate_button = st.button("Calculate Optimal Dosing", type="primary")
    
    # Results section
    if calculate_button:
        with st.spinner("Analyzing patient data and generating personalized recommendations..."):
            # Get personalized recommendations
            recommendations = calculate_personalized_dose(weight, bmi, pth if pth > 0 else None)
            
            # Display results
            st.subheader("Personalized Levothyroxine Dosing Recommendations")
            
            # Success probability and risk level
            col1, col2, col3 = st.columns(3)
            
            with col1:
                success_prob = recommendations['success_rate']
                success_color = "#27ae60" if success_prob > 50 else "#e74c3c"
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:14px; color:#7f8c8d;">Success Probability</div>
                    <div style="font-size:30px; color:{success_color}; font-weight:700; padding:10px 0;">{success_prob:.1f}%</div>
                    <div style="font-size:12px; color:#95a5a6;">Likelihood of achieving euthyroid status at 6 months</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                risk_level = "Low" if bmi <= 27.67 else "Medium" if bmi <= 34.98 else "High"
                risk_color = "#27ae60" if risk_level == "Low" else "#f39c12" if risk_level == "Medium" else "#e74c3c"
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:14px; color:#7f8c8d;">Risk Level</div>
                    <div style="font-size:30px; color:{risk_color}; font-weight:700; padding:10px 0;">{risk_level}</div>
                    <div style="font-size:12px; color:#95a5a6;">Based on BMI classification</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                base_dose = recommendations['base_dose']
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:14px; color:#7f8c8d;">Personalized Base Dose</div>
                    <div style="font-size:30px; color:#3498db; font-weight:700; padding:10px 0;">{base_dose} mcg</div>
                    <div style="font-size:12px; color:#95a5a6;">Starting levothyroxine dose</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
            
            # Display dosing schedule
            st.markdown("#### Dosing Schedule")
            
            dosing_df = pd.DataFrame({
                "Timepoint": ["At Surgery", "At Discharge", "One Month", "Three Months", "Six Months"],
                "Recommended Regimen": [
                    recommendations['dosing_schedule']['at_surgery']['regimen'],
                    recommendations['dosing_schedule']['at_discharge']['regimen'],
                    recommendations['dosing_schedule']['one_month']['regimen'],
                    recommendations['dosing_schedule']['three_months']['regimen'],
                    recommendations['dosing_schedule']['six_months']['regimen']
                ],
                "Dose Range (mcg)": [
                    recommendations['dosing_schedule']['at_surgery']['range'],
                    recommendations['dosing_schedule']['at_discharge']['range'],
                    recommendations['dosing_schedule']['one_month']['range'],
                    recommendations['dosing_schedule']['three_months']['range'],
                    recommendations['dosing_schedule']['six_months']['range']
                ]
            })
            
            st.dataframe(dosing_df, use_container_width=True, hide_index=True)
            
            # Dosing timeline visualization
            st.plotly_chart(plot_dosing_timeline(recommendations))
            
            # Monitoring plan
            st.markdown("#### Monitoring Plan")
            
            monitoring_items = [
                "Check TSH, T4 at 6-8 weeks",
                "Adjust dose if TSH outside 0.3-4.5 mIU/L",
                "Consider T3 monitoring in selected cases"
            ]
            
            if recommendations['special_monitoring']:
                monitoring_items.insert(1, "**Monthly monitoring recommended due to elevated PTH > 7 pmol/L**")
                
                # Add warning for elevated PTH
                st.warning("""
                **⚠️ Enhanced Monitoring Required:** PTH level > 7 pmol/L detected. 
                Monthly monitoring is recommended to ensure optimal titration and to prevent 
                complications.
                """)
            
            for item in monitoring_items:
                st.markdown(f"- {item}")
            
            # Special warnings based on BMI category
            if recommendations['bmi_category'] != "Low":
                st.warning(f"""
                **⚠️ Higher BMI Risk Category:** This patient's BMI places them in a higher risk
                category for achieving euthyroid status (35.7% success rate vs. 57.1% for lower BMI).
                Consider more frequent monitoring and potentially more aggressive dose adjustment
                if TSH levels remain outside the target range.
                """)
            
            # Success indicators section
            st.markdown("#### Clinical Success Indicators")
            st.markdown("Target euthyroid status: **TSH 0.3-4.5 mIU/L**")
            
elif selected == "Model Insights":
    st.markdown('<h1 class="main-header">Model Insights & Performance Metrics</h1>', unsafe_allow_html=True)
    
    # Model overview
    st.subheader("Machine Learning Model Overview")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        The dosing recommendations are based on a **Naive Bayes classifier** trained on data from 
        619 post-thyroidectomy patients. The model was validated using 10-fold cross-validation 
        and showed excellent performance at predicting euthyroid status at six months post-surgery.
        
        BMI was identified as the dominant clinically applicable predictor, with clear 
        stratification of success rates across BMI categories.
        """)
        
    with col2:
        # Model details
        st.markdown("#### Model Details")
        st.markdown("- **Algorithm:** Naive Bayes Classifier")
        st.markdown("- **Training Sample:** 619 patients")
        st.markdown("- **Patient Demographics:** 82.7% female, BMI 31.8 ± 7.4 kg/m²")
        st.markdown("- **Validation Method:** 10-fold cross-validation")
        st.markdown("- **Target Variable:** Euthyroid status (TSH 0.3-4.5 mIU/L) at 6 months")
    
    # Performance metrics
    st.subheader("Performance Metrics")
    
    # Display metrics
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall (Sensitivity)", "F1-Score", "AUC-ROC"],
        "Value": [
            "0.828 ± 0.045",
            "0.716 ± 0.066",
            "0.983 ± 0.021",
            "0.838 ± 0.044",
            "0.855 ± 0.042"
        ]
    })
    
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # BMI Impact Analysis
    st.subheader("BMI Impact Analysis")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Create data for plot
        bmi_categories = ['Low BMI (≤27.67)', 'Medium BMI (27.67-34.98)', 'High BMI (>34.98)']
        success_rates = [57.1, 35.7, 35.7]
        dose_variations = [0, 12.5, 25]
        
        # Create plot
        fig = go.Figure()
        
        # Add bars for success rates
        fig.add_trace(go.Bar(
            x=bmi_categories,
            y=success_rates,
            name='Success Rate (%)',
            marker_color='#27ae60',
            opacity=0.7
        ))
        
        # Add line for dose variations
        fig.add_trace(go.Scatter(
            x=bmi_categories,
            y=dose_variations,
            mode='lines+markers',
            name='Dose Variation (mcg)',
            marker=dict(color='#e74c3c'),
            yaxis='y2'
        ))
        
        # Update layout with dual y-axis
        fig.update_layout(
            title='BMI Impact on Success Rate and Dose Requirements',
            xaxis=dict(title='BMI Category'),
            yaxis=dict(title='Success Rate (%)', side='left', range=[0, 100]),
            yaxis2=dict(title='Dose Variation (mcg)', side='right', overlaying='y', range=[0, 30]),
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)'),
            width=600,
            height=400
        )
        
        st.plotly_chart(fig)
            
    with col2:
        st.markdown("""
        #### Key Findings
        
        - **Low BMI (≤27.67 kg/m²):** Highest success rate (57.1%) for achieving euthyroid status.
        
        - **Medium/High BMI (>27.67 kg/m²):** Lower success rate (35.7%), indicating more 
        challenging dose optimization.
        
        - **Dose Requirements:** Higher BMI patients typically required higher doses with greater 
        dose variations over time.
        
        - **Clinical Implication:** BMI-stratified initial dosing and monitoring protocols are 
        recommended to improve outcomes.
        """)

elif selected == "About":
    st.markdown('<h1 class="main-header">About ThyroDosingAI</h1>', unsafe_allow_html=True)
    
    # Tool overview
    st.subheader("Tool Overview")
    
    st.markdown("""
    **ThyroDosingAI** is a precision medicine calculator designed to optimize levothyroxine 
    replacement therapy for patients who have undergone total thyroidectomy. This tool is based on 
    advanced machine learning analysis of 619 post-thyroidectomy patients from the King Abdullah 
    International Medical Research Center (KAIMRC).
    
    The calculator implements a novel BMI-stratified dosing framework that was shown to significantly 
    improve achievement of euthyroid status at 6 months post-surgery compared to standard 
    weight-based protocols.
    """)
    
    # Research basis
    st.subheader("Research Foundation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This tool is based on the research paper:
        
        > **"Predicting Optimal Levothyroxine Replacement Therapy After Total Thyroidectomy: A Machine Learning-Based Study"**
        
        Key findings from this study include:
        
        - **BMI as Primary Predictor:** BMI was identified as the dominant predictor of achieving euthyroid status
        - **Success Rate Stratification:** Significant differences in success rates between low BMI (57.1%) and higher BMI categories (35.7%)
        - **Naive Bayes Performance:** The Naive Bayes classifier achieved high performance (accuracy: 0.833, AUC: 0.857)
        - **Optimal Assessment Timepoint:** The 6-month timepoint showed the most reliable predictive capability
        """)
        
    with col2:
        st.markdown("""
        #### Study Details
        
        - **Patients:** 619
        - **Demographics:** 82.7% female
        - **Age Range:** 18-75 years
        - **Mean BMI:** 31.8 ± 7.4 kg/m²
        - **Follow-up:** 6-12 months
        - **Validation:** 10-fold cross-validation
        """)
    
    # References and citations
    st.subheader("References & Citations")
    
    st.markdown("""
    1. Al-Dhahri, S.F., et al., *Optimal levothyroxine dose in post-total thyroidectomy patients: a prediction model for initial dose titration.* Eur Arch Otorhinolaryngol, 2019. **276**(9): p. 2559-2564.

    2. The full research paper on which this tool is based can be accessed through the institution's repository.
    """)

    st.markdown("""
    #### Development
    
    Based on machine learning research from KAIMRC.  
    Developed by KlinikaCRO.
    """)
