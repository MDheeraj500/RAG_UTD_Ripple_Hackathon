import streamlit as st
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import os
from claims_analysis_agent import ClaimsAnalysisAgent
from claims_data_loader import ClaimsDataLoader

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

def initialize_agents():
    """Initialize the claims analysis agent and data loader"""
    claims_agent = ClaimsAnalysisAgent(groq_api_key)
    data_loader = ClaimsDataLoader()
    
    # Set sample policy data
    policy_data = {
        "policy_type": "Health Insurance",
        "coverage_limit": 100000.00,
        "deductible": 1000.00,
        "status": "Active"
    }
    claims_agent.set_policy_context(policy_data)
    
    return claims_agent, data_loader

def render_historical_data_tab(data_loader):
    """Render historical claims data visualization"""
    st.header("📊 Historical Claims Analysis")
    
    # Get claims statistics
    stats = data_loader.get_claim_statistics()
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Claims", stats.get('total_claims', 0))
    with col2:
        st.metric("Approval Rate", f"{stats.get('approval_rate', 0):.1f}%")
    with col3:
        st.metric("Avg Amount", f"${stats.get('average_amount', 0):,.2f}")
    with col4:
        st.metric("Avg Processing Time", f"{stats.get('average_processing_time', 0):.1f} days")
    
    # Load claims data for visualization
    claims_data = data_loader.load_claims_history()
    if claims_data:
        df = pd.DataFrame(claims_data)
        
        # Claims by type
        st.subheader("Claims by Type")
        fig_types = px.pie(df, names='claim_type', values='amount')
        st.plotly_chart(fig_types)
        
        # Claims amount distribution
        st.subheader("Claims Amount Distribution")
        fig_amounts = px.histogram(df, x='amount', nbins=20)
        st.plotly_chart(fig_amounts)
        
        # Claims status breakdown
        st.subheader("Claims Status")
        status_counts = df['status'].value_counts()
        fig_status = px.bar(x=status_counts.index, y=status_counts.values)
        st.plotly_chart(fig_status)
        
        # Raw data viewer
        with st.expander("View Raw Claims Data"):
            st.dataframe(df)
    else:
        st.info("No historical claims data available")

def render_claims_interface():
    st.title("🏥 Insurance Claims Analysis Advisor")
    
    # Initialize agents
    if 'claims_agent' not in st.session_state:
        claims_agent, data_loader = initialize_agents()
        st.session_state.claims_agent = claims_agent
        st.session_state.data_loader = data_loader
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 New Claim Analysis", 
        "🔍 Similar Claims", 
        "📊 Historical Data",
        "⚙️ Optimization"
    ])
    
    with tab1:
        st.header("New Claim Analysis")
        
        # Claim input form
        with st.form("claim_form"):
            claim_id = st.text_input("Claim ID", value=f"CLM{datetime.now().strftime('%Y%m%d%H%M%S')}")
            
            col1, col2 = st.columns(2)
            with col1:
                claim_type = st.selectbox(
                    "Claim Type",
                    ["Medical", "Prescription", "Dental", "Vision", "Other"]
                )
                amount = st.number_input("Claim Amount ($)", min_value=0.0, step=100.0)
            
            with col2:
                date_filed = st.date_input("Date Filed")
                urgency = st.select_slider(
                    "Urgency Level",
                    options=["Low", "Medium", "High", "Critical"]
                )
            
            description = st.text_area("Claim Description")
            additional_info = st.text_area("Additional Information")
            
            submitted = st.form_submit_button("Analyze Claim")
            
            if submitted:
                claim_details = {
                    "claim_id": claim_id,
                    "claim_type": claim_type,
                    "amount": amount,
                    "date_filed": date_filed.strftime("%Y-%m-%d"),
                    "urgency": urgency,
                    "description": description,
                    "additional_info": additional_info
                }
                
                with st.spinner("Analyzing claim..."):
                    analysis = st.session_state.claims_agent.analyze_claim(claim_details)
                    st.markdown(analysis)
                    
                    # Store analysis in session state
                    if 'claim_analyses' not in st.session_state:
                        st.session_state.claim_analyses = {}
                    st.session_state.claim_analyses[claim_id] = {
                        'details': claim_details,
                        'analysis': analysis
                    }
    
    with tab2:
        st.header("Similar Claims Analysis")
        if 'claim_analyses' in st.session_state and st.session_state.claim_analyses:
            selected_claim = st.selectbox(
                "Select a claim to find similar cases",
                list(st.session_state.claim_analyses.keys())
            )
            
            if st.button("Find Similar Claims"):
                claim_details = st.session_state.claim_analyses[selected_claim]['details']
                with st.spinner("Searching for similar claims..."):
                    similar_claims = st.session_state.claims_agent.get_similar_claims(claim_details)
                    st.markdown(similar_claims)
        else:
            st.info("No claims analyzed yet. Please analyze a claim first.")

    with tab3:
        render_historical_data_tab(st.session_state.data_loader)
    
    with tab4:
        st.header("Settlement Optimization")
        if 'claim_analyses' in st.session_state and st.session_state.claim_analyses:
            selected_claim = st.selectbox(
                "Select a claim to optimize",
                list(st.session_state.claim_analyses.keys()),
                key="optimize_claim_select"
            )
            
            if st.button("Suggest Optimizations"):
                claim_details = st.session_state.claim_analyses[selected_claim]['details']
                with st.spinner("Generating optimization suggestions..."):
                    optimization = st.session_state.claims_agent.suggest_optimization(claim_details)
                    st.markdown(optimization)
        else:
            st.info("No claims analyzed yet. Please analyze a claim first.")

if __name__ == "__main__":
    # Check for API key
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in environment variables. Please add it to your .env file.")
        st.stop()
    
    render_claims_interface()