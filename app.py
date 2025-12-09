"""
PyCausalSim Interactive Demo
============================

A web interface for exploring causal discovery and inference.
Deploy on Streamlit Cloud or run locally with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import io

# Page config
st.set_page_config(
    page_title="PyCausalSim Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    .info-box {
        background-color: #e8f4f8;
        border-left: 5px solid #667eea;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Helper Functions
# ============================================================================

@st.cache_data
def generate_ecommerce_data(n_samples=1000, seed=42):
    """Generate synthetic e-commerce data with known causal structure."""
    np.random.seed(seed)
    
    # Confounders
    traffic_source = np.random.choice(['organic', 'paid', 'social', 'direct'], n_samples, 
                                       p=[0.4, 0.3, 0.2, 0.1])
    device_type = np.random.choice(['mobile', 'desktop', 'tablet'], n_samples,
                                    p=[0.5, 0.35, 0.15])
    time_of_day = np.random.choice(['morning', 'afternoon', 'evening', 'night'], n_samples)
    
    # Encode for causal model
    traffic_encoded = (traffic_source == 'paid').astype(float) * 0.5 + \
                      (traffic_source == 'organic').astype(float) * 0.3
    device_encoded = (device_type == 'desktop').astype(float) * 0.3
    
    # Treatment variables (with confounding)
    page_load_time = 2.5 + np.random.exponential(1.0, n_samples) - 0.5 * traffic_encoded
    price = 25 + np.random.normal(0, 8, n_samples)
    
    # Outcome with TRUE causal effects
    # True effects: page_load_time = -0.03, price = -0.005, traffic = +0.02
    conversion_logit = (-1.5 
                        - 0.3 * page_load_time  # Slower = worse
                        - 0.02 * price           # Higher price = worse
                        + 0.5 * traffic_encoded  # Paid traffic = better
                        + 0.3 * device_encoded   # Desktop = better
                        + np.random.normal(0, 0.5, n_samples))
    
    conversion_prob = 1 / (1 + np.exp(-conversion_logit))
    converted = np.random.binomial(1, conversion_prob)
    
    return pd.DataFrame({
        'traffic_source': traffic_source,
        'device_type': device_type,
        'time_of_day': time_of_day,
        'page_load_time': np.round(page_load_time, 2),
        'price': np.round(price, 2),
        'converted': converted
    })


@st.cache_data
def generate_ab_test_data(n_samples=2000, true_effect=0.5, seed=42):
    """Generate A/B test data with heterogeneous effects."""
    np.random.seed(seed)
    
    user_tenure = np.random.exponential(12, n_samples)
    activity_level = np.random.uniform(0, 10, n_samples)
    treatment = np.random.binomial(1, 0.5, n_samples)
    
    # Heterogeneous effect: stronger for high-tenure users
    effect = true_effect + 0.02 * user_tenure - 0.03 * activity_level
    outcome = 5 + treatment * effect + 0.1 * activity_level + np.random.normal(0, 1, n_samples)
    
    return pd.DataFrame({
        'treatment': treatment,
        'outcome': np.round(outcome, 2),
        'user_tenure': np.round(user_tenure, 1),
        'activity_level': np.round(activity_level, 1)
    })


@st.cache_data
def generate_marketing_data(n_samples=2000, seed=42):
    """Generate marketing touchpoint data."""
    np.random.seed(seed)
    
    data = pd.DataFrame({
        'email': np.random.binomial(1, 0.3, n_samples),
        'display': np.random.binomial(1, 0.4, n_samples),
        'search': np.random.binomial(1, 0.5, n_samples),
        'social': np.random.binomial(1, 0.25, n_samples),
        'direct': np.random.binomial(1, 0.15, n_samples)
    })
    
    # True causal contribution
    conversion_prob = (0.05 
                       + 0.15 * data['search']
                       + 0.10 * data['email']
                       + 0.05 * data['display']
                       + 0.03 * data['social']
                       + 0.01 * data['direct'])
    
    data['converted'] = np.random.binomial(1, np.clip(conversion_prob, 0, 1))
    
    return data


def run_causal_analysis(data, target, treatment_vars, confounders, method='ges'):
    """Run PyCausalSim analysis."""
    try:
        from pycausalsim import CausalSimulator
        
        sim = CausalSimulator(
            data=data,
            target=target,
            treatment_vars=treatment_vars,
            confounders=confounders,
            random_state=42
        )
        
        sim.discover_graph(method=method)
        
        return sim
    except ImportError:
        st.error("PyCausalSim not installed. Install with: `pip install git+https://github.com/Bodhi8/pycausalsim.git`")
        return None


def plot_causal_graph(graph, target, treatment_vars, confounders):
    """Create interactive causal graph visualization."""
    import networkx as nx
    
    G = nx.DiGraph()
    
    # Add all nodes
    all_vars = set(graph.keys())
    for child, parents in graph.items():
        all_vars.add(child)
        all_vars.update(parents)
    
    for var in all_vars:
        G.add_node(var)
    
    # Add edges
    for child, parents in graph.items():
        for parent in parents:
            G.add_edge(parent, child)
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node traces
    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        
        if node == target:
            node_colors.append('#e74c3c')  # Red for target
        elif node in treatment_vars:
            node_colors.append('#27ae60')  # Green for treatment
        elif node in confounders:
            node_colors.append('#f39c12')  # Orange for confounders
        else:
            node_colors.append('#3498db')  # Blue for others
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            size=40,
            color=node_colors,
            line=dict(width=2, color='white')
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=400
                    ))
    
    return fig


def plot_effect_distribution(samples, effect_name, ci_lower, ci_upper):
    """Plot distribution of causal effects."""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=samples,
        nbinsx=50,
        name='Effect Distribution',
        marker_color='#667eea',
        opacity=0.7
    ))
    
    # Add CI lines
    fig.add_vline(x=ci_lower, line_dash="dash", line_color="red", 
                  annotation_text=f"95% CI Lower: {ci_lower:.4f}")
    fig.add_vline(x=ci_upper, line_dash="dash", line_color="red",
                  annotation_text=f"95% CI Upper: {ci_upper:.4f}")
    fig.add_vline(x=np.mean(samples), line_color="green", line_width=3,
                  annotation_text=f"Mean: {np.mean(samples):.4f}")
    
    fig.update_layout(
        title=f"Distribution of Causal Effect: {effect_name}",
        xaxis_title="Effect Size",
        yaxis_title="Frequency",
        height=400
    )
    
    return fig


def plot_drivers(drivers_df):
    """Plot causal drivers bar chart."""
    fig = px.bar(
        drivers_df,
        x='effect',
        y='variable',
        orientation='h',
        color='effect',
        color_continuous_scale=['#e74c3c', '#f0f0f0', '#27ae60'],
        color_continuous_midpoint=0
    )
    
    fig.update_layout(
        title="Causal Drivers (Ranked by Effect Size)",
        xaxis_title="Causal Effect",
        yaxis_title="Variable",
        height=400,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


# ============================================================================
# Main App
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 PyCausalSim</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Causal Discovery and Inference Through Simulation</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.shields.io/badge/version-0.1.0-blue", width=100)
        st.markdown("---")
        
        st.markdown("### 🎯 About")
        st.markdown("""
        PyCausalSim helps you discover **why** things happen, 
        not just **what** happened.
        
        Unlike correlation-based ML, it uses counterfactual 
        simulation to establish true causal relationships.
        """)
        
        st.markdown("---")
        st.markdown("### 📚 Resources")
        st.markdown("""
        - [GitHub Repository](https://github.com/Bodhi8/pycausalsim)
        - [Documentation](https://pycausalsim.readthedocs.io)
        - [Medium Article](https://medium.com/@briancurry)
        """)
        
        st.markdown("---")
        st.markdown("### 🚀 Install")
        st.code("pip install git+https://github.com/Bodhi8/pycausalsim.git", language="bash")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Causal Simulator",
        "📊 Marketing Attribution", 
        "🧪 A/B Test Analysis",
        "👥 Uplift Modeling",
        "📖 Learn More"
    ])
    
    # ========================================================================
    # Tab 1: Causal Simulator
    # ========================================================================
    with tab1:
        st.markdown("## Discover Causal Relationships")
        
        st.markdown("""
        <div class="info-box">
        <strong>What does this do?</strong><br>
        The Causal Simulator learns the causal structure of your data and lets you 
        simulate "what-if" scenarios. Unlike correlation, this tells you what actually 
        <em>causes</em> your outcomes to change.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Settings")
            
            data_source = st.radio(
                "Data Source",
                ["Use Demo Data", "Upload CSV"],
                help="Try with our demo e-commerce data or upload your own"
            )
            
            if data_source == "Upload CSV":
                uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
                if uploaded_file:
                    data = pd.read_csv(uploaded_file)
                else:
                    st.info("Using demo data until you upload a file")
                    data = generate_ecommerce_data()
            else:
                n_samples = st.slider("Sample Size", 500, 5000, 1000, 500)
                data = generate_ecommerce_data(n_samples)
            
            st.markdown("### 🎯 Variables")
            
            all_cols = list(data.columns)
            
            target = st.selectbox(
                "Target Variable (Outcome)",
                all_cols,
                index=all_cols.index('converted') if 'converted' in all_cols else 0
            )
            
            remaining_cols = [c for c in all_cols if c != target]
            
            treatment_vars = st.multiselect(
                "Treatment Variables (What you can control)",
                remaining_cols,
                default=['page_load_time', 'price'] if 'page_load_time' in remaining_cols else remaining_cols[:2]
            )
            
            confounders = st.multiselect(
                "Confounders (What affects both treatment & outcome)",
                [c for c in remaining_cols if c not in treatment_vars],
                default=['traffic_source', 'device_type'] if 'traffic_source' in remaining_cols else []
            )
            
            discovery_method = st.selectbox(
                "Discovery Algorithm",
                ['ges', 'pc', 'correlation', 'hybrid'],
                help="GES is recommended for most cases"
            )
        
        with col2:
            st.markdown("### 📊 Data Preview")
            st.dataframe(data.head(10), use_container_width=True)
            
            col2a, col2b, col2c = st.columns(3)
            col2a.metric("Observations", f"{len(data):,}")
            col2b.metric("Variables", len(data.columns))
            col2c.metric("Conversion Rate", f"{data[target].mean():.1%}" if data[target].dtype in ['int64', 'float64'] else "N/A")
        
        st.markdown("---")
        
        if st.button("🚀 Run Causal Discovery", type="primary", use_container_width=True):
            with st.spinner("Discovering causal structure..."):
                try:
                    from pycausalsim import CausalSimulator
                    
                    # Encode categorical variables
                    data_encoded = data.copy()
                    for col in data_encoded.select_dtypes(include=['object']).columns:
                        data_encoded[col] = pd.factorize(data_encoded[col])[0]
                    
                    sim = CausalSimulator(
                        data=data_encoded,
                        target=target,
                        treatment_vars=treatment_vars,
                        confounders=confounders,
                        random_state=42
                    )
                    
                    sim.discover_graph(method=discovery_method)
                    
                    st.session_state['simulator'] = sim
                    st.session_state['data_encoded'] = data_encoded
                    
                    st.success("✅ Causal graph discovered!")
                    
                    # Display results
                    col_res1, col_res2 = st.columns(2)
                    
                    with col_res1:
                        st.markdown("### 🕸️ Causal Graph")
                        fig = plot_causal_graph(sim.graph, target, treatment_vars, confounders)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("""
                        **Legend:**
                        - 🔴 Red = Target (outcome)
                        - 🟢 Green = Treatment (controllable)
                        - 🟠 Orange = Confounders
                        - 🔵 Blue = Other variables
                        """)
                    
                    with col_res2:
                        st.markdown("### 📈 Causal Drivers")
                        
                        with st.spinner("Ranking drivers..."):
                            drivers = sim.rank_drivers(n_simulations=200)
                            drivers_df = pd.DataFrame(drivers.drivers, columns=['variable', 'effect'])
                        
                        fig = plot_drivers(drivers_df)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Intervention simulation
                    st.markdown("---")
                    st.markdown("### 🔮 Simulate Intervention")
                    
                    col_int1, col_int2 = st.columns(2)
                    
                    with col_int1:
                        intervention_var = st.selectbox(
                            "Variable to Intervene On",
                            treatment_vars if treatment_vars else [c for c in data_encoded.columns if c != target]
                        )
                        
                        current_val = data_encoded[intervention_var].mean()
                        min_val = float(data_encoded[intervention_var].min())
                        max_val = float(data_encoded[intervention_var].max())
                        
                        intervention_val = st.slider(
                            f"Set {intervention_var} to:",
                            min_val, max_val, current_val,
                            help=f"Current mean: {current_val:.2f}"
                        )
                    
                    with col_int2:
                        if st.button("🎯 Simulate Effect", use_container_width=True):
                            with st.spinner("Running simulation..."):
                                effect = sim.simulate_intervention(
                                    variable=intervention_var,
                                    value=intervention_val,
                                    n_simulations=500
                                )
                                
                                st.markdown(f"""
                                <div class="success-box">
                                <h4>Causal Effect of {intervention_var} = {intervention_val:.2f}</h4>
                                <p><strong>Effect on {target}:</strong> {effect.point_estimate:+.4f}</p>
                                <p><strong>95% CI:</strong> [{effect.ci_lower:.4f}, {effect.ci_upper:.4f}]</p>
                                <p><strong>P-value:</strong> {effect.p_value:.4f}</p>
                                <p><strong>Significant:</strong> {'✅ Yes' if effect.p_value < 0.05 else '❌ No'}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                if effect.samples is not None:
                                    fig = plot_effect_distribution(
                                        effect.samples, intervention_var,
                                        effect.ci_lower, effect.ci_upper
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                
                except ImportError:
                    st.error("""
                    ⚠️ PyCausalSim not installed!
                    
                    Install with:
                    ```
                    pip install git+https://github.com/Bodhi8/pycausalsim.git
                    ```
                    """)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # ========================================================================
    # Tab 2: Marketing Attribution
    # ========================================================================
    with tab2:
        st.markdown("## Marketing Attribution")
        
        st.markdown("""
        <div class="info-box">
        <strong>What does this do?</strong><br>
        Move beyond last-touch attribution! This uses causal Shapley values to 
        determine the <em>true incremental value</em> of each marketing channel.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Settings")
            
            n_customers = st.slider("Number of Customers", 1000, 10000, 2000, 500, key="mkt_n")
            
            marketing_data = generate_marketing_data(n_customers)
            
            attribution_method = st.selectbox(
                "Attribution Method",
                ['shapley', 'logistic', 'markov', 'last_touch', 'first_touch'],
                help="Shapley values are recommended for causal attribution"
            )
            
            total_budget = st.number_input(
                "Total Budget ($)",
                min_value=10000,
                max_value=1000000,
                value=100000,
                step=10000
            )
        
        with col2:
            st.markdown("### 📊 Touchpoint Data")
            st.dataframe(marketing_data.head(10), use_container_width=True)
            
            col2a, col2b = st.columns(2)
            col2a.metric("Customers", f"{len(marketing_data):,}")
            col2b.metric("Conversion Rate", f"{marketing_data['converted'].mean():.1%}")
        
        if st.button("📊 Calculate Attribution", type="primary", use_container_width=True, key="mkt_btn"):
            with st.spinner("Calculating attribution..."):
                try:
                    from pycausalsim import MarketingAttribution
                    
                    attr = MarketingAttribution(
                        data=marketing_data,
                        conversion_col='converted',
                        touchpoint_cols=['email', 'display', 'search', 'social', 'direct']
                    )
                    
                    attr.fit(method=attribution_method)
                    weights = attr.get_attribution()
                    optimal = attr.optimize_budget(total_budget=total_budget)
                    
                    col_r1, col_r2 = st.columns(2)
                    
                    with col_r1:
                        st.markdown("### 📈 Attribution Weights")
                        
                        weights_df = pd.DataFrame([
                            {'Channel': k, 'Attribution': v}
                            for k, v in sorted(weights.items(), key=lambda x: x[1], reverse=True)
                        ])
                        
                        fig = px.pie(
                            weights_df, 
                            values='Attribution', 
                            names='Channel',
                            color_discrete_sequence=px.colors.qualitative.Set2
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_r2:
                        st.markdown("### 💰 Optimal Budget Allocation")
                        
                        budget_df = pd.DataFrame([
                            {'Channel': k, 'Budget': v}
                            for k, v in sorted(optimal.items(), key=lambda x: x[1], reverse=True)
                        ])
                        
                        fig = px.bar(
                            budget_df,
                            x='Channel',
                            y='Budget',
                            color='Budget',
                            color_continuous_scale='Greens'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### 📋 Summary")
                    
                    summary_data = []
                    for channel in weights.keys():
                        summary_data.append({
                            'Channel': channel,
                            'Attribution': f"{weights[channel]:.1%}",
                            'Recommended Budget': f"${optimal[channel]:,.0f}"
                        })
                    
                    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
                    
                except ImportError:
                    st.error("PyCausalSim not installed!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # ========================================================================
    # Tab 3: A/B Test Analysis
    # ========================================================================
    with tab3:
        st.markdown("## A/B Test Analysis")
        
        st.markdown("""
        <div class="info-box">
        <strong>What does this do?</strong><br>
        Go beyond simple t-tests! Uses doubly-robust estimation and analyzes 
        heterogeneous treatment effects to understand <em>who</em> responds differently.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Settings")
            
            n_users = st.slider("Number of Users", 500, 5000, 2000, 500, key="ab_n")
            true_effect = st.slider("True Effect Size", 0.0, 2.0, 0.5, 0.1, 
                                   help="The actual treatment effect in the simulation")
            
            ab_data = generate_ab_test_data(n_users, true_effect)
            
            estimation_method = st.selectbox(
                "Estimation Method",
                ['difference', 'ols', 'dr', 'ipw', 'matching'],
                index=2,
                help="DR (Doubly Robust) is recommended"
            )
        
        with col2:
            st.markdown("### 📊 Experiment Data")
            st.dataframe(ab_data.head(10), use_container_width=True)
            
            col2a, col2b, col2c = st.columns(3)
            col2a.metric("Total Users", f"{len(ab_data):,}")
            col2b.metric("Treatment Group", f"{ab_data['treatment'].sum():,}")
            col2c.metric("Control Group", f"{(1-ab_data['treatment']).sum():,}")
        
        if st.button("🧪 Analyze Experiment", type="primary", use_container_width=True, key="ab_btn"):
            with st.spinner("Analyzing..."):
                try:
                    from pycausalsim import ExperimentAnalysis
                    
                    exp = ExperimentAnalysis(
                        data=ab_data,
                        treatment='treatment',
                        outcome='outcome',
                        covariates=['user_tenure', 'activity_level']
                    )
                    
                    effect = exp.estimate_effect(method=estimation_method)
                    
                    col_r1, col_r2 = st.columns(2)
                    
                    with col_r1:
                        st.markdown("### 📈 Treatment Effect")
                        
                        st.markdown(f"""
                        <div class="success-box">
                        <h3>Effect Estimate: {effect.estimate:.4f}</h3>
                        <p><strong>95% CI:</strong> [{effect.ci_lower:.4f}, {effect.ci_upper:.4f}]</p>
                        <p><strong>P-value:</strong> {effect.p_value:.4f}</p>
                        <p><strong>Significant:</strong> {'✅ Yes (p < 0.05)' if effect.p_value < 0.05 else '❌ No'}</p>
                        <p><strong>Method:</strong> {estimation_method}</p>
                        <hr>
                        <p><em>True effect was {true_effect:.2f}</em></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_r2:
                        st.markdown("### 📊 Group Comparison")
                        
                        treated = ab_data[ab_data['treatment'] == 1]['outcome']
                        control = ab_data[ab_data['treatment'] == 0]['outcome']
                        
                        fig = go.Figure()
                        fig.add_trace(go.Box(y=control, name='Control', marker_color='#3498db'))
                        fig.add_trace(go.Box(y=treated, name='Treatment', marker_color='#27ae60'))
                        fig.update_layout(title="Outcome Distribution by Group", height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Heterogeneous effects
                    st.markdown("### 👥 Heterogeneous Effects")
                    st.markdown("*Do different users respond differently to treatment?*")
                    
                    het = exp.analyze_heterogeneity(covariates=['user_tenure'])
                    
                    if het and het[0].segments:
                        het_df = pd.DataFrame([
                            {'Segment': k, 'Effect': v}
                            for k, v in het[0].segments.items()
                        ])
                        
                        fig = px.bar(het_df, x='Segment', y='Effect', 
                                    color='Effect', color_continuous_scale='RdYlGn',
                                    color_continuous_midpoint=0)
                        fig.update_layout(title=f"Effect by {het[0].covariate} Segment")
                        st.plotly_chart(fig, use_container_width=True)
                
                except ImportError:
                    st.error("PyCausalSim not installed!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # ========================================================================
    # Tab 4: Uplift Modeling
    # ========================================================================
    with tab4:
        st.markdown("## Uplift Modeling")
        
        st.markdown("""
        <div class="info-box">
        <strong>What does this do?</strong><br>
        Identifies <em>who</em> will respond to treatment. Segments users into:
        <ul>
        <li><strong>Persuadables:</strong> Convert only if treated (target these!)</li>
        <li><strong>Sure Things:</strong> Convert anyway (don't waste budget)</li>
        <li><strong>Lost Causes:</strong> Won't convert regardless</li>
        <li><strong>Sleeping Dogs:</strong> Treatment hurts them (avoid!)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Settings")
            
            n_users_uplift = st.slider("Number of Users", 1000, 10000, 3000, 500, key="uplift_n")
            
            # Generate uplift data
            np.random.seed(42)
            feature1 = np.random.randn(n_users_uplift)
            feature2 = np.random.randn(n_users_uplift)
            treatment = np.random.binomial(1, 0.5, n_users_uplift)
            
            base_prob = 0.3 + 0.1 * feature2
            uplift_effect = 0.15 * feature1 * treatment
            outcome_prob = np.clip(base_prob + uplift_effect, 0.01, 0.99)
            outcome = np.random.binomial(1, outcome_prob)
            
            uplift_data = pd.DataFrame({
                'responsiveness': np.round(feature1, 2),
                'baseline_propensity': np.round(feature2, 2),
                'received_treatment': treatment,
                'converted': outcome
            })
            
            uplift_method = st.selectbox(
                "Uplift Method",
                ['two_model', 'x_learner', 'transformed'],
                help="Two-model is recommended for most cases"
            )
        
        with col2:
            st.markdown("### 📊 Campaign Data")
            st.dataframe(uplift_data.head(10), use_container_width=True)
            
            col2a, col2b = st.columns(2)
            col2a.metric("Total Users", f"{len(uplift_data):,}")
            col2b.metric("Overall Conversion", f"{uplift_data['converted'].mean():.1%}")
        
        if st.button("👥 Segment Users", type="primary", use_container_width=True, key="uplift_btn"):
            with st.spinner("Building uplift model..."):
                try:
                    from pycausalsim.uplift import UpliftModeler
                    
                    uplift = UpliftModeler(
                        data=uplift_data,
                        treatment='received_treatment',
                        outcome='converted',
                        features=['responsiveness', 'baseline_propensity']
                    )
                    
                    uplift.fit(method=uplift_method)
                    segments = uplift.segment_by_effect()
                    
                    col_r1, col_r2 = st.columns(2)
                    
                    with col_r1:
                        st.markdown("### 📊 Segment Distribution")
                        
                        seg_df = pd.DataFrame([
                            {
                                'Segment': s.name,
                                'Size': s.size,
                                'Predicted Uplift': s.predicted_uplift,
                                'Actual Uplift': s.actual_uplift
                            }
                            for s in segments
                        ])
                        
                        fig = px.pie(seg_df, values='Size', names='Segment',
                                    color_discrete_sequence=['#e74c3c', '#f39c12', '#3498db', '#27ae60'])
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_r2:
                        st.markdown("### 📈 Uplift by Segment")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=[s.name for s in segments],
                            y=[s.predicted_uplift for s in segments],
                            name='Predicted',
                            marker_color='#667eea'
                        ))
                        fig.add_trace(go.Bar(
                            x=[s.name for s in segments],
                            y=[s.actual_uplift for s in segments],
                            name='Actual',
                            marker_color='#27ae60'
                        ))
                        fig.update_layout(barmode='group', height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### 📋 Segment Details")
                    
                    for seg in segments:
                        emoji = "🎯" if seg.name == "Persuadables" else "✅" if seg.name == "Sure Things" else "❌" if seg.name == "Lost Causes" else "😴"
                        
                        st.markdown(f"""
                        **{emoji} {seg.name}** ({seg.size:,} users, {seg.size/len(uplift_data):.1%})
                        - Predicted uplift: {seg.predicted_uplift:.4f}
                        - Actual uplift: {seg.actual_uplift:.4f}
                        - Conversion (treated): {seg.conversion_treated:.1%}
                        - Conversion (control): {seg.conversion_control:.1%}
                        """)
                
                except ImportError:
                    st.error("PyCausalSim not installed!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # ========================================================================
    # Tab 5: Learn More
    # ========================================================================
    with tab5:
        st.markdown("## Learn More About Causal Inference")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📖 Why Causal Inference?
            
            **The Problem with Correlation**
            
            Traditional ML tells you what *predicts* outcomes, but:
            - Feature importance ≠ causal importance
            - Correlation doesn't imply causation
            - Confounding leads to wrong conclusions
            - Selection bias distorts estimates
            
            **The Solution**
            
            Causal inference answers different questions:
            - **ML:** "What will happen?"
            - **Causal:** "What would happen IF we changed X?"
            
            This is crucial for:
            - A/B test analysis
            - Marketing optimization
            - Product decisions
            - Policy evaluation
            """)
            
            st.markdown("""
            ### 🔬 Methods in PyCausalSim
            
            **Causal Discovery:**
            - PC Algorithm (constraint-based)
            - GES (score-based)
            - LiNGAM (functional)
            - NOTEARS (neural)
            
            **Effect Estimation:**
            - Inverse Probability Weighting (IPW)
            - Doubly Robust / AIPW
            - Propensity Score Matching
            - Monte Carlo Simulation
            
            **Validation:**
            - Sensitivity Analysis
            - Placebo Tests
            - Refutation Methods
            """)
        
        with col2:
            st.markdown("""
            ### 📚 Recommended Reading
            
            **Books:**
            - *Causality* by Judea Pearl
            - *The Book of Why* by Pearl & Mackenzie
            - *Causal Inference in Statistics* by Pearl et al.
            - *Mostly Harmless Econometrics* by Angrist & Pischke
            
            **Papers:**
            - "NOTEARS: DAGs via Continuous Optimization" (Zheng et al.)
            - "Double Machine Learning" (Chernozhukov et al.)
            - "Causal Forests" (Wager & Athey)
            
            **Online Resources:**
            - [DoWhy Documentation](https://py-why.github.io/dowhy/)
            - [EconML](https://econml.azurewebsites.net/)
            - [Brady Neal's Causal Course](https://www.bradyneal.com/causal-inference-course)
            """)
            
            st.markdown("""
            ### 🚀 Get Started
            
            ```python
            # Install
            pip install git+https://github.com/Bodhi8/pycausalsim.git
            
            # Quick start
            from pycausalsim import CausalSimulator
            
            sim = CausalSimulator(data, target='outcome')
            sim.discover_graph()
            
            effect = sim.simulate_intervention('treatment', new_value)
            print(effect.summary())
            ```
            """)
            
            st.markdown("""
            ### 🤝 Contributing
            
            We welcome contributions! 
            
            - [GitHub Issues](https://github.com/Bodhi8/pycausalsim/issues)
            - [Pull Requests](https://github.com/Bodhi8/pycausalsim/pulls)
            - Email: brian@vector1.ai
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Built with ❤️ by <a href="https://vector1.ai">Brian Curry</a> | 
        <a href="https://github.com/Bodhi8/pycausalsim">GitHub</a> | 
        <a href="https://medium.com/@briancurry">Medium</a></p>
        <p>© 2025 Vector1 Research | MIT License</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
