import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="🌍 Climate & Water Analysis",
    page_icon="🌍",
    layout="wide",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #0d2137);
        border-radius: 12px;
        padding: 18px 22px;
        border: 1px solid #2a5298;
        margin-bottom: 8px;
    }
    .metric-card h2 { color: #4fc3f7; font-size: 2rem; margin: 0; }
    .metric-card p  { color: #90caf9; margin: 0; font-size: 0.9rem; }
    .section-title {
        font-size: 1.5rem; font-weight: 700;
        color: #4fc3f7; margin: 24px 0 12px;
        border-left: 4px solid #4fc3f7; padding-left: 10px;
    }
    .badge-green  { background:#1b5e20; color:#a5d6a7; padding:3px 10px; border-radius:20px; font-size:.8rem; }
    .badge-blue   { background:#0d47a1; color:#90caf9; padding:3px 10px; border-radius:20px; font-size:.8rem; }
    .badge-orange { background:#bf360c; color:#ffcc80; padding:3px 10px; border-radius:20px; font-size:.8rem; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─── Load Data ─────────────────────────────────────────────────────────────────
def find_file(patterns):
    base_path = Path(__file__).parent
    for pattern in patterns:
        matches = list(base_path.glob(pattern))
        if matches:
            return matches[0]
    return None

@st.cache_data
def load_water():
    water_file = find_file(["water_resources_final_analysis*.csv"])
    if water_file is None:
        st.error("Water dataset not found. Put water_resources_final_analysis (1).csv beside app.py")
        st.stop()

    df = pd.read_csv(water_file)
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    df['water_value'] = pd.to_numeric(df['water_value'], errors='coerce')
    df = df.dropna(subset=['year', 'water_value'])
    df['year'] = df['year'].astype(int)

    if 'scarcity_level' not in df.columns:
        df['scarcity_level'] = pd.cut(
            df['water_value'],
            bins=[-1, 500, 1000, 1700, float('inf')],
            labels=['Absolute Scarcity', 'Scarcity', 'Stress', 'Abundant']
        )

    if 'world_rank' not in df.columns:
        df['world_rank'] = df.groupby('year')['water_value'].rank(ascending=False, method='dense')

    return df

@st.cache_data
def load_climate():
    climate_file = find_file(["Climate_change_Emissions_indicators*.xlsx", "Climate_change_Emissions_indicators*.csv"])
    if climate_file is None:
        st.error("Climate dataset not found. Put the Climate_change_Emissions_indicators file beside app.py")
        st.stop()

    if climate_file.suffix.lower() == '.xlsx':
        df = pd.read_excel(climate_file)
    else:
        df = pd.read_csv(climate_file)

    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    return df.dropna(subset=['Value', 'Year'])

water_df  = load_water()
climate_df = load_climate()

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://em-content.zobj.net/source/apple/354/globe-showing-europe-africa_1f30d.png", width=70)
    st.title("🌍 Dashboard")
    st.markdown("**Climate & Water Resources**\nPredictive Analytics")
    st.divider()
    st.caption("Data sources: FAO / World Bank")
    st.caption(f"🌊 Water records: {len(water_df):,}")
    st.caption(f"🌡️ Climate records: {len(climate_df):,}")

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🌊 Water Resources", "🌡️ Climate Emissions", "🤖 ML Prediction"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — WATER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">🌊 Water Resources Analysis</div>', unsafe_allow_html=True)

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    latest = water_df['year'].max()
    latest_data = water_df[water_df['year'] == latest]
    scarcity_pct = (latest_data['scarcity_level'] == 'Absolute Scarcity').mean() * 100

    with col1:
        st.markdown(f"""<div class="metric-card"><h2>{water_df['country'].nunique()}</h2>
        <p>🌐 Countries Covered</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card"><h2>{latest}</h2>
        <p>📅 Latest Year</p></div>""", unsafe_allow_html=True)
    with col3:
        avg_val = latest_data['water_value'].mean()
        st.markdown(f"""<div class="metric-card"><h2>{avg_val:,.0f}</h2>
        <p>💧 Avg Water per Capita (m³)</p></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card"><h2>{scarcity_pct:.1f}%</h2>
        <p>🔴 Absolute Scarcity Countries</p></div>""", unsafe_allow_html=True)

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("📈 Global Average Water Trend")
        trend = water_df.groupby('year')['water_value'].mean().reset_index()
        fig = px.area(trend, x='year', y='water_value',
                      color_discrete_sequence=['#4fc3f7'],
                      labels={'water_value': 'Avg m³/capita', 'year': 'Year'})
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("🥧 Scarcity Distribution (Latest Year)")
        pie_data = latest_data['scarcity_level'].value_counts().reset_index()
        pie_data.columns = ['Level', 'Count']
        fig = px.pie(pie_data, names='Level', values='Count',
                     color_discrete_sequence=px.colors.sequential.Blues_r,
                     hole=0.4)
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    # Top / Bottom countries
    st.subheader(f"🏆 Top 10 vs Bottom 10 Countries ({latest})")
    top10  = latest_data.nlargest(10, 'water_value')
    bot10  = latest_data.nsmallest(10, 'water_value')
    comp   = pd.concat([top10, bot10])
    comp['Group'] = ['Top 10'] * 10 + ['Bottom 10'] * 10
    fig = px.bar(comp, x='water_value', y='country', color='Group',
                 orientation='h', log_x=True,
                 color_discrete_map={'Top 10': '#4fc3f7', 'Bottom 10': '#ef5350'},
                 labels={'water_value': 'Water per Capita (m³, log)', 'country': ''})
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Country deep-dive
    st.subheader("🔍 Country Deep-Dive")
    selected = st.selectbox("Select a country", sorted(water_df['country'].unique()))
    cdf = water_df[water_df['country'] == selected]
    fig = px.line(cdf, x='year', y='water_value', markers=True,
                  color_discrete_sequence=['#66bb6a'],
                  labels={'water_value': 'm³/capita', 'year': 'Year'},
                  title=f"{selected} — Water per Capita over Time")
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CLIMATE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">🌡️ Climate Emissions Analysis</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="metric-card"><h2>{climate_df['Area'].nunique()}</h2>
        <p>🌐 Countries / Regions</p></div>""", unsafe_allow_html=True)
    with col2:
        yr_rng = f"{int(climate_df['Year'].min())}–{int(climate_df['Year'].max())}"
        st.markdown(f"""<div class="metric-card"><h2>{yr_rng}</h2>
        <p>📅 Year Range</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card"><h2>{climate_df['Element'].nunique()}</h2>
        <p>📊 Emission Elements</p></div>""", unsafe_allow_html=True)

    st.divider()

    # Top emitters
    latest_cl = climate_df['Year'].max()
    top_emitters = (climate_df[climate_df['Year'] == latest_cl]
                    .groupby('Area')['Value'].sum()
                    .nlargest(10).reset_index())

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader(f"🏭 Top 10 Emitters ({int(latest_cl)})")
        fig = px.bar(top_emitters, y='Area', x='Value', orientation='h',
                     color='Value', color_continuous_scale='Reds',
                     labels={'Value': 'Emission Value', 'Area': ''})
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)', height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("📊 Emissions by Element Type")
        elem_sum = climate_df.groupby('Element')['Value'].sum().nlargest(8).reset_index()
        fig = px.pie(elem_sum, names='Element', values='Value',
                     color_discrete_sequence=px.colors.sequential.Plasma_r, hole=0.3)
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    # Trends for top 5
    st.subheader("📈 Emissions Growth Trends (Top 5 Countries)")
    top5 = climate_df.groupby('Area')['Value'].sum().nlargest(5).index.tolist()
    trend5 = (climate_df[climate_df['Area'].isin(top5)]
              .groupby(['Area', 'Year'])['Value'].sum().reset_index())
    fig = px.line(trend5, x='Year', y='Value', color='Area', markers=False,
                  color_discrete_sequence=px.colors.qualitative.Vivid)
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    # Country selector
    st.subheader("🔍 Country Emissions Explorer")
    cl_country = st.selectbox("Select country/area", sorted(climate_df['Area'].unique()))
    cl_df = climate_df[climate_df['Area'] == cl_country].groupby('Year')['Value'].sum().reset_index()
    fig = px.area(cl_df, x='Year', y='Value',
                  color_discrete_sequence=['#ff7043'],
                  labels={'Value': 'Total Emissions', 'Year': 'Year'},
                  title=f"{cl_country} — Total Emissions over Time")
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">🤖 Machine Learning — Water Scarcity Prediction</div>',
                unsafe_allow_html=True)

    st.info("""
    **Goal:** Predict a country's **Water Scarcity Level** (Abundant / Stress / Scarcity / Absolute Scarcity)
    based on historical water-per-capita values and time trends.

    **Task Type:** Multi-class Classification (4 classes)
    """)

    # ── Feature Engineering ────────────────────────────────────────────────────
    @st.cache_data
    def prepare_ml(df):
        ml = df.copy()
        # Encode country
        le_country = LabelEncoder()
        ml['country_enc'] = le_country.fit_transform(ml['country'])

        # Lag features per country
        ml = ml.sort_values(['country', 'year'])
        ml['water_lag1'] = ml.groupby('country')['water_value'].shift(1)
        ml['water_lag5'] = ml.groupby('country')['water_value'].shift(5)
        ml['water_change'] = ml['water_value'] - ml['water_lag1']
        ml['water_pct_change'] = ml['water_change'] / (ml['water_lag1'] + 1)

        # Rolling mean
        ml['water_roll5'] = (ml.groupby('country')['water_value']
                               .transform(lambda x: x.rolling(5, min_periods=1).mean()))

        ml = ml.dropna()

        features = ['year', 'country_enc', 'water_lag1', 'water_lag5',
                    'water_change', 'water_pct_change', 'water_roll5', 'world_rank']
        target = 'scarcity_level'

        le_target = LabelEncoder()
        ml['target_enc'] = le_target.fit_transform(ml[target])

        X = ml[features]
        y = ml['target_enc']
        return X, y, le_target, features, ml

    X, y, le_target, feature_names, ml_df = prepare_ml(water_df)

    # ── Model Selection ─────────────────────────────────────────────────────────
    st.subheader("⚙️ Choose & Train Model")

    col_m, col_s = st.columns([1, 2])
    with col_m:
        model_choice = st.selectbox("🧠 Algorithm", [
            "Random Forest (Recommended)",
            "Gradient Boosting (XGBoost-style)",
            "Logistic Regression (Baseline)"
        ])
        test_size = st.slider("Test size %", 10, 40, 20) / 100

    with col_s:
        st.markdown("""
        | Algorithm | Strengths | Notes |
        |---|---|---|
        | **Random Forest** | Handles imbalance, robust, fast | ✅ Best for this dataset |
        | **Gradient Boosting** | High accuracy, captures non-linearity | ⚡ Slightly slower |
        | **Logistic Regression** | Interpretable baseline | ⚠️ Linear boundaries only |
        """)

    if st.button("🚀 Train Model", use_container_width=True):
        with st.spinner("Training…"):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_test)

            if "Random Forest" in model_choice:
                model = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                               random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                fi = pd.Series(model.feature_importances_, index=feature_names)

            elif "Gradient Boosting" in model_choice:
                model = GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                                   random_state=42, learning_rate=0.1)
                model.fit(X_train_s, y_train)
                y_pred = model.predict(X_test_s)
                fi = pd.Series(model.feature_importances_, index=feature_names)

            else:
                model = LogisticRegression(max_iter=1000, class_weight='balanced',
                                           random_state=42, multi_class='ovr')
                model.fit(X_train_s, y_train)
                y_pred = model.predict(X_test_s)
                fi = pd.Series(np.abs(model.coef_).mean(axis=0), index=feature_names)

            acc = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(
                model,
                X_train_s if "Forest" not in model_choice else X_train,
                y_train, cv=5, scoring='accuracy'
            )

            # ── Results Layout ─────────────────────────────────────────────────
            st.divider()
            st.subheader("📊 Model Performance")

            c1, c2, c3 = st.columns(3)
            c1.metric("🎯 Test Accuracy",  f"{acc*100:.2f}%")
            c2.metric("📉 CV Mean",        f"{cv_scores.mean()*100:.2f}%")
            c3.metric("📏 CV Std",         f"±{cv_scores.std()*100:.2f}%")

            col_fi, col_cm = st.columns(2)

            with col_fi:
                st.subheader("🔑 Feature Importance")
                fi_df = fi.sort_values(ascending=True).reset_index()
                fi_df.columns = ['Feature', 'Importance']
                fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                             color='Importance', color_continuous_scale='Blues',
                             labels={'Feature': ''})
                fig.update_layout(template='plotly_dark',
                                  paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

            with col_cm:
                st.subheader("🗺️ Confusion Matrix")
                labels_ord = le_target.classes_
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(cm,
                                x=labels_ord, y=labels_ord,
                                color_continuous_scale='Blues',
                                labels={'x': 'Predicted', 'y': 'Actual', 'color': 'Count'},
                                text_auto=True)
                fig.update_layout(template='plotly_dark',
                                  paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

            # Classification report
            st.subheader("📋 Per-class Report")
            report = classification_report(
                y_test, y_pred,
                target_names=le_target.classes_,
                output_dict=True
            )
            rep_df = pd.DataFrame(report).T.round(3)
            rep_df = rep_df.drop(index=['accuracy'], errors='ignore')
            st.dataframe(rep_df.style.background_gradient(cmap='Blues', subset=['f1-score']),
                         use_container_width=True)

            # ── Predictions vs Actual over time ───────────────────────────────
            st.subheader("📅 Prediction vs Actual (Test Set Sample)")
            test_idx = X_test.index[:200]
            sample = ml_df.loc[test_idx].copy()
            sample['Predicted'] = le_target.inverse_transform(y_pred[:200])
            sample['Actual']    = le_target.inverse_transform(y_test.values[:200])
            sample = sample.sort_values('year')

            fig = go.Figure()
            for label in le_target.classes_:
                sub = sample[sample['Actual'] == label]
                fig.add_trace(go.Scatter(x=sub['year'], y=sub['water_value'],
                                         mode='markers', name=f'Actual: {label}',
                                         marker=dict(size=6, opacity=0.7)))
            fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)',
                              title='Water Value by Scarcity Class (Test Set)')
            st.plotly_chart(fig, use_container_width=True)

            st.success(f"✅ Model trained! Test Accuracy: **{acc*100:.2f}%** | CV: **{cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%**")

    st.divider()
    st.markdown("""
    **📌 Why Random Forest is the best choice here:**
    - The target has **4 classes** with class imbalance (Abundant >> others) → RF handles it with `class_weight='balanced'`
    - Features include **lag values and rolling means** → RF captures non-linear interactions naturally
    - **Robust to outliers** in water_value (which you flagged with IQR method)
    - Easy to interpret via feature importance
    - **Gradient Boosting** is a strong alternative if you want higher accuracy at the cost of training time
    """)
