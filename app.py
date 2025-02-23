import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Set page config
st.set_page_config(page_title="Investment Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for dark theme
st.markdown("""
    <style>
    /* Main page background */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    .main {
        padding: 0rem 1rem;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Metric cards */
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    div[data-testid="stMetricDelta"] {
        color: #4CAF50 !important;
    }
    div[data-testid="metric-container"] {
        background-color: #1e2130;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    div[data-testid="stHorizontalBlock"] {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #1e2130;
    }

    .stTabs [data-baseweb="tab"] {
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

def load_data():
    # Dicionário de dados mensais - Atualize aqui os valores de cada mês
    monthly_data = {
        'Jan': {
            'Total': 300000.00,
            'CDI': 1.1,
            'Rendimento_Fixo': 0,
            'Rendimento_Variavel': 0,
            'Performance_Carteira': 0.0,
            'Ops_Aberto': 0,
        },
        'Fev': {
            'Total': 304490.71,
            'CDI': 0.74,
            'Rendimento_Fixo': 1946.69,
            'Rendimento_Variavel': 2544.02,
            'Performance_Carteira': 1.50,
            'Ops_Aberto': 5316.98,
        }
    }
    
    # Criando DataFrame
    df = pd.DataFrame.from_dict(monthly_data, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Mês'}, inplace=True)
    
    # Calculando rendimento acumulado
    df['Rendimento_Acumulado'] = df['Total'] - df['Total'].iloc[0]
    df['Retorno_Percentual'] = df['Total'].pct_change() * 100
    
    return df  # ✅ Agora retorna corretamente o DataFrame
 

def create_waterfall_chart(df):
    fig = go.Figure(go.Waterfall(
        name="Variação Patrimonial",
        orientation="v",
        measure=["absolute"] + ["relative"] * (len(df) - 1),
        x=df['Mês'],
        textposition="outside",
        text=[f"R$ {x:,.2f}" for x in df['Total']],
        y=df['Total'],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    fig.update_layout(
        title="Evolução Patrimonial (Waterfall)",
        height=400,
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='#ffffff')
    )
    return fig

def create_performance_gauge(current_performance, max_performance=2):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_performance,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, max_performance]},
            'steps': [
                {'range': [0, max_performance/2], 'color': "#1e2130"},
                {'range': [max_performance/2, max_performance], 'color': "#2c3147"}],
            'threshold': {
                'line': {'color': "#4CAF50", 'width': 4},
                'thickness': 0.75,
                'value': current_performance}},
        title={'text': "Performance Atual (%)", 'font': {'color': '#ffffff'}}))
    fig.update_layout(height=250, paper_bgcolor='#0e1117', font={'color': '#ffffff'})
    return fig

def main():
    st.title("📈 Dashboard de Investimentos")
    
    # Load data
    df = load_data()
    
    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)  # Adicionada uma coluna
    
    with col1:
        st.metric(
            "Patrimônio Total",
            f"R$ {df['Total'].iloc[-1]:,.2f}",
            f"{df['Retorno_Percentual'].iloc[-1]:.2f}%" if len(df) > 1 else None,
            help="Valor total atual da carteira"
        )
    
    with col2:
        st.metric(
            "Rendimento Acumulado",
            f"R$ {df['Rendimento_Acumulado'].iloc[-1]:,.2f}",
            f"{df['Performance_Carteira'].iloc[-1]:.2f}%",
            help="Rendimento total acumulado desde o início"
        )
    
    with col3:
        st.metric(
            "CDI Acumulado",
            f"{df['CDI'].sum():.2f}%",
            help="Taxa CDI acumulada no período"
        )
    
    with col4:
        st.metric(
            "Melhor Rendimento Mensal",
            f"R$ {max(df['Rendimento_Fixo'] + df['Rendimento_Variavel']):,.2f}",
            help="Maior rendimento mensal registrado"
        )
    
    with col5:
        st.metric(
            "Operações em Aberto",
            f"R$ {df['Ops_Aberto'].iloc[-1]:,.2f}",
            help="Valor total das operações em aberto no mês atual"
        )
    # Charts section
    st.markdown("### Análise Detalhada")
    
    tab1, tab2, tab3 = st.tabs(["📊 Evolução Patrimonial", "💰 Rendimentos", "📈 Performance"])
    
    with tab1:
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.plotly_chart(create_waterfall_chart(df), use_container_width=True)
            
        with col_chart2:
            fig_area = px.area(df, x='Mês', y='Total',
                             title='Evolução Patrimonial (Área)',
                             labels={'Total': 'Patrimônio Total'})
            fig_area.update_layout(
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117',
                font=dict(color='#ffffff')
            )
            st.plotly_chart(fig_area, use_container_width=True)
    
    with tab2:
        col_chart3, col_chart4 = st.columns(2)
        
        with col_chart3:
            rendimentos_df = df.melt(
                id_vars=['Mês'],
                value_vars=['Rendimento_Fixo', 'Rendimento_Variavel'],
                var_name='Tipo',
                value_name='Valor'
            )
            fig_stack = px.bar(rendimentos_df, x='Mês', y='Valor',
                             color='Tipo', title='Composição dos Rendimentos',
                             barmode='stack')
            fig_stack.update_layout(
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117',
                font=dict(color='#ffffff')
            )
            st.plotly_chart(fig_stack, use_container_width=True)
            
        with col_chart4:
            total_fixo = df['Rendimento_Fixo'].sum()
            total_var = df['Rendimento_Variavel'].sum()
            fig_donut = go.Figure(data=[go.Pie(
                labels=['Renda Fixa', 'Renda Variável'],
                values=[total_fixo, total_var],
                hole=.3
            )])
            fig_donut.update_layout(
                title='Distribuição dos Rendimentos',
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117',
                font=dict(color='#ffffff')
            )
            st.plotly_chart(fig_donut, use_container_width=True)
    
    with tab3:
        col_chart5, col_chart6 = st.columns(2)
        
        with col_chart5:
            st.plotly_chart(
                create_performance_gauge(df['Performance_Carteira'].iloc[-1]),
                use_container_width=True
            )
            
        with col_chart6:
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(
                x=df['Mês'],
                y=df['Performance_Carteira'],
                name='Performance Carteira',
                line=dict(color='#00B8D4')
            ))
            fig_comp.add_trace(go.Scatter(
                x=df['Mês'],
                y=df['CDI'],
                name='CDI',
                line=dict(color='#FFB300', dash='dash')
            ))
            fig_comp.update_layout(
                title='Performance vs CDI',
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117',
                font=dict(color='#ffffff')
            )
            st.plotly_chart(fig_comp, use_container_width=True)
    
    st.markdown("### 📋 Operações em Aberto")
    ops_df = pd.DataFrame({
        'Mês': df['Mês'],
        'Valor em Aberto': df['Ops_Aberto']
    })
    
    st.dataframe(
        ops_df.style.format({
            'Valor em Aberto': 'R$ {:,.2f}'
        }),
        hide_index=True,
        use_container_width=True
    )
    
    # Detailed data section with expander
    with st.expander("📋 Dados Detalhados"):
        st.dataframe(
            df.style.format({
                'Total': 'R$ {:,.2f}',
                'CDI': '{:.2f}%',
                'Rendimento_Fixo': 'R$ {:,.2f}',
                'Rendimento_Variavel': 'R$ {:,.2f}',
                'Performance_Carteira': '{:.2f}%',
                'Rendimento_Acumulado': 'R$ {:,.2f}',
                'Retorno_Percentual': '{:.2f}%'
            }),
            use_container_width=True
        )

if __name__ == "__main__":
    main()