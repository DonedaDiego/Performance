import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Set page config
st.set_page_config(page_title="Investment Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    div[data-testid="stHorizontalBlock"] {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Function to load and prepare data
def load_data():
    data = {
        'Mês': ['Jan', 'Fev', 'Mar'],
        'Total': [300000.00, 303360.71],
        'CDI': [1.1, 0.74],
        'Rendimento_Fixo': [0, 1946.69],
        'Rendimento_Variavel': [0, 1414.02],
        'Performance_Carteira': [0.0, 1.12],
        'Rendimento_Acumulado': [0, 3360.71]
    }
    df = pd.DataFrame(data)
    df['Retorno_Percentual'] = df['Total'].pct_change() * 100
    return df

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
        showlegend=False,
        height=400
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
                {'range': [0, max_performance/2], 'color': "lightgray"},
                {'range': [max_performance/2, max_performance], 'color': "gray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': current_performance}},
        title={'text': "Performance Atual (%)"}))
    fig.update_layout(height=250)
    return fig

def main():
    st.title("📈 Dashboard Avançado de Investimentos")
    
    # Load data
    df = load_data()
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Patrimônio Total",
            f"R$ {df['Total'].iloc[-1]:,.2f}",
            f"{df['Retorno_Percentual'].iloc[-1]:.2f}%",
            help="Valor total atual da carteira com variação percentual"
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

    # Charts section
    st.markdown("### Análise Detalhada")
    
    tab1, tab2, tab3 = st.tabs(["📊 Evolução Patrimonial", "💰 Rendimentos", "📈 Performance"])
    
    with tab1:
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Waterfall chart
            st.plotly_chart(create_waterfall_chart(df), use_container_width=True)
            
        with col_chart2:
            # Área chart
            fig_area = px.area(df, x='Mês', y='Total',
                             title='Evolução Patrimonial (Área)',
                             labels={'Total': 'Patrimônio Total'},
                             line_shape='spline')
            st.plotly_chart(fig_area, use_container_width=True)
    
    with tab2:
        col_chart3, col_chart4 = st.columns(2)
        
        with col_chart3:
            # Rendimentos stacked bar
            rendimentos_df = df.melt(
                id_vars=['Mês'],
                value_vars=['Rendimento_Fixo', 'Rendimento_Variavel'],
                var_name='Tipo',
                value_name='Valor'
            )
            fig_stack = px.bar(rendimentos_df, x='Mês', y='Valor',
                             color='Tipo', title='Composição dos Rendimentos',
                             barmode='stack')
            st.plotly_chart(fig_stack, use_container_width=True)
            
        with col_chart4:
            # Donut chart for composition
            total_fixo = df['Rendimento_Fixo'].sum()
            total_var = df['Rendimento_Variavel'].sum()
            fig_donut = go.Figure(data=[go.Pie(
                labels=['Renda Fixa', 'Renda Variável'],
                values=[total_fixo, total_var],
                hole=.3
            )])
            fig_donut.update_layout(title='Distribuição dos Rendimentos')
            st.plotly_chart(fig_donut, use_container_width=True)
    
    with tab3:
        col_chart5, col_chart6 = st.columns(2)
        
        with col_chart5:
            # Performance gauge
            st.plotly_chart(
                create_performance_gauge(df['Performance_Carteira'].iloc[-1]),
                use_container_width=True
            )
            
        with col_chart6:
            # Performance vs CDI
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(
                x=df['Mês'],
                y=df['Performance_Carteira'],
                name='Performance Carteira',
                line=dict(color='blue')
            ))
            fig_comp.add_trace(go.Scatter(
                x=df['Mês'],
                y=df['CDI'],
                name='CDI',
                line=dict(color='red', dash='dash')
            ))
            fig_comp.update_layout(title='Performance vs CDI')
            st.plotly_chart(fig_comp, use_container_width=True)
    
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

    # Footer with additional information
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <p>Dashboard atualizado em tempo real | Dados históricos disponíveis</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()