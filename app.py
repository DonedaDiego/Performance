import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Função auxiliar para formatar em padrão brasileiro
def format_brl(value):
    """Formata um valor para o padrão brasileiro (R$ 1.234,56)."""
    return f"R$ {value:_.2f}".replace(".", ",").replace("_", ".")

# Set page config
st.set_page_config(page_title="Investment Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS para tema escuro
st.markdown(
    """
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
    /* Removida a estilização fixa de cor para os deltas */
    /* div[data-testid="stMetricDelta"] {
        color: #4CAF50 !important;
    } */
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
    """,
    unsafe_allow_html=True
)

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
            'Total': 307390.71,
            'CDI': 1.0,
            'Rendimento_Fixo': 1946.69,
            'Rendimento_Variavel': 5444.02,
            'Performance_Carteira': 1.46,
            'Ops_Aberto': 0,
        },
        'Mar': {
            'Total': 302282.20,
            'CDI': 0.98,
            'Rendimento_Fixo': 1001.09,
            'Rendimento_Variavel': -6109.51,
            'Performance_Carteira': -2.64,
            'Ops_Aberto': 7202.49,
        }
    }
    
    # Criando DataFrame
    df = pd.DataFrame.from_dict(monthly_data, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Mês'}, inplace=True)
    
    # Calculando rendimento acumulado
    df['Rendimento_Acumulado'] = df['Total'] - df['Total'].iloc[0]
    df['Retorno_Percentual'] = df['Total'].pct_change() * 100
    
    return df

def create_waterfall_chart(df):
    # 1) Valor absoluto do primeiro mês
    first_value = df['Total'].iloc[0]

    # 2) Diferenças dos meses seguintes
    diffs = df['Total'].diff().iloc[1:].tolist()

    # Montando a lista final para o Waterfall:
    #   - Primeiro item é absoluto
    #   - Demais são "relative"
    y_values = [first_value] + diffs 
    measures = ["absolute"] + ["relative"] * (len(diffs))

    # Texto que mostra o valor final de cada mês (para exibir lá em cima/outside)
    custom_texts = [f"R$ {val:,.2f}".replace(".", ",").replace(",", ".", 1) 
                    for val in df['Total']]

    fig = go.Figure(go.Waterfall(
        name="Variação Patrimonial",
        orientation="v",
        measure=measures,
        x=df['Mês'],
        text=custom_texts,         # Mostra o valor total daquele mês
        textposition="outside",
        y=y_values,                # Agora vai conter [300k, +7k, -5k, etc...]
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#4CAF50"}},
        decreasing={"marker": {"color": "#F44336"}}
    ))
    fig.update_layout(
        title="Evolução Patrimonial (Waterfall)",
        height=400,
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='#ffffff'),
        margin=dict(l=40, r=40, t=80, b=40),
        )
    fig.update_traces(
        textposition="auto",
        textfont_size=12,
        cliponaxis=False,
        width=0.6
        )

    return fig


def create_performance_gauge(current_performance, max_performance=3):
    # Ajustado para lidar com valores negativos
    min_value = -max_performance if current_performance < 0 else -max_performance/2
    max_value = max_performance
    
    # Definindo cores baseadas no valor atual
    color = "#4CAF50" if current_performance >= 0 else "#F44336"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_performance,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [min_value, max_value]},
            'steps': [
                {'range': [min_value, 0], 'color': "#2c3147"},
                {'range': [0, max_value], 'color': "#1e2130"}
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': current_performance
            }
        },
        title={'text': "Performance Atual (%)", 'font': {'color': '#ffffff'}}
    ))
    fig.update_layout(height=250, paper_bgcolor='#0e1117', font={'color': '#ffffff'})
    return fig

def main():
    st.title("📈 Dashboard de Investimentos")
    
    # Load data
    df = load_data()
    
    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Patrimônio Total",
            format_brl(df['Total'].iloc[-1]),
            f"{df['Retorno_Percentual'].iloc[-1]:.2f}%" if len(df) > 1 else None,
            help="Valor total atual da carteira"
        )
    
    with col2:
        st.metric(
            "Rendimento Acumulado",
            format_brl(df['Rendimento_Acumulado'].iloc[-1]),
            f"{df['Performance_Carteira'].iloc[-1]:.2f}%",
            help="Rendimento total acumulado desde o início"
        )
    
    with col3:
        st.metric(
            "CDI Acumulado",
            f"{df['CDI'].sum():.2f}%",  # Continua em formato de porcentagem
            help="Taxa CDI acumulada no período"
        )
    
    with col4:
        rendimentos_mensais = df['Rendimento_Fixo'] + df['Rendimento_Variavel']
        melhor_rendimento = rendimentos_mensais.max()
        pior_rendimento = rendimentos_mensais.min()
        
        melhor_mes = df.loc[rendimentos_mensais.idxmax(), 'Mês']
        st.metric(
            "Melhor Rendimento Mensal",
            format_brl(melhor_rendimento),
            f"Mês: {melhor_mes}",
            help="Maior rendimento mensal registrado"
        )
    
    with col5:
        delta_ops = None
        if len(df) > 1:
            # Calcula a diferença entre o valor atual e o valor do mês anterior
            diff_val = df['Ops_Aberto'].iloc[-1] - df['Ops_Aberto'].iloc[-2]
            # Formata com sinal + ou -
            delta_ops = f"{'+' if diff_val >= 0 else '-'}{format_brl(abs(diff_val))[3:]}"
        
        st.metric(
            "Operações em Aberto",
            format_brl(df['Ops_Aberto'].iloc[-1]),
            delta_ops,
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
            fig_area = px.area(
                df, 
                x='Mês', 
                y='Total',
                title='Evolução Patrimonial (Área)',
                labels={'Total': 'Patrimônio Total'}
            )
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
            fig_stack = px.bar(
                rendimentos_df, 
                x='Mês', 
                y='Valor',
                color='Tipo', 
                title='Composição dos Rendimentos',
                color_discrete_map={
                    'Rendimento_Fixo': '#4CAF50',
                    'Rendimento_Variavel': '#2196F3'
                }
            )
            
            fig_stack.update_traces(marker_line_width=0)
            fig_stack.update_layout(
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117',
                font=dict(color='#ffffff'),
                barmode='relative'
            )
            st.plotly_chart(fig_stack, use_container_width=True)
            
        with col_chart4:
            total_fixo = df['Rendimento_Fixo'].sum()
            total_var = df['Rendimento_Variavel'].sum()
            
            valores_abs = [abs(total_fixo), abs(total_var)]
            cores = [
                '#4CAF50' if total_fixo >= 0 else '#F44336',
                '#2196F3' if total_var >= 0 else '#F44336'
            ]
            
            fig_donut = go.Figure(data=[go.Pie(
                labels=['Renda Fixa', 'Renda Variável'],
                values=valores_abs,
                hole=.3,
                marker_colors=cores,
                textinfo='label+percent',
                hoverinfo='label+value'
            )])
            
            fig_donut.add_annotation(
                text=f"RF: {format_brl(total_fixo)}<br>RV: {format_brl(total_var)}",
                x=0.5, y=0.5,
                font_size=10,
                showarrow=False
            )
            
            fig_donut.update_layout(
                title='Distribuição dos Rendimentos (valores absolutos)',
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
            
            # Adicionando linha de zero para melhor visualização
            fig_comp.add_shape(
                type="line",
                x0=df['Mês'].iloc[0],
                y0=0,
                x1=df['Mês'].iloc[-1],
                y1=0,
                line=dict(color="gray", width=1, dash="dot")
            )
            
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
    
    # Aqui usamos a função na formatação do dataframe
    st.dataframe(
        ops_df.style.format({
            'Valor em Aberto': lambda x: format_brl(x)
        }),
        hide_index=True,
        use_container_width=True
    )
    
    # Detailed data section with expander
    with st.expander("📋 Dados Detalhados"):
        st.dataframe(
            df.style.format({
                'Total': lambda x: format_brl(x),
                'CDI': '{:.2f}%',
                'Rendimento_Fixo': lambda x: format_brl(x),
                'Rendimento_Variavel': lambda x: format_brl(x),
                'Performance_Carteira': '{:.2f}%',
                'Rendimento_Acumulado': lambda x: format_brl(x),
                'Retorno_Percentual': '{:.2f}%'
            }),
            use_container_width=True
        )

if __name__ == "__main__":
    main()
