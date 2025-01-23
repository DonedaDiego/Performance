import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import numpy as np
from datetime import datetime
from st_aggrid import AgGrid, GridOptionsBuilder
import openpyxl
import xlrd
import xlsxwriter

st.set_page_config(layout="wide", page_title="Dashboard Financeiro", page_icon="📊")

st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117 !important;
    }
    .css-1kyxreq, .css-1544g2n {
        background-color: #262730 !important;
    }
    .st-emotion-cache-16idsys p {
        font-size: 14px;
    }
    div[data-testid="stMetric"] {
        background-color: #262730;
        padding: 15px;
        border-radius: 5px;
    }
    .st-emotion-cache-1wivap2 {
        background-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_excel("base.xlsx")
    except:
        df = pd.read_csv("base.csv")
    return df

df = load_data()
months = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']

# Processamento dos dados
resultado_row = df.loc[df['Status'] == 'Resultado'].iloc[0, 1:].values * 100
cdi_row = df.loc[df['Status'] == 'Cdi'].iloc[0, 1:].values * 100
excedente_row = df.loc[df['Status'] == 'Exedente'].iloc[0, 1:].values * 100
desconto_row = df.loc[df['Status'] == 'Desconto CDI'].iloc[0, 1:].values
gemini_row = df.loc[df['Status'] == 'Geminiii'].iloc[0, 1:].values
liquido_row = df.loc[df['Status'] == 'Liquido'].iloc[0, 1:].values

# Sidebar com controles
st.sidebar.title("Configurações")
chart_type = st.sidebar.selectbox(
    "Tipo de Visualização",
    ["Retorno Acumulado", "Retorno Mensal", "Comparação", "Fluxo de Caixa"]
)

date_range = st.sidebar.select_slider(
    "Período de Análise",
    options=range(len(months)),
    value=(0, len(months)-1),
    format_func=lambda x: months[x]
)

df_performance = pd.DataFrame({
    'Mês': months[date_range[0]:date_range[1]+1],
    'Carteira (%)': pd.Series(resultado_row[date_range[0]:date_range[1]+1]).round(2),
    'CDI (%)': pd.Series(cdi_row[date_range[0]:date_range[1]+1]).round(2),
    'Excedente (%)': pd.Series(excedente_row[date_range[0]:date_range[1]+1]).round(2)
})

df_fluxo = pd.DataFrame({
    'Mês': months[date_range[0]:date_range[1]+1],
    'Desconto CDI': pd.Series(desconto_row[date_range[0]:date_range[1]+1]),
    'Gemini': pd.Series(gemini_row[date_range[0]:date_range[1]+1]),
    'Líquido': pd.Series(liquido_row[date_range[0]:date_range[1]+1])
})

for col in ['Desconto CDI', 'Gemini', 'Líquido']:
    df_fluxo[col] = df_fluxo[col].apply(lambda x: f"R$ {x:_.2f}".replace(".", ",").replace("_", "."))

# Filtre apenas linhas com dados
df_performance = df_performance.dropna()
df_fluxo = df_fluxo.dropna()



# Header com métricas principais
st.title("📊 Dashboard Financeiro")

capital_inicial = 300000  # Capital inicial de 300k

# Atualizar a seção de métricas
col1, col2, col3, col4, col5 = st.columns(5)


cum_resultado = np.cumsum(resultado_row)
cum_cdi = np.cumsum(cdi_row)

# Encontrar último índice válido
last_valid_idx = -1
for i in reversed(range(len(resultado_row))):
    if not np.isnan(resultado_row[i]):
        last_valid_idx = i
        break

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    if last_valid_idx >= 0:
        st.metric("Retorno Total", 
                 f"{cum_resultado[last_valid_idx]:.1f}%", 
                 f"{resultado_row[last_valid_idx]:+.1f}% último mês")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    if last_valid_idx >= 0:
        cdi_mes = cdi_row[last_valid_idx]  # CDI do mês (0.68%)
        excedente_mensal = excedente_row[last_valid_idx]  # Excedente do mês (9.32%)
        percentual_excedente = (excedente_mensal / cdi_mes * 100)  # Aproximadamente 1370%
        
        st.metric("vs CDI", 
                 f"{(cum_resultado[last_valid_idx] - cum_cdi[last_valid_idx]):.1f}%",
                 f"{percentual_excedente:_.2f}".replace(".", ",").replace("_", ".") + "% acima do CDI no mês")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    if last_valid_idx >= 0:
        st.metric("Patrimônio Líquido", 
                 f"R$ {liquido_row[last_valid_idx]:_.2f}".replace(".", ",").replace("_", "."),
                 f"R$ {(liquido_row[last_valid_idx] - liquido_row[max(0, last_valid_idx-1)]):+,.0f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    if last_valid_idx >= 0:
        patrimonio_total = capital_inicial + liquido_row[last_valid_idx]
        evolucao_capital = (patrimonio_total / capital_inicial - 1) * 100
        st.metric("Patrimônio Total", 
                 f"R$ {patrimonio_total:_.2f}".replace(".", ",").replace("_", "."),
                 f"{evolucao_capital:+.2f}% desde o início")
    st.markdown('</div>', unsafe_allow_html=True)

with col5:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    if last_valid_idx >= 0:
        st.metric("Capital Base", 
                 f"R$ {capital_inicial:_.2f}".replace(".", ",").replace("_", "."),
                 "Capital inicial")
    st.markdown('</div>', unsafe_allow_html=True)

# Gráficos principais baseados na seleção
if chart_type == "Retorno Acumulado":
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=months[date_range[0]:date_range[1]+1],
        y=cum_resultado[date_range[0]:date_range[1]+1],
        name='Carteira',
        line=dict(color='#4B9CD3', width=3),
        fill='tonexty',
        fillcolor='rgba(75,156,211,0.2)'
    ))
    
    fig.add_trace(go.Scatter(
        x=months[date_range[0]:date_range[1]+1],
        y=cum_cdi[date_range[0]:date_range[1]+1],
        name='CDI',
        line=dict(color='#FFA500', width=3),
        fill='tonexty',
        fillcolor='rgba(255,165,0,0.1)'
    ))
    
    fig.update_layout(
        title="Retorno Acumulado",
        template='plotly_dark',
        height=500
    )
    
elif chart_type == "Retorno Mensal":
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=months[date_range[0]:date_range[1]+1],
        y=resultado_row[date_range[0]:date_range[1]+1],
        name='Retorno',
        marker_color='#4B9CD3'
    ))
    
    fig.add_trace(go.Scatter(
        x=months[date_range[0]:date_range[1]+1],
        y=cdi_row[date_range[0]:date_range[1]+1],
        name='CDI',
        line=dict(color='#FFA500', width=3)
    ))
    
    fig.update_layout(
        title="Retorno Mensal",
        template='plotly_dark',
        height=500
    )

elif chart_type == "Comparação":
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=months[date_range[0]:date_range[1]+1],
        y=excedente_row[date_range[0]:date_range[1]+1],
        name='Excedente ao CDI',
        marker_color=np.where(excedente_row[date_range[0]:date_range[1]+1] >= 0, '#4B9CD3', '#ff4444')
    ))
    
    fig.update_layout(
        title="Excedente ao CDI",
        template='plotly_dark',
        height=500
    )

else:  # Fluxo de Caixa
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=months[date_range[0]:date_range[1]+1],
        y=gemini_row[date_range[0]:date_range[1]+1],
        name='Gemini',
        marker_color='#4B9CD3'
    ))
    
    fig.add_trace(go.Bar(
        x=months[date_range[0]:date_range[1]+1],
        y=desconto_row[date_range[0]:date_range[1]+1],
        name='Desconto CDI',
        marker_color='#FFA500'
    ))
    
    fig.update_layout(
        title="Fluxo de Caixa",
        template='plotly_dark',
        height=500,
        barmode='group'
    )

st.plotly_chart(fig, use_container_width=True)

def config_grid(df):
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(min_column_width=100)
    gb.configure_grid_options(domLayout='normal')
    return gb.build()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Performance Detalhada")
    grid_options = config_grid(df_performance)
    AgGrid(
        df_performance,
        gridOptions=grid_options,
        theme='streamlit',
        fit_columns_on_grid_load=True
    )

with col2:
    st.markdown("### Fluxo Financeiro")
    grid_options = config_grid(df_fluxo)
    AgGrid(
        df_fluxo,
        gridOptions=grid_options,
        theme='streamlit',
        fit_columns_on_grid_load=True
    )

# Análise estatística
with st.expander("📊 Análise Estatística"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Melhor Mês", 
                 f"{max(resultado_row[date_range[0]:date_range[1]+1]):.1f}%",
                 months[np.argmax(resultado_row[date_range[0]:date_range[1]+1])])
    
    with col2:
        st.metric("Pior Mês",
                 f"{min(resultado_row[date_range[0]:date_range[1]+1]):.1f}%",
                 months[np.argmin(resultado_row[date_range[0]:date_range[1]+1])])
    
    with col3:
        meses_acima_cdi = sum(excedente_row[date_range[0]:date_range[1]+1] > 0)
        total_meses = date_range[1] - date_range[0] + 1
        st.metric("Meses acima do CDI",
                 f"{meses_acima_cdi}/{total_meses}",
                 f"{(meses_acima_cdi/total_meses*100):.1f}% do período")

# Adiciona seção de KPIs detalhados
st.markdown("### 📈 Indicadores Detalhados")
kpi_tabs = st.tabs(["Volatilidade", "Correlações", "Distribuição"])

with kpi_tabs[0]:
    col1, col2 = st.columns(2)
    
    with col1:
        # Cálculo de volatilidade móvel
        rolling_vol = pd.Series(resultado_row).rolling(3).std() * np.sqrt(12)
        
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=months,
            y=rolling_vol,
            fill='tozeroy',
            line=dict(color='#ff6b6b'),
            name='Volatilidade'
        ))
        fig_vol.update_layout(
            title="Volatilidade Móvel (3 meses)",
            template='plotly_dark',
            height=300
        )
        st.plotly_chart(fig_vol, use_container_width=True)
        
    with col2:
        # Drawdown
        cum_max = pd.Series(cum_resultado).cummax()
        drawdown = (pd.Series(cum_resultado) - cum_max) / cum_max * 100
        
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=months,
            y=drawdown,
            fill='tozeroy',
            line=dict(color='#ff9f43'),
            name='Drawdown'
        ))
        fig_dd.update_layout(
            title="Drawdown Histórico",
            template='plotly_dark',
            height=300
        )
        st.plotly_chart(fig_dd, use_container_width=True)

with kpi_tabs[1]:
    # Matriz de correlação
    data_corr = pd.DataFrame({
    'Carteira': pd.Series(resultado_row),
    'CDI': pd.Series(cdi_row),
    'Gemini': pd.Series(gemini_row),
    'Desconto': pd.Series(desconto_row)
    })
    
    corr_matrix = data_corr.corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    fig_corr.update_layout(
        title="Matriz de Correlação",
        template='plotly_dark',
        height=400
    )
    st.plotly_chart(fig_corr, use_container_width=True)

with kpi_tabs[2]:
    col1, col2 = st.columns(2)
    
    with col1:
        # Histograma de retornos
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=resultado_row,
            nbinsx=10,
            name='Retornos',
            marker_color='#4B9CD3'
        ))
        fig_hist.update_layout(
            title="Distribuição dos Retornos",
            template='plotly_dark',
            height=300
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Box plot comparativo
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=resultado_row,
            name='Carteira',
            marker_color='#4B9CD3'
        ))
        fig_box.add_trace(go.Box(
            y=cdi_row,
            name='CDI',
            marker_color='#FFA500'
        ))
        fig_box.update_layout(
            title="Comparação de Distribuições",
            template='plotly_dark',
            height=300
        )
        st.plotly_chart(fig_box, use_container_width=True)


# Seção de Análise Preditiva
st.markdown("### 🎯 Projeções")
proj_col1, proj_col2 = st.columns([2,1])


with proj_col1:
    # Criando os arrays corretamente
    x = np.array(range(12))  # array fixo de 12 meses
    y = np.array(resultado_row, dtype=float)  # converter para float
    
    # Calcular regressão
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Criar linha de tendência
    trend_line = slope * x + intercept
    
    # Projetar próximos meses
    future_x = np.array(range(12, 15))  # próximos 3 meses
    future_trend = slope * future_x + intercept
    
    # Criar gráfico
    fig_proj = go.Figure()
    
    # Dados reais
    fig_proj.add_trace(go.Scatter(
        x=months,
        y=y,
        name='Retornos Reais',
        mode='markers+lines',
        marker=dict(size=8)
    ))
    
    # Linha de tendência + projeção
    fig_proj.add_trace(go.Scatter(
        x=months + ['Proj 1', 'Proj 2', 'Proj 3'],
        y=np.concatenate([trend_line, future_trend]),
        name='Tendência',
        line=dict(dash='dash')
    ))
    
    fig_proj.update_layout(
        title="Tendência e Projeção",
        template='plotly_dark',
        height=400
    )
    
    st.plotly_chart(fig_proj, use_container_width=True)

    with proj_col2:
        st.markdown("#### Métricas de Tendência")
        st.metric("Inclinação", f"{slope:.4f}")
        st.metric("R²", f"{r_value**2:.4f}")
        st.metric("Projeção 3 meses", f"{future_trend[-1]:.2f}%")

# Seção de Alertas e Recomendações
st.markdown("### ⚠️ Alertas e Recomendações")
alert_cols = st.columns(3)

with alert_cols[0]:
    if np.mean(resultado_row[-3:]) < np.mean(cdi_row[-3:]):
        st.warning("⚠️ Retorno abaixo do CDI nos últimos 3 meses")
    else:
        st.success("✅ Retorno acima do CDI mantido")

with alert_cols[1]:
    vol_recente = rolling_vol.iloc[-1]
    if vol_recente > rolling_vol.mean():
        st.warning(f"⚠️ Volatilidade elevada: {vol_recente:.2f}%")
    else:
        st.success("✅ Volatilidade controlada")

with alert_cols[2]:
    drawdown_atual = drawdown.iloc[-1]
    if drawdown_atual < -5:
        st.warning(f"⚠️ Drawdown significativo: {drawdown_atual:.2f}%")
    else:
        st.success("✅ Drawdown controlado")

# # Exportação de dados
# st.markdown("### 📥 Exportar Dados")
# export_cols = st.columns(3)

# with export_cols[0]:
#     # Preparar dados para CSV
#     export_df = pd.DataFrame({
#         'Mês': months,
#         'Retorno (%)': resultado_row,
#         'CDI (%)': cdi_row,
#         'Excedente (%)': excedente_row,
#         'Patrimônio': liquido_row
#     })
    
#     csv = export_df.to_csv(index=False).encode('utf-8')
#     st.download_button(
#         "📥 Download CSV",
#         csv,
#         "performance_data.csv",
#         "text/csv",
#         key='download-csv'
#     )

# with export_cols[1]:
#     import io
#     import xlsxwriter
    
#     # Preparar Excel
#     buffer = io.BytesIO()
#     with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
#         export_df.to_excel(writer, sheet_name='Dados', index=False)
    
#     st.download_button(
#         "📥 Download Excel",
#         buffer.getvalue(),
#         "performance_data.xlsx",
#         "application/vnd.ms-excel",
#         key='download-excel'
#     )

# # with export_cols[2]:
# #     # Gerar relatório PDF
#     st.button("📥 Gerar Relatório PDF", key='generate-pdf')        