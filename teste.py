import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import numpy as np
from datetime import datetime
from st_aggrid import AgGrid, GridOptionsBuilder
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
import yfinance as yf
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

# Função para buscar dados do Ibovespa
def get_ibov_data():
    try:
        ibov = yf.download('^BVSP', start='2024-01-01', end='2025-12-31')
        ibov_returns = ibov['Close'].pct_change()
        return ibov_returns
    except Exception as e:
        st.error(f"Erro ao carregar dados do Ibovespa: {e}")
        return None


def load_data(force_update=False):
    if not force_update:
        # Use cache apenas quando não forçar atualização
        @st.cache_data
        def cached_load():
            return fetch_data_from_sheets()
        return cached_load()
    else:
        # Busca direta sem cache quando forçar atualização
        return fetch_data_from_sheets()

def fetch_data_from_sheets():
    SPREADSHEET_ID = '1qEQoNFQPKP64-DXOvNSquKXy8B36Og4i5YSB-LXS5dQ'
    RANGE_NAME = 'dash!A:N'  # Modificado para pegar todas as linhas da coluna A até N
    API_KEY = 'AIzaSyBh1_W5gtZdJlQdNbA7bUrSsmzMPSQIfVE'

    try:
        service = build('sheets', 'v4', developerKey=API_KEY)
        
        # Primeiro, vamos verificar quantas linhas têm dados
        result = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=RANGE_NAME
        ).execute()

        values = result.get('values', [])
        
        if not values:
            st.error("Nenhum dado encontrado na planilha")
            return None

        print("Valores brutos do Google Sheets:", values)
        print("Número de linhas encontradas:", len(values))

        # Ajuste para normalizar as linhas
        max_cols = len(values[0])  # Número máximo de colunas baseado no cabeçalho
        normalized_values = [
            row + [''] * (max_cols - len(row))  # Preenche linhas com colunas faltantes
            for row in values
        ]

        # Cria o DataFrame com os dados normalizados
        df = pd.DataFrame(normalized_values[1:], columns=normalized_values[0])

        # Converte colunas numéricas
        numeric_columns = df.columns[1:]  # Ignorar a primeira coluna "Status"
        for col in numeric_columns:
            if df[col].dtype == 'object':  # Apenas processar colunas que sejam strings
                df[col] = (
                    df[col]
                    .str.replace('.', '', regex=False)  # Remove separadores de milhar
                    .str.replace(',', '.', regex=False)  # Substitui vírgulas por pontos decimais
                )
                df[col] = pd.to_numeric(df[col], errors='coerce')

        print("\nDataFrame após normalização:")
        print(df)
        return df

    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        print(f"Erro detalhado: {e}")
        return None

# Adicione esta função para limpar o cache quando necessário
def clear_cache():
    st.cache_data.clear()

df = load_data()
months = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']

# Processamento dos dados
resultado_row = df.loc[df['Status'] == 'Resultado'].iloc[0, 1:].values * 100
cdi_row = df.loc[df['Status'] == 'Cdi'].iloc[0, 1:].values * 100
excedente_row = df.loc[df['Status'] == 'Exedente'].iloc[0, 1:].values * 100
desconto_row = df.loc[df['Status'] == 'Desconto CDI'].iloc[0, 1:].values
gemini_row = df.loc[df['Status'] == 'Geminiii'].iloc[0, 1:].values
liquido_row = df.loc[df['Status'] == 'Liquido'].iloc[0, 1:].values

capital_row = df.loc[df['Status'] == 'Capital'].iloc[0, 1:].values
cdi_row = df.loc[df['Status'] == 'Cdi'].iloc[0, 1:].values
retornos_mensais = pd.Series(capital_row).pct_change() * 100

def calculate_metrics(capital_values, cdi_values):
    # Retorno total baseado no capital
    retorno_total = ((capital_values[-1] / capital_values[0]) - 1) * 100
    
    # CDI acumulado do período
    cdi_acumulado = np.prod(1 + cdi_values[~np.isnan(cdi_values)]) - 1
    cdi_periodo = cdi_acumulado * 100
    
    # Último retorno mensal
    retorno_mensal = ((capital_values[-1] / capital_values[-2]) - 1) * 100 if len(capital_values) > 1 else 0
    
    return retorno_total, cdi_periodo, retorno_mensal

# Sidebar com controles
st.sidebar.title("Configurações")
chart_type = st.sidebar.selectbox(
    "Tipo de Visualização",
    ["Retorno Acumulado", "Retorno Mensal", "Comparação", "Fluxo de Caixa"]
)

if st.sidebar.button("🔄 Atualizar Base de Dados"):
    with st.spinner('Atualizando dados...'):
        clear_cache()  # Limpa o cache antes de atualizar
        st.session_state.data = load_data(force_update=True)
        st.success('Dados atualizados com sucesso!')
        st.rerun()

# Configuração do Período de Análise
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
for i in reversed(range(len(capital_row))):
    if not np.isnan(capital_row[i]):
        last_valid_idx = i
        break

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    if last_valid_idx >= 0:
        retorno_total = ((capital_row[last_valid_idx] / capital_row[0]) - 1) * 100
        retorno_mensal = ((capital_row[last_valid_idx] / capital_row[max(0, last_valid_idx-1)]) - 1) * 100
        st.metric("Retorno Total", 
                 f"{retorno_total:.1f}%", 
                 f"{retorno_mensal:+.1f}% último mês")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    if last_valid_idx >= 0:
        # CDI acumulado usando pandas para tratamento de NaN
        cdi_series = pd.Series(cdi_row[:last_valid_idx+1])
        cdi_acumulado = np.prod(1 + cdi_series.dropna().values) - 1
        cdi_periodo = cdi_acumulado * 100
        
        st.metric("CDI do Acumulado", 
                 f"{cdi_periodo:.1f}%",
                 f"{(retorno_total - cdi_periodo):+.1f}% vs CDI")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    if last_valid_idx >= 0:
        st.metric("Patrimônio", 
                 f"R$ {capital_row[last_valid_idx]:_.2f}".replace(".", ",").replace("_", "."),
                 f"{((capital_row[last_valid_idx] / capital_inicial) - 1) * 100:+.2f}% desde o início")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
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
    
    colors = ['#4B9CD3' if x >= 0 else '#ff4444' for x in retornos_mensais]
    
    fig.add_trace(go.Bar(
        x=months[date_range[0]:date_range[1]+1],
        y=retornos_mensais[date_range[0]:date_range[1]+1],
        name='Retorno',
        marker_color=colors
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
        marker_color=np.where(pd.notnull(excedente_row[date_range[0]:date_range[1]+1]) & 
                            (excedente_row[date_range[0]:date_range[1]+1] >= 0), '#4B9CD3', '#ff4444')
    ))
    
    fig.update_layout(
        title="Excedente ao CDI",
        template='plotly_dark',
        height=500
    )

else:  # Fluxo de Caixa
    fig = go.Figure()
    
    cores_gemini = ['#4B9CD3' if x >= 0 else '#ff4444' for x in gemini_row]
    fig.add_trace(go.Bar(
        x=months[date_range[0]:date_range[1]+1],
        y=gemini_row[date_range[0]:date_range[1]+1],
        name='Gemini',
        marker_color=cores_gemini,
        text=[f"R$ {x:,.2f}" for x in gemini_row[date_range[0]:date_range[1]+1]],
        textposition='outside'
    ))
    
    cores_desconto = ['#FFA500' if x >= 0 else '#ff6b6b' for x in desconto_row]
    fig.add_trace(go.Bar(
        x=months[date_range[0]:date_range[1]+1],
        y=desconto_row[date_range[0]:date_range[1]+1],
        name='Desconto CDI',
        marker_color=cores_desconto,
        text=[f"R$ {x:,.2f}" for x in desconto_row[date_range[0]:date_range[1]+1]],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Fluxo de Caixa",
        template='plotly_dark',
        height=500,
        barmode='group',
        yaxis=dict(
            title="Valor (R$)",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='white',
            range=[
                min(min(gemini_row), min(desconto_row), 0) * 1.1,
                max(max(gemini_row), max(desconto_row), 0) * 1.1
            ]
        )
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
        melhor_mes = max(resultado_row)
        mes_melhor = months[np.argmax(resultado_row)]
        st.metric("Melhor Mês", 
                 f"{melhor_mes:+.1f}%",
                 mes_melhor,
                 delta_color="normal")
    
    with col2:
        pior_mes = min(resultado_row)
        mes_pior = months[np.argmin(resultado_row)]
        st.metric("Pior Mês",
                 f"{pior_mes:+.1f}%",
                 mes_pior,
                 delta_color="inverse" if pior_mes < 0 else "normal")
    
    with col3:
        valid_data = pd.Series(resultado_row) > cdi_row
        meses_acima_cdi = valid_data[~pd.isna(valid_data)].sum()
        total_meses = sum(~pd.isna(valid_data))
        st.metric("Meses acima do CDI",
                 f"{meses_acima_cdi}/{total_meses}",
                 f"{(meses_acima_cdi/total_meses*100):.1f}% do período")


st.markdown("### 📈 Indicadores Detalhados")
kpi_tabs = st.tabs(["Risco vs Retorno", "Comportamento", "Comparativo de Mercado"])

# Na seção onde estão os cálculos de volatilidade e drawdown, ajuste:
with kpi_tabs[0]:
    col1, col2 = st.columns(2)
    
    with col1:
        # Volatilidade baseada nos retornos reais do capital
        rolling_vol = pd.Series(retornos_mensais/100).rolling(3).std() * np.sqrt(12) * 100
        
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=months,
            y=rolling_vol,
            fill='tozeroy',
            line=dict(color='#4B9CD3', width=2),
            name='Volatilidade'
        ))
        fig_vol.update_layout(
            title="Volatilidade Móvel (3 meses)",
            template='plotly_dark',
            height=300,
            yaxis_title="Volatilidade Anualizada (%)"
        )
        st.plotly_chart(fig_vol, use_container_width=True)
        
    with col2:
        # Drawdown baseado no capital
        capital_series = pd.Series(capital_row)
        capital_max = capital_series.cummax()
        drawdown = ((capital_series - capital_max) / capital_max) * 100
        
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=months,
            y=drawdown,
            fill='tozeroy',
            line=dict(color='#ff9f43', width=2),
            name='Drawdown'
        ))
        fig_dd.update_layout(
            title="Drawdown Histórico",
            template='plotly_dark',
            height=300,
            yaxis_title="Drawdown (%)"
        )
        st.plotly_chart(fig_dd, use_container_width=True)

with kpi_tabs[1]:
    col1, col2 = st.columns(2)
    
    with col1:
        # Consistência de Resultados
        meses_positivos = sum(resultado_row > 0)
        total_meses = len(resultado_row)
        
        fig_consistency = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = (meses_positivos/total_meses) * 100,
            title = {'text': "Meses com Resultado Positivo"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#4B9CD3"},
                'steps': [
                    {'range': [0, 50], 'color': '#ff6b6b'},
                    {'range': [50, 75], 'color': '#ffd93d'},
                    {'range': [75, 100], 'color': '#6BCB77'}
                ]
            }
        ))
        
        fig_consistency.update_layout(
            template='plotly_dark',
            height=300
        )
        st.plotly_chart(fig_consistency, use_container_width=True)
        
    with col2:
        fig_capital = go.Figure()
        fig_capital.add_trace(go.Scatter(
            x=months,
            y=capital_row,
            fill='tozeroy',
            name='Capital Total',
            line=dict(color='#4B9CD3', width=2)
        ))
        
        fig_capital.add_hline(
            y=capital_inicial,
            line_dash="dash",
            line_color="#FFA500",
            annotation_text="Capital Inicial"
        )
        
        fig_capital.update_layout(
            title="Evolução do Capital",
            template='plotly_dark',
            height=300,
            hovermode='x unified',
            yaxis_title="Capital (R$)",
            yaxis=dict(
                tickformat=",.0f",
                tickprefix="R$ "
            )
        )
        st.plotly_chart(fig_capital, use_container_width=True)

with kpi_tabs[2]:
    col1, col2 = st.columns(2)
    
    with col1:
        fig_retorno = go.Figure()
        
        # Cores diferentes para valores positivos e negativos no retorno realizado
        cores_retorno = ['#4B9CD3' if x >= 0 else '#ff4444' for x in resultado_row]
        
        fig_retorno.add_trace(go.Bar(
            x=months,
            y=resultado_row,
            name='Retorno Realizado',
            marker_color=cores_retorno,
            text=[f"{x:.1f}%" for x in resultado_row],
            textposition='outside'
        ))
        
        fig_retorno.add_trace(go.Scatter(
            x=months,
            y=cdi_row,
            name='Meta (CDI)',
            line=dict(color='#FFA500', width=2, dash='dash')
        ))
        
        fig_retorno.update_layout(
            title="Retorno vs Meta Mensal",
            template='plotly_dark',
            height=300,
            hovermode='x unified',
            yaxis=dict(
                title="Retorno (%)",
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='white',
                range=[min(min(resultado_row), 0) * 1.1, max(max(resultado_row), 0) * 1.1]
            )
        )
        st.plotly_chart(fig_retorno, use_container_width=True)
    
    with col2:
        # Calculando o último retorno acumulado válido usando pandas
        cum_resultado_series = pd.Series(cum_resultado)
        cum_cdi_series = pd.Series(cum_cdi)
        
        last_valid_resultado = cum_resultado_series.iloc[-1] if not pd.isna(cum_resultado_series.iloc[-1]) else 0
        last_valid_cdi = cum_cdi_series.iloc[-1] if not pd.isna(cum_cdi_series.iloc[-1]) else 0
        
        rent_comparison = pd.DataFrame({
            'Investimento': ['Sua Carteira', 'CDI', 'Poupança'],
            'Rentabilidade': [
                last_valid_resultado,
                last_valid_cdi,
                last_valid_cdi * 0.7  # Aproximação da poupança
            ]
        })
        
        fig_rent = go.Figure(go.Bar(
            x=rent_comparison['Investimento'],
            y=rent_comparison['Rentabilidade'],
            text=[f"{val:.2f}%" for val in rent_comparison['Rentabilidade']],
            textposition='auto',
            marker_color=['#4B9CD3', '#FFA500', '#ff6b6b']
        ))
        
        fig_rent.update_layout(
            title="Comparativo de Rentabilidade",
            template='plotly_dark',
            height=300,
            showlegend=False,
            yaxis_title="Rentabilidade Total (%)",
            yaxis=dict(
                tickformat='.2f',
                ticksuffix='%'
            ),
            # Ajustando os limites do eixo Y para sempre mostrar um pouco acima do valor máximo
            yaxis_range=[0, max(rent_comparison['Rentabilidade']) * 1.2] if max(rent_comparison['Rentabilidade']) > 0 else [0, 1]
        )
        st.plotly_chart(fig_rent, use_container_width=True)

# Adicionar explicações para os indicadores
with st.expander("ℹ️ Entenda os Indicadores"):
    st.write("""
    **Volatilidade Móvel:** Mede a variação dos retornos nos últimos 3 meses. Quanto menor, mais estável é seu investimento.
    
    **Drawdown:** Representa a queda do investimento em relação ao seu ponto mais alto. Quanto menor o drawdown, menor o risco de perda.
    
    **Meses com Resultado Positivo:** Mostra a consistência dos seus resultados, indicando quantos meses você teve retorno positivo.
    
    **Evolução do Capital:** Acompanha o crescimento do seu patrimônio ao longo do tempo.
    
    **Retorno vs Meta:** Compara seu retorno mensal com o CDI (meta).
    
    **Comparativo de Rentabilidade:** Mostra como seu investimento se compara com outras opções do mercado.
    """)								