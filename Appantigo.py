import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from st_aggrid import AgGrid, GridOptionsBuilder
from googleapiclient.discovery import build
import yfinance as yf

# =========================
# 1. Configurações Iniciais
# =========================

st.set_page_config(layout="wide", page_title="Dashboard Financeiro", page_icon="📊")
st.markdown("""
    <style>
    .stApp { background-color: #0e1117 !important; }
    .css-1kyxreq, .css-1544g2n { background-color: #262730 !important; }
    .st-emotion-cache-16idsys p { font-size: 14px; }
    div[data-testid="stMetric"] { background-color: #262730; padding: 15px; border-radius: 5px; }
    .st-emotion-cache-1wivap2 { background-color: transparent; }
    </style>
""", unsafe_allow_html=True)

# ======================
# 2. Funções Auxiliares
# ======================

def get_ibov_data():
    """Busca dados do Ibovespa (não usado no código principal)."""
    try:
        ibov = yf.download('^BVSP', start='2024-01-01', end='2025-12-31')
        return ibov['Close'].pct_change()
    except Exception as e:
        st.error(f"Erro ao carregar dados do Ibovespa: {e}")
        return None

@st.cache_data
def fetch_data_from_sheets():
    """Busca dados do Google Sheets e converte colunas numéricas."""
    SPREADSHEET_ID = '1qEQoNFQPKP64-DXOvNSquKXy8B36Og4i5YSB-LXS5dQ'
    RANGE_NAME = 'dash!A:N'
    API_KEY = 'AIzaSyBh1_W5gtZdJlQdNbA7bUrSsmzMPSQIfVE'
    try:
        service = build('sheets', 'v4', developerKey=API_KEY)
        result = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME
        ).execute()
        values = result.get('values', [])
        if not values:
            st.error("Nenhum dado encontrado na planilha")
            return None
        
        # Normaliza linhas
        max_cols = len(values[0])
        normalized = [row + [''] * (max_cols - len(row)) for row in values]
        df = pd.DataFrame(normalized[1:], columns=normalized[0])
        
        # Converte colunas numéricas
        for col in df.columns[1:]:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(
                    df[col].str.replace('.', '', regex=False)
                           .str.replace(',', '.', regex=False),
                    errors='coerce'
                )
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

def load_data(force_update=False):
    """Carrega dados (cache)."""
    return fetch_data_from_sheets() if not force_update else fetch_data_from_sheets()

def clear_cache():
    """Limpa cache do Streamlit."""
    st.cache_data.clear()

def get_data_array(df, status_label, multiply=1):
    """
    Retorna os valores (a partir da coluna 1) da linha cujo 'Status' == status_label
    sempre como um array NumPy. Se não houver dados, retorna array vazio.
    multiply: fator multiplicador (ex: 100 para transformar em %).
    """
    subset = df.loc[df['Status'] == status_label]
    if subset.empty:
        return np.array([])  # Sem linha correspondente
    
    # Pega apenas colunas a partir da 2ª (index 1)
    row = subset.iloc[0, 1:]
    if row.shape[0] == 0:
        return np.array([])  # Nenhuma coluna
    
    # Converte para numérico e multiplica
    row = pd.to_numeric(row, errors='coerce') * multiply
    
    # Converte Series em array
    arr = row.to_numpy()
    # Se for tudo NaN ou só 1 valor, continua sendo array. Está OK.
    return arr

def calculate_metrics(capital_values, cdi_values):
    """Exemplo de função auxiliar."""
    retorno_total = ((capital_values[-1] / capital_values[0]) - 1) * 100
    cdi_acumulado = np.prod(1 + cdi_values[~np.isnan(cdi_values)]) - 1
    retorno_mensal = ((capital_values[-1] / capital_values[-2]) - 1) * 100 if len(capital_values) > 1 else 0
    return retorno_total, cdi_acumulado * 100, retorno_mensal

# =========================
# 3. Carregamento do DataFrame
# =========================

df = load_data()
months = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']

# Se o DF for None (planilha vazia ou erro), interrompe
if df is None:
    st.stop()

# Carrega cada linha como array
resultado_row   = get_data_array(df, 'Resultado',   multiply=100)
cdi_row         = get_data_array(df, 'Cdi',         multiply=100)
excedente_row   = get_data_array(df, 'Exedente',    multiply=100)
desconto_row    = get_data_array(df, 'Desconto CDI')
gemini_row      = get_data_array(df, 'Geminiii')
liquido_row     = get_data_array(df, 'Liquido')
capital_row     = get_data_array(df, 'Capital')

# Se não houver capital, não dá pra calcular retornos
if capital_row.size < 1:
    st.warning("Não há dados de 'Capital' na planilha. O app não pode prosseguir.")
    st.stop()
else:
    # Calcula retornos mensais se tiver pelo menos 2 pontos
    if capital_row.size > 1:
        retornos_mensais = pd.Series(capital_row).pct_change() * 100
    else:
        retornos_mensais = pd.Series([np.nan])  # Sem variação

# Acumulados (cumsum) - se o array estiver vazio, evitamos erro
cum_resultado = np.cumsum(resultado_row) if resultado_row.size > 0 else np.array([])
cum_cdi       = np.cumsum(cdi_row)       if cdi_row.size       > 0 else np.array([])

# =========================
# 4. Sidebar e Filtros
# =========================

st.sidebar.title("Configurações")
chart_type = st.sidebar.selectbox(
    "Tipo de Visualização",
    ["Retorno Acumulado", "Retorno Mensal", "Comparação", "Fluxo de Caixa"]
)

if st.sidebar.button("🔄 Atualizar Base de Dados"):
    with st.spinner('Atualizando dados...'):
        clear_cache()
        st.session_state.data = load_data(force_update=True)
        st.success('Dados atualizados com sucesso!')
        st.rerun()

date_range = st.sidebar.select_slider(
    "Período de Análise",
    options=range(len(months)),
    value=(0, len(months)-1),
    format_func=lambda x: months[x]
)
selected_months = months[date_range[0]:date_range[1]+1]

# =========================
# 5. Criação dos DataFrames (Performance e Fluxo)
# =========================

def safe_slice(arr, start, end):
    """Retorna o slice de arr, se possível. Se arr for vazio, retorna vazio."""
    if arr.size == 0:
        return arr
    return arr[start:end]

df_performance = pd.DataFrame({
    'Mês': selected_months,
    'Carteira (%)': safe_slice(resultado_row, date_range[0], date_range[1]+1).round(2),
    'CDI (%)': safe_slice(cdi_row, date_range[0], date_range[1]+1).round(2),
    'Excedente (%)': safe_slice(excedente_row, date_range[0], date_range[1]+1).round(2)
}).dropna()

df_fluxo = pd.DataFrame({
    'Mês': selected_months,
    'Desconto CDI': safe_slice(desconto_row, date_range[0], date_range[1]+1),
    'Gemini': safe_slice(gemini_row, date_range[0], date_range[1]+1),
    'Líquido': safe_slice(liquido_row, date_range[0], date_range[1]+1)
}).dropna()

for col in ['Desconto CDI', 'Gemini', 'Líquido']:
    if col in df_fluxo.columns:
        df_fluxo[col] = df_fluxo[col].apply(lambda x: f"R$ {x:_.2f}".replace(".", ",").replace("_", "."))

# =========================
# 6. Cabeçalho com Métricas
# =========================

st.title("📊 Dashboard Financeiro")
capital_inicial = 300000

# Descobre último índice válido em resultado_row
if resultado_row.size > 0:
    last_valid_idx = next((i for i in range(len(resultado_row)-1, -1, -1) if not np.isnan(resultado_row[i])), -1)
else:
    last_valid_idx = -1

col1, col2, col3, col4 = st.columns(4)
with col1:
    if last_valid_idx >= 0 and last_valid_idx < capital_row.size:
        retorno_total = ((capital_row[last_valid_idx] / capital_row[0]) - 1) * 100
        if last_valid_idx > 0:
            retorno_mensal = ((capital_row[last_valid_idx] / capital_row[last_valid_idx-1]) - 1) * 100
        else:
            retorno_mensal = np.nan
        st.metric("Retorno Total", f"{retorno_total:.1f}%", f"{retorno_mensal:+.1f}% último mês")
with col2:
    if last_valid_idx >= 0 and last_valid_idx < cdi_row.size and last_valid_idx < excedente_row.size:
        cdi_mes = cdi_row[last_valid_idx]
        excedente_mensal = excedente_row[last_valid_idx]
        percentual_excedente = (excedente_mensal / cdi_mes * 100) if cdi_mes != 0 else np.nan
        # Se também existir cum_resultado e cum_cdi
        if cum_resultado.size > 0 and cum_cdi.size > 0 and last_valid_idx < cum_resultado.size and last_valid_idx < cum_cdi.size:
            dif_acumulada = (cum_resultado[last_valid_idx] - cum_cdi[last_valid_idx])
        else:
            dif_acumulada = np.nan
        st.metric("vs CDI", 
                  f"{dif_acumulada:.1f}%" if not np.isnan(dif_acumulada) else "N/A",
                  f"{percentual_excedente:.2f}% acima do CDI no mês" if not np.isnan(percentual_excedente) else "N/A")
with col3:
    if last_valid_idx >= 0 and last_valid_idx < capital_row.size:
        st.metric("Patrimônio Líquido", 
                  f"R$ {capital_row[last_valid_idx]:_.2f}".replace(".", ",").replace("_", "."),
                  f"{((capital_row[last_valid_idx] / capital_inicial) - 1)*100:+.2f}% desde o início")
with col4:
    st.metric("Capital Base", 
              f"R$ {capital_inicial:_.2f}".replace(".", ",").replace("_", "."),
              "Capital inicial")

# =========================
# 7. Gráficos Principais
# =========================

fig = go.Figure()
slice_range = slice(date_range[0], date_range[1]+1)

def safe_array_for_plot(arr):
    """Retorna o slice para plot, se arr for vazio, retorna array de NaN."""
    if arr.size == 0:
        return np.array([np.nan]*(date_range[1]-date_range[0]+1))
    return arr[slice_range]

if chart_type == "Retorno Acumulado":
    y_carteira = safe_array_for_plot(cum_resultado)
    y_cdi = safe_array_for_plot(cum_cdi)

    fig.add_trace(go.Scatter(
        x=selected_months,
        y=y_carteira,
        name='Carteira',
        line=dict(color='#4B9CD3', width=3),
        fill='tonexty',
        fillcolor='rgba(75,156,211,0.2)'
    ))
    fig.add_trace(go.Scatter(
        x=selected_months,
        y=y_cdi,
        name='CDI',
        line=dict(color='#FFA500', width=3),
        fill='tonexty',
        fillcolor='rgba(255,165,0,0.1)'
    ))
    fig.update_layout(title="Retorno Acumulado", template='plotly_dark', height=500)

elif chart_type == "Retorno Mensal":
    y_retorno = safe_array_for_plot(retornos_mensais.to_numpy())
    y_cdi = safe_array_for_plot(cdi_row)

    colors = ['#4B9CD3' if (not np.isnan(x) and x >= 0) else '#ff4444' for x in y_retorno]
    fig.add_trace(go.Bar(
        x=selected_months,
        y=y_retorno,
        name='Retorno',
        marker_color=colors
    ))
    fig.add_trace(go.Scatter(
        x=selected_months,
        y=y_cdi,
        name='CDI',
        line=dict(color='#FFA500', width=3)
    ))
    fig.update_layout(title="Retorno Mensal", template='plotly_dark', height=500)

elif chart_type == "Comparação":
    y_excedente = safe_array_for_plot(excedente_row)
    fig.add_trace(go.Bar(
        x=selected_months,
        y=y_excedente,
        name='Excedente ao CDI',
        marker_color=np.where((~np.isnan(y_excedente)) & (y_excedente >= 0), '#4B9CD3', '#ff4444')
    ))
    fig.update_layout(title="Excedente ao CDI", template='plotly_dark', height=500)

else:  # Fluxo de Caixa
    y_gemini = safe_array_for_plot(gemini_row)
    y_desconto = safe_array_for_plot(desconto_row)
    cores_gemini = ['#4B9CD3' if (not np.isnan(x) and x >= 0) else '#ff4444' for x in y_gemini]
    cores_desconto = ['#FFA500' if (not np.isnan(x) and x >= 0) else '#ff6b6b' for x in y_desconto]

    fig.add_trace(go.Bar(
        x=selected_months,
        y=y_gemini,
        name='Gemini',
        marker_color=cores_gemini,
        text=[f"R$ {x:,.2f}" if not np.isnan(x) else "" for x in y_gemini],
        textposition='outside'
    ))
    fig.add_trace(go.Bar(
        x=selected_months,
        y=y_desconto,
        name='Desconto CDI',
        marker_color=cores_desconto,
        text=[f"R$ {x:,.2f}" if not np.isnan(x) else "" for x in y_desconto],
        textposition='outside'
    ))
    y_min = np.nanmin([0, np.nanmin(y_gemini), np.nanmin(y_desconto)])
    y_max = np.nanmax([0, np.nanmax(y_gemini), np.nanmax(y_desconto)])
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
            range=[y_min*1.1 if not np.isnan(y_min) else 0,
                   y_max*1.1 if not np.isnan(y_max) else 1]
        )
    )

st.plotly_chart(fig, use_container_width=True)

# =========================
# 8. Tabelas (AgGrid)
# =========================

def config_grid(df_data):
    gb = GridOptionsBuilder.from_dataframe(df_data)
    gb.configure_default_column(min_column_width=100)
    gb.configure_grid_options(domLayout='normal')
    return gb.build()

col_left, col_right = st.columns(2)
with col_left:
    st.markdown("### Performance Detalhada")
    AgGrid(df_performance, gridOptions=config_grid(df_performance),
           theme='streamlit', fit_columns_on_grid_load=True)

with col_right:
    st.markdown("### Fluxo Financeiro")
    AgGrid(df_fluxo, gridOptions=config_grid(df_fluxo),
           theme='streamlit', fit_columns_on_grid_load=True)

# =========================
# 9. Análise Estatística
# =========================

with st.expander("📊 Análise Estatística"):
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    
    with stat_col1:
        if resultado_row.size == 0 or np.isnan(resultado_row).all():
            st.warning("Sem dados de 'Resultado' para calcular melhor mês.")
        else:
            melhor_mes = np.nanmax(resultado_row)
            idx_melhor = np.nanargmax(resultado_row)
            # Garante que não vai estourar o index de months
            mes_melhor = months[idx_melhor] if idx_melhor < len(months) else "N/D"
            st.metric("Melhor Mês", f"{melhor_mes:+.1f}%", mes_melhor)
    
    with stat_col2:
        if resultado_row.size == 0 or np.isnan(resultado_row).all():
            st.warning("Sem dados de 'Resultado' para calcular pior mês.")
        else:
            pior_mes = np.nanmin(resultado_row)
            idx_pior = np.nanargmin(resultado_row)
            mes_pior = months[idx_pior] if idx_pior < len(months) else "N/D"
            st.metric("Pior Mês", f"{pior_mes:+.1f}%", mes_pior,
                      delta_color="inverse" if pior_mes < 0 else "normal")
    
    with stat_col3:
        # Precisamos garantir que resultado_row e cdi_row tenham mesmo tamanho
        if resultado_row.size == 0 or cdi_row.size == 0:
            st.warning("Sem dados suficientes para comparar com CDI.")
        else:
            # Cria uma Series para poder comparar
            valid_data = pd.Series(resultado_row[:len(cdi_row)]) > cdi_row[:len(resultado_row)]
            valid_data = valid_data.dropna()
            meses_acima_cdi = valid_data.sum()
            total_meses = valid_data.shape[0]
            if total_meses > 0:
                st.metric("Meses acima do CDI", f"{meses_acima_cdi}/{total_meses}",
                          f"{(meses_acima_cdi/total_meses*100):.1f}% do período")
            else:
                st.metric("Meses acima do CDI", "0/0", "Sem dados")

# =========================
# 10. Indicadores Detalhados
# =========================

st.markdown("### 📈 Indicadores Detalhados")
tabs = st.tabs(["Risco vs Retorno", "Comportamento", "Comparativo de Mercado"])

# -- Aba: Risco vs Retorno
with tabs[0]:
    col_risk1, col_risk2 = st.columns(2)
    
    with col_risk1:
        # Se só tiver 1 valor, rolling(3) não faz sentido; tratamos para evitar erro
        if len(retornos_mensais) < 3:
            rolling_vol = pd.Series([np.nan]*len(retornos_mensais))
        else:
            rolling_vol = (retornos_mensais/100).rolling(3).std() * np.sqrt(12) * 100
        
        fig_vol = go.Figure(go.Scatter(
            x=months[:len(rolling_vol)],
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
    
    with col_risk2:
        if capital_row.size == 0:
            st.warning("Sem dados de capital para drawdown.")
        else:
            capital_series = pd.Series(capital_row)
            capital_max = capital_series.cummax()
            drawdown = ((capital_series - capital_max) / capital_max) * 100
            fig_dd = go.Figure(go.Scatter(
                x=months[:len(drawdown)],  # limita ao tamanho de capital_row
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

# -- Aba: Comportamento
with tabs[1]:
    col_comp1, col_comp2 = st.columns(2)
    
    with col_comp1:
        if resultado_row.size == 0:
            st.warning("Sem dados de 'Resultado' para calcular meses positivos.")
            meses_positivos = 0
            total_meses = 0
        else:
            meses_positivos = np.sum(resultado_row > 0)
            total_meses = len(resultado_row)
        
        if total_meses > 0:
            val_gauge = (meses_positivos / total_meses)*100
        else:
            val_gauge = 0
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val_gauge,
            title={'text': "Meses com Resultado Positivo"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#4B9CD3"},
                'steps': [
                    {'range': [0, 50], 'color': '#ff6b6b'},
                    {'range': [50, 75], 'color': '#ffd93d'},
                    {'range': [75, 100], 'color': '#6BCB77'}
                ]
            }
        ))
        fig_gauge.update_layout(template='plotly_dark', height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col_comp2:
        if capital_row.size == 0:
            st.warning("Sem dados de capital para evolução.")
        else:
            fig_capital = go.Figure()
            fig_capital.add_trace(go.Scatter(
                x=months[:len(capital_row)],
                y=capital_row,
                fill='tozeroy',
                name='Capital Total',
                line=dict(color='#4B9CD3', width=2)
            ))
            fig_capital.add_hline(
                y=300000,
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
                yaxis=dict(tickformat=",.0f", tickprefix="R$ ")
            )
            st.plotly_chart(fig_capital, use_container_width=True)

# -- Aba: Comparativo de Mercado
with tabs[2]:
    col_market1, col_market2 = st.columns(2)
    
    with col_market1:
        # Se resultado_row estiver vazio, não plotar
        if resultado_row.size == 0:
            st.warning("Sem dados de 'Resultado' para comparar com meta mensal.")
        else:
            cores_retorno = ['#4B9CD3' if x >= 0 else '#ff4444' for x in resultado_row]
            fig_retorno = go.Figure()
            fig_retorno.add_trace(go.Bar(
                x=months[:len(resultado_row)],
                y=resultado_row,
                name='Retorno Realizado',
                marker_color=cores_retorno,
                text=[f"{x:.1f}%" for x in resultado_row],
                textposition='outside'
            ))
            
            # Se cdi_row tiver tamanho suficiente
            if cdi_row.size == 0:
                st.warning("Sem dados de CDI para comparar no gráfico.")
                y_cdi = np.array([np.nan]*len(resultado_row))
            else:
                y_cdi = cdi_row[:len(resultado_row)]
            
            fig_retorno.add_trace(go.Scatter(
                x=months[:len(y_cdi)],
                y=y_cdi,
                name='Meta (CDI)',
                line=dict(color='#FFA500', width=2, dash='dash')
            ))
            
            y_min = np.nanmin([0, np.nanmin(resultado_row)]) if resultado_row.size > 0 else 0
            y_max = np.nanmax([0, np.nanmax(resultado_row)]) if resultado_row.size > 0 else 1
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
                    range=[y_min*1.1, y_max*1.1]
                )
            )
            st.plotly_chart(fig_retorno, use_container_width=True)
    
    with col_market2:
        # Calcula comparativo de rentabilidade
        if cum_resultado.size == 0:
            last_valid_resultado = 0
        else:
            last_valid_resultado = cum_resultado[-1]  # último valor
        
        if cum_cdi.size == 0:
            last_valid_cdi = 0
        else:
            last_valid_cdi = cum_cdi[-1]
        
        rent_comparison = pd.DataFrame({
            'Investimento': ['Sua Carteira', 'CDI', 'Poupança'],
            'Rentabilidade': [
                last_valid_resultado,
                last_valid_cdi,
                last_valid_cdi * 0.7  # Exemplo de poupança ~70% CDI
            ]
        })
        
        fig_rent = go.Figure(go.Bar(
            x=rent_comparison['Investimento'],
            y=rent_comparison['Rentabilidade'],
            text=[f"{val:.2f}%" for val in rent_comparison['Rentabilidade']],
            textposition='auto',
            marker_color=['#4B9CD3', '#FFA500', '#ff4444']
        ))
        
        y_max = max(rent_comparison['Rentabilidade']) if not rent_comparison['Rentabilidade'].isna().all() else 0
        fig_rent.update_layout(
            title="Comparativo de Rentabilidade",
            template='plotly_dark',
            height=300,
            showlegend=False,
            yaxis_title="Rentabilidade Total (%)",
            yaxis=dict(tickformat='.2f', ticksuffix='%'),
            yaxis_range=[0, y_max*1.2 if y_max > 0 else 1]
        )
        st.plotly_chart(fig_rent, use_container_width=True)

# =========================
# 11. Explicações
# =========================

with st.expander("ℹ️ Entenda os Indicadores"):
    st.write("""
    **Volatilidade Móvel:** Mede a variação dos retornos nos últimos 3 meses. Quanto menor, mais estável é seu investimento.
    
    **Drawdown:** Representa a queda do investimento em relação ao seu ponto mais alto. Quanto menor o drawdown, menor o risco de perda.
    
    **Meses com Resultado Positivo:** Mostra a consistência dos seus resultados, indicando quantos meses você teve retorno positivo.
    
    **Evolução do Capital:** Acompanha o crescimento do seu patrimônio ao longo do tempo.
    
    **Retorno vs Meta:** Compara seu retorno mensal com o CDI (meta).
    
    **Comparativo de Rentabilidade:** Mostra como seu investimento se compara com outras opções do mercado.
    """)

