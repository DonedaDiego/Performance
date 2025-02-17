import streamlit as st
import pandas as pd
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder
import plotly.graph_objects as go

# =======================================
# EXEMPLO DE FUNÇÕES E DADOS (SIMPLIFICADO)
# =======================================
def config_grid(df_data):
    gb = GridOptionsBuilder.from_dataframe(df_data)
    gb.configure_default_column(min_column_width=100)
    gb.configure_grid_options(domLayout='normal')
    return gb.build()

# Função auxiliar para não quebrar ao fatiar arrays vazios
def safe_slice(arr, start, end):
    if arr.size == 0:
        return arr
    return arr[start:end]

# =======================================
# SUPOREMOS QUE VOCÊ JÁ CARREGOU OS DADOS
# =======================================
# Exemplo de arrays simulando seus dados
# (substitua pelas variáveis corretas do seu app)
months = ['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez']

# Imagine que parte esteja vazio (NaN)
resultado_row   = np.array([0.001, np.nan, 0.02, 0.03, np.nan, 0.05, np.nan, 0.07, 0.08, 0.09, 0.10, 0.12]) * 100
cdi_row         = np.array([0.001, 0.001, 0.001, np.nan, np.nan, 0.001, 0.001, 0.001, 0.001, 0.001, np.nan, 0.001]) * 100
excedente_row   = resultado_row - cdi_row
capital_row     = np.array([300000, 301000, 305000, 310000, np.nan, 320000, 325000, np.nan, 340000, 345000, 350000, 360000])
retornos_mensais = pd.Series(capital_row).pct_change() * 100

# Exemplo de DataFrame para tabelas
df_performance = pd.DataFrame({
    'Mês': months,
    'Carteira (%)': resultado_row.round(2),
    'CDI (%)': cdi_row.round(2),
    'Excedente (%)': excedente_row.round(2),
})

df_fluxo = pd.DataFrame({
    'Mês': months,
    'Algum Valor': [1000, 2000, 1500, None, 3000, 3500, None, 200, 700, 1000, 1200, None],
})

# =======================================
# CONFIGURAÇÃO DE PÁGINA
# =======================================
st.set_page_config(layout="wide", page_title="Dashboard Financeiro", page_icon="📊")

st.title("📊 Dashboard Financeiro")

# =======================================
# EXEMPLO DE MÉTRICAS
# =======================================
capital_inicial = 300000

# Achar último índice válido
# (ou -1 se nenhum for válido)
if resultado_row.size > 0:
    last_valid_idx = next((i for i in range(len(resultado_row)-1, -1, -1)
                           if not np.isnan(resultado_row[i])), -1)
else:
    last_valid_idx = -1

col1, col2, col3, col4 = st.columns(4)

with col1:
    if last_valid_idx >= 0 and last_valid_idx < capital_row.size and not np.isnan(capital_row[0]):
        # Calcula retorno total
        retorno_total = ((capital_row[last_valid_idx] / capital_row[0]) - 1)*100
        retorno_mensal = np.nan
        if last_valid_idx > 0 and not np.isnan(capital_row[last_valid_idx-1]):
            retorno_mensal = ((capital_row[last_valid_idx] / capital_row[last_valid_idx-1]) - 1)*100
        
        # Formata strings, tratando NaN
        retorno_total_str  = f"{retorno_total:.2f}%" if not np.isnan(retorno_total) else "Sem dados"
        retorno_mensal_str = f"{retorno_mensal:+.2f}%" if not np.isnan(retorno_mensal) else " "
        
        st.metric("Retorno Total", retorno_total_str, retorno_mensal_str)
    else:
        # Se não tem dados, exibe aviso ou 0
        st.metric("Retorno Total", "Sem dados", "")

with col2:
    # vs CDI
    if last_valid_idx >= 0 and last_valid_idx < cdi_row.size:
        cdi_mes = cdi_row[last_valid_idx]
        excedente_mensal = excedente_row[last_valid_idx] if last_valid_idx < excedente_row.size else np.nan
        
        # Diferença acumulada
        # Exemplo: se quiser cumsum
        cum_resultado = np.nancumsum(resultado_row)
        cum_cdi = np.nancumsum(cdi_row)
        
        if (last_valid_idx < cum_resultado.size) and (last_valid_idx < cum_cdi.size):
            dif_acumulada = cum_resultado[last_valid_idx] - cum_cdi[last_valid_idx]
        else:
            dif_acumulada = np.nan
        
        if not np.isnan(cdi_mes) and cdi_mes != 0:
            percentual_excedente = (excedente_mensal / cdi_mes)*100
        else:
            percentual_excedente = np.nan
        
        dif_str = f"{dif_acumulada:.2f}%" if not np.isnan(dif_acumulada) else "Sem dados"
        perc_excedente_str = f"{percentual_excedente:.2f}% acima do CDI no mês" if not np.isnan(percentual_excedente) else ""
        
        st.metric("vs CDI", dif_str, perc_excedente_str)
    else:
        st.metric("vs CDI", "Sem dados", "")

with col3:
    if last_valid_idx >= 0 and last_valid_idx < capital_row.size:
        pl = capital_row[last_valid_idx]
        pl_str = f"R$ {pl:_.2f}".replace(".", ",").replace("_", ".")
        
        if not np.isnan(pl):
            variacao_inicial = ((pl / capital_inicial) - 1)*100
            variacao_str = f"{variacao_inicial:+.2f}% desde o início"
        else:
            variacao_str = "Sem dados"
        
        st.metric("Patrimônio Líquido", pl_str, variacao_str)
    else:
        st.metric("Patrimônio Líquido", "Sem dados", "")

with col4:
    base_str = f"R$ {capital_inicial:_.2f}".replace(".", ",").replace("_", ".")
    st.metric("Capital Base", base_str, "Capital inicial")

# =======================================
# EXEMPLO DE GRÁFICO
# =======================================
chart_type = st.selectbox("Tipo de Visualização", ["Retorno Acumulado", "Retorno Mensal"])

fig = go.Figure()

if chart_type == "Retorno Acumulado":
    # Se não houver dados, mostra aviso
    if np.isnan(resultado_row).all() or np.isnan(cdi_row).all():
        st.warning("Sem dados para exibir Retorno Acumulado.")
    else:
        cum_resultado = np.nancumsum(resultado_row)
        cum_cdi       = np.nancumsum(cdi_row)
        
        fig.add_trace(go.Scatter(
            x=months[:len(cum_resultado)],
            y=cum_resultado,
            name='Carteira',
            line=dict(color='#4B9CD3', width=3),
            fill='tonexty',
            fillcolor='rgba(75,156,211,0.2)'
        ))
        fig.add_trace(go.Scatter(
            x=months[:len(cum_cdi)],
            y=cum_cdi,
            name='CDI',
            line=dict(color='#FFA500', width=3),
            fill='tonexty',
            fillcolor='rgba(255,165,0,0.1)'
        ))
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Retorno Mensal":
    if np.isnan(retornos_mensais).all():
        st.warning("Sem dados para exibir Retorno Mensal.")
    else:
        colors = ['#4B9CD3' if x >= 0 else '#ff4444' for x in retornos_mensais.fillna(0)]
        fig.add_trace(go.Bar(
            x=months[:len(retornos_mensais)],
            y=retornos_mensais.fillna(0),
            marker_color=colors,
            name='Retorno'
        ))
        # Comparar com CDI
        fig.add_trace(go.Scatter(
            x=months[:len(cdi_row)],
            y=np.nan_to_num(cdi_row),  # substitui NaN por 0
            name='CDI',
            line=dict(color='#FFA500', width=3)
        ))
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)

# =======================================
# EXEMPLO DE TABELAS
# =======================================
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### Performance Detalhada")
    grid_options = config_grid(df_performance)
    AgGrid(df_performance, gridOptions=grid_options,
           theme='streamlit', fit_columns_on_grid_load=True)

with col_b:
    st.markdown("### Fluxo Financeiro")
    grid_options = config_grid(df_fluxo)
    AgGrid(df_fluxo, gridOptions=grid_options,
           theme='streamlit', fit_columns_on_grid_load=True)

# =======================================
# EXEMPLO DE ANÁLISE ESTATÍSTICA
# =======================================
with st.expander("📊 Análise Estatística"):
    st.write("Exemplo de melhor/pior mês e meses acima do CDI")
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    
    with col_stat1:
        if resultado_row.size == 0 or np.isnan(resultado_row).all():
            st.warning("Sem dados de 'Resultado' para calcular melhor mês.")
        else:
            melhor_valor = np.nanmax(resultado_row)
            idx_melhor = np.nanargmax(resultado_row)
            mes_melhor = months[idx_melhor] if idx_melhor < len(months) else "N/D"
            st.metric("Melhor Mês", f"{melhor_valor:.2f}%", mes_melhor)
    
    with col_stat2:
        if resultado_row.size == 0 or np.isnan(resultado_row).all():
            st.warning("Sem dados de 'Resultado' para calcular pior mês.")
        else:
            pior_valor = np.nanmin(resultado_row)
            idx_pior = np.nanargmin(resultado_row)
            mes_pior = months[idx_pior] if idx_pior < len(months) else "N/D"
            delta_color = "inverse" if pior_valor < 0 else "normal"
            st.metric("Pior Mês", f"{pior_valor:.2f}%", mes_pior, delta_color=delta_color)
    
    with col_stat3:
        # Exemplo de comparação com CDI
        if resultado_row.size == 0 or cdi_row.size == 0:
            st.warning("Sem dados para comparar com CDI.")
        else:
            length = min(len(resultado_row), len(cdi_row))
            valid_data = pd.Series(resultado_row[:length]) > cdi_row[:length]
            valid_data = valid_data.dropna()
            meses_acima_cdi = valid_data.sum()
            total_meses = len(valid_data)
            if total_meses > 0:
                perc = (meses_acima_cdi / total_meses)*100
                st.metric("Meses acima do CDI", f"{meses_acima_cdi}/{total_meses}", f"{perc:.1f}% do período")
            else:
                st.metric("Meses acima do CDI", "0/0", "Sem dados")


st.write("---")
st.markdown("#### Observações finais:")
st.write("""
- Se preferir trocar "Sem dados" por "N/A" ou deixar em branco, basta ajustar as strings onde for necessário.
- Em cenários em que só há 1 mês preenchido, o `retornos_mensais` não consegue calcular variação, pois precisa de pelo menos 2 pontos.
- Para lidar com NaN em gráficos, utilizamos `np.nan_to_num(...)` ou `fillna(0)`, mas verifique se faz sentido transformar em 0 ou se prefere remover do gráfico.
""")


