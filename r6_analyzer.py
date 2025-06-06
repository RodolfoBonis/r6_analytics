# r6_analyzer.py

import os
import json
import pandas as pd
import streamlit as st
import altair as alt
import io

# ------------------------------------------------------------
# 1) Importa ReportLab e Matplotlib para gerar PDF com gráficos
# ------------------------------------------------------------
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_LEFT

import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 2) Função para carregar os JSONs de cada jogador
# ------------------------------------------------------------
@st.cache_data
def load_player_data(player_dir):
    """
    Para um diretório de jogador (ex: "players/AlphaWolf.DD"), carrega:
      - overview.json
      - maps.json
      - operators.json

    Retorna uma tupla (overview_json, maps_json, ops_json).
    Qualquer um dos três que não exista retorna None.
    """
    overview_path = os.path.join(player_dir, "overview.json")
    maps_path = os.path.join(player_dir, "maps.json")
    ops_path = os.path.join(player_dir, "operators.json")

    overview = None
    maps = None
    ops = None

    if os.path.isfile(overview_path):
        with open(overview_path, "r", encoding="utf-8") as f:
            overview = json.load(f)

    if os.path.isfile(maps_path):
        with open(maps_path, "r", encoding="utf-8") as f:
            maps = json.load(f)

    if os.path.isfile(ops_path):
        with open(ops_path, "r", encoding="utf-8") as f:
            ops = json.load(f)

    return overview, maps, ops


# ------------------------------------------------------------
# 3) Funções de parsing para DataFrames
# ------------------------------------------------------------
def parse_overview_to_df(overview_json, player_name):
    """
    Extrai do overview.json os principais indicadores:
      - matchesPlayed, matchesWon, matchesLost, winPct, kills, deaths, kdRatio

    Retorna um DataFrame de uma linha:
        [player, matchesPlayed, matchesWon, matchesLost, winPct, kills, deaths, kdRatio]
    """
    segmento_overview = None
    for seg in overview_json.get("segments", []):
        if seg.get("type") == "overview":
            segmento_overview = seg
            break

    if segmento_overview is None:
        return pd.DataFrame()

    stats = segmento_overview.get("stats", {})

    def get_stat(key):
        entry = stats.get(key)
        return entry.get("value") if (entry and entry.get("value") is not None) else 0

    data = {
        "player": player_name,
        "matchesPlayed": get_stat("matchesPlayed"),
        "matchesWon": get_stat("matchesWon"),
        "matchesLost": get_stat("matchesLost"),
        "winPct": get_stat("winPercentage"),
        "kills": get_stat("kills"),
        "deaths": get_stat("deaths"),
        "kdRatio": get_stat("kdRatio"),
    }
    return pd.DataFrame([data])


def parse_maps_to_df(maps_json, player_name):
    """
    Converte maps.json em DataFrame, extraindo para cada mapa:
      - mapName, matchesPlayed, matchesWon, winPct, kills, deaths, kdRatio

    Retorna um DataFrame com colunas:
      [player, mapName, matchesPlayed, matchesWon, winPct, kills, deaths, kdRatio]
    """
    rows = []
    for entry in maps_json:
        metadata = entry.get("metadata", {})
        stats = entry.get("stats", {})

        map_name = metadata.get("mapName", entry.get("attributes", {}).get("map", "Unknown"))

        def get_stat(key):
            sub = stats.get(key)
            return sub.get("value") if (sub and sub.get("value") is not None) else 0

        row = {
            "player": player_name,
            "mapName": map_name,
            "matchesPlayed": get_stat("matchesPlayed"),
            "matchesWon": get_stat("matchesWon"),
            "winPct": get_stat("winPercentage"),
            "kills": get_stat("kills"),
            "deaths": get_stat("deaths"),
            "kdRatio": get_stat("kdRatio"),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def parse_operators_to_df(ops_json, player_name):
    """
    Converte operators.json em DataFrame, extraindo para cada operador:
      - operatorName, side (attacker/defender), matchesPlayed, matchesWon, winPct, kills, deaths, kdRatio

    Filtra qualquer operador cujo nome seja "Unknown" (ou vazio).
    Retorna um DataFrame com colunas:
      [player, operatorName, side, matchesPlayed, matchesWon, winPct, kills, deaths, kdRatio]
    """
    rows = []
    for entry in ops_json:
        metadata = entry.get("metadata", {})
        attributes = entry.get("attributes", {})
        stats = entry.get("stats", {})

        op_name = metadata.get("operatorName", attributes.get("operator", "")).strip()
        side = attributes.get("side", "all")  # "attacker" ou "defender"

        # === FILTRO: descartar se op_name estiver vazio ou for apenas "Unknown" ===
        if not op_name or op_name.lower() == "unknown":
            continue

        def get_stat(key):
            sub = stats.get(key)
            return sub.get("value") if (sub and sub.get("value") is not None) else 0

        row = {
            "player": player_name,
            "operatorName": op_name,
            "side": side,
            "matchesPlayed": get_stat("matchesPlayed"),
            "matchesWon": get_stat("matchesWon"),
            "winPct": get_stat("winPercentage"),
            "kills": get_stat("kills"),
            "deaths": get_stat("deaths"),
            "kdRatio": get_stat("kdRatio"),
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 4) Função para extrair “Playstyle” (estilo de jogo) do overview.json
# ------------------------------------------------------------
def extract_playstyles(overview_json):
    """
    Recebe o overview_json de um jogador. Procura todas as entradas de playstyle:
      - playstyleAttackerBreacher, playstyleAttackerEntryFragger, etc.

    Cada chave de playstyle possui, dentro de 'metadata', um campo 'usage.value' que indica
    o percentual de uso daquele estilo.

    Retorna uma lista de tuplas [(nome_do_playstyle, usage_percent), ...].
    """
    segmento_overview = None
    for seg in overview_json.get("segments", []):
        if seg.get("type") == "overview":
            segmento_overview = seg
            break

    if segmento_overview is None:
        return []

    stats = segmento_overview.get("stats", {})
    playstyles = []

    for key, entry in stats.items():
        if key.startswith("playstyle"):
            display_name = entry.get("displayName", "")
            metadata = entry.get("metadata", {})
            usage = metadata.get("usage", {})
            percent = usage.get("value", 0)
            playstyles.append((display_name, percent))

    playstyles.sort(key=lambda x: x[1], reverse=True)
    return playstyles  # lista de tuplas (playstyle, usage_percent)


# ------------------------------------------------------------
# 5) Funções de cálculo/agregação
# ------------------------------------------------------------
def compute_best_worst_maps(maps_df_agg, min_map_matches):
    """
    Recebe um DataFrame agregado de mapas e o mínimo de partidas para considerar:
      colunas: [mapName, matchesPlayed, matchesWon, winPct, kills, deaths, kdRatio]

    Retorna dois DataFrames (melhores_maps, piores_maps),
    ordenados por winPct desc e winPct asc, filtrados por matchesPlayed >= min_map_matches.
    """
    df = maps_df_agg[maps_df_agg["matchesPlayed"] >= min_map_matches].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    melhores = df.sort_values("winPct", ascending=False).reset_index(drop=True)
    piores = df.sort_values("winPct", ascending=True).reset_index(drop=True)
    return melhores, piores


def compute_side_performance(ops_df_agg):
    """
    Recebe DataFrame de operadores agregados por equipe.
    Agrupa por side e calcula:
      matchesPlayed, matchesWon, matchesLost, winPctSide

    Retorna DataFrame colunas: [side, matchesPlayed, matchesWon, matchesLost, winPctSide]
    """
    grouped = ops_df_agg.groupby("side").agg(
        {
            "matchesPlayed": "sum",
            "matchesWon": "sum",
        }
    ).reset_index()

    grouped["matchesLost"] = grouped["matchesPlayed"] - grouped["matchesWon"]
    grouped["winPctSide"] = grouped.apply(
        lambda row: (row["matchesWon"] / row["matchesPlayed"] * 100) if row["matchesPlayed"] > 0 else 0,
        axis=1,
    )
    return grouped.sort_values("matchesPlayed", ascending=False)


def compute_most_played_operators(ops_df_agg_player, min_operator_matches, top_n=10):
    """
    Recebe DataFrame de operadores agregado por jogador e o mínimo de partidas para considerar.
    Retorna 3 DataFrames (cada um limitados a top_n):
      - mais_jogados: top_n por matchesPlayed (matchesPlayed >= min_operator_matches)
      - maior_win:    top_n por winPct (matchesPlayed >= min_operator_matches)
      - maior_kpm:    top_n por killsPerMatch (matchesPlayed >= min_operator_matches)
    """
    df = ops_df_agg_player[ops_df_agg_player["matchesPlayed"] >= min_operator_matches].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    mais_jogados = df.sort_values("matchesPlayed", ascending=False).head(top_n).reset_index(drop=True)
    maior_win = df.sort_values("winPct", ascending=False).head(top_n).reset_index(drop=True)
    df["killsPerMatch"] = df["kills"] / df["matchesPlayed"].replace(0, pd.NA)
    maior_kpm = df.sort_values("killsPerMatch", ascending=False).head(top_n).reset_index(drop=True)

    return mais_jogados, maior_win, maior_kpm


# ------------------------------------------------------------
# 6) Função para gerar PDF com as análises (usando ReportLab + Matplotlib)
# ------------------------------------------------------------
def create_pdf_report(
    jogadores: list[str],
    overview_raw_by_player: dict[str, pd.DataFrame],
    maps_df_agg: pd.DataFrame,
    ops_df_agg_player: pd.DataFrame,
    ops_df_agg_team: pd.DataFrame,
    sides_df_agg: pd.DataFrame,
    min_operator_matches: int,
    min_map_matches: int,
) -> bytes:
    """
    Gera um relatório PDF com:
      - Melhores e piores operadores por jogador (filtrados por min_operator_matches)
      - Melhores e piores mapas da equipe (filtrados por min_map_matches)
      - Melhor e pior lado (ATK x DEF) da equipe
      - Estatísticas gerais de equipe (win%, K/D, etc.)
      - Gráficos embutidos: desempenho de operadores e de mapas (com anotações)
    """

    # --------------------------------------------------------------------------------
    # 1) Preparação inicial: estilos e Document
    # --------------------------------------------------------------------------------
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=24,
        leftMargin=24,
        topMargin=24,
        bottomMargin=24,
    )

    styles = getSampleStyleSheet()
    estilos = {
        "titulo": ParagraphStyle(
            "titulo", parent=styles["Heading1"], alignment=TA_LEFT, fontSize=16, spaceAfter=12
        ),
        "subtitulo": ParagraphStyle(
            "subtitulo", parent=styles["Heading2"], alignment=TA_LEFT, fontSize=12, spaceAfter=8
        ),
        "normal": ParagraphStyle(
            "normal", parent=styles["BodyText"], fontSize=10, spaceAfter=6
        ),
        "tabela_cabecalho": ParagraphStyle(
            "tabela_cabecalho", parent=styles["Heading4"], fontSize=10, alignment=TA_LEFT, spaceAfter=4
        ),
    }

    elements = []

    # --------------------------------------------------------------------------------
    # 2) Limpeza / filtros gerais
    # --------------------------------------------------------------------------------
    # 2.1 Remover qualquer linha com "unknown" em operador ou mapa:
    if "operatorName" in ops_df_agg_player.columns:
        ops_df_agg_player = ops_df_agg_player[
            ops_df_agg_player["operatorName"].str.lower() != "unknown"
        ].copy()

    if "mapName" in maps_df_agg.columns:
        maps_df_agg = maps_df_agg[maps_df_agg["mapName"].str.lower() != "unknown"].copy()

    # 2.2 Filtrar operadores abaixo de min_operator_matches
    ops_df_agg_player = ops_df_agg_player[
        ops_df_agg_player["matchesPlayed"] >= min_operator_matches
    ].copy()

    # 2.3 Filtrar mapas abaixo de min_map_matches
    maps_df_agg = maps_df_agg[maps_df_agg["matchesPlayed"] >= min_map_matches].copy()

    # 2.4 (os lados não têm “player”; é agregado por equipe.
    #     Caso haja “unknown” na coluna “side”, eliminamos também:)
    if "side" in sides_df_agg.columns:
        sides_df_agg = sides_df_agg[sides_df_agg["side"].str.lower() != "unknown"].copy()

    # --------------------------------------------------------------------------------
    # 3) Título geral
    # --------------------------------------------------------------------------------
    elements.append(Paragraph("Relatório Detalhado de R6 Siege", estilos["titulo"]))
    elements.append(Spacer(1, 12))

    # --------------------------------------------------------------------------------
    # 4) Estatísticas da equipe: melhores/piores mapas e lados
    # --------------------------------------------------------------------------------
    # 4.1 Melhores e piores mapas (toda equipe):
    if not maps_df_agg.empty:
        # Ordenar por winPct desc para melhores, asc para piores
        maps_sorted = maps_df_agg.sort_values("winPct", ascending=False).reset_index(drop=True)

        # Melhor mapa (primeira linha), pior mapa (última linha)
        melhor_mapa = maps_sorted.iloc[0]
        pior_mapa = maps_sorted.iloc[-1]

        elements.append(Paragraph("📊 Melhores e Piores Mapas (Equipe)", estilos["subtitulo"]))
        texto_mapas = (
            f"<b>Melhor Mapa:</b> {melhor_mapa['mapName']} "
            f"(Partidas: {melhor_mapa['matchesPlayed']}, Win%: {melhor_mapa['winPct']:.2f}%, "
            f"K/D: {melhor_mapa.get('kdRatio', 0):.2f})<br/>"
            f"<b>Pior Mapa:</b> {pior_mapa['mapName']} "
            f"(Partidas: {pior_mapa['matchesPlayed']}, Win%: {pior_mapa['winPct']:.2f}%, "
            f"K/D: {pior_mapa.get('kdRatio', 0):.2f})"
        )
        elements.append(Paragraph(texto_mapas, estilos["normal"]))
        elements.append(Spacer(1, 6))

    # 4.2 Melhor e pior lado (ATK x DEF, para toda equipe):
    if not sides_df_agg.empty:
        sides_sorted = sides_df_agg.sort_values("winPctSide", ascending=False).reset_index(drop=True)
        melhor_lado = sides_sorted.iloc[0]
        pior_lado = sides_sorted.iloc[-1]

        elements.append(Paragraph("⚔️ Melhor e Pior Lado (Equipe)", estilos["subtitulo"]))
        texto_lados = (
            f"<b>Melhor Lado:</b> {melhor_lado['side']} "
            f"(Partidas: {melhor_lado['matchesPlayed']}, Win%: {melhor_lado['winPctSide']:.2f}%)<br/>"
            f"<b>Pior Lado:</b> {pior_lado['side']} "
            f"(Partidas: {pior_lado['matchesPlayed']}, Win%: {pior_lado['winPctSide']:.2f}%)"
        )
        elements.append(Paragraph(texto_lados, estilos["normal"]))
        elements.append(Spacer(1, 12))

    # --------------------------------------------------------------------------------
    # 5) Para cada jogador, mostrar insights de operadores
    # --------------------------------------------------------------------------------
    for player in jogadores:
        elements.append(Paragraph(f"🎯 Insights de Operadores: {player}", estilos["subtitulo"]))

        # 5.1 Filtrar só linhas deste jogador:
        df_ops_jog = ops_df_agg_player[ops_df_agg_player["player"] == player].copy()
        if df_ops_jog.empty:
            elements.append(Paragraph("Sem dados suficientes de operadores para este jogador.", estilos["normal"]))
            elements.append(Spacer(1, 8))
            continue

        # 5.2 Encontrar top e bottom operators (por winPct)
        df_ops_sorted = df_ops_jog.sort_values("winPct", ascending=False).reset_index(drop=True)
        top_op = df_ops_sorted.iloc[0]
        bot_op = df_ops_sorted.iloc[-1]

        texto_ops = (
            f"<b>Top Operator:</b> {top_op['operatorName']} "
            f"(Partidas: {top_op['matchesPlayed']}, Win%: {top_op['winPct']:.2f}%, "
            f"Kills/Match: {top_op['kills'] / top_op['matchesPlayed']:.2f})<br/>"
            f"<b>Worst Operator:</b> {bot_op['operatorName']} "
            f"(Partidas: {bot_op['matchesPlayed']}, Win%: {bot_op['winPct']:.2f}%, "
            f"Kills/Match: {bot_op['kills'] / bot_op['matchesPlayed']:.2f})"
        )
        elements.append(Paragraph(texto_ops, estilos["normal"]))
        elements.append(Spacer(1, 12))

        # 5.3 Gráfico de operadores do jogador: scatter de (matchesPlayed vs winPct)
        fig_op = plt.figure(figsize=(5, 3))
        ax_op = fig_op.add_subplot(111)
        ax_op.scatter(df_ops_jog["matchesPlayed"], df_ops_jog["winPct"], s=40, c="tab:blue", alpha=0.7)
        ax_op.set_title(f"Desempenho de Operadores – {player}")
        ax_op.set_xlabel("Partidas Jogadas")
        ax_op.set_ylabel("Win %")
        ax_op.grid(alpha=0.3)

        for _, row in df_ops_jog.iterrows():
            ax_op.annotate(
                row["operatorName"],
                xy=(row["matchesPlayed"], row["winPct"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6,
            )

        buf_op = io.BytesIO()
        fig_op.tight_layout()
        fig_op.savefig(buf_op, format="PNG", dpi=150)
        plt.close(fig_op)
        buf_op.seek(0)

        elements.append(Image(buf_op, width=400, height=240))
        elements.append(Spacer(1, 12))

    # --------------------------------------------------------------------------------
    # 6) Gráfico conjunto de mapas de equipe
    # --------------------------------------------------------------------------------
    if not maps_df_agg.empty:
        elements.append(Paragraph("🗺️ Desempenho de Mapas (Equipe)", estilos["subtitulo"]))

        fig_map = plt.figure(figsize=(6, 3.5))
        ax_map = fig_map.add_subplot(111)
        ax_map.scatter(maps_df_agg["matchesPlayed"], maps_df_agg["winPct"], s=50, c="tab:green", alpha=0.7)
        ax_map.set_title("Desempenho de Mapas – Equipe")
        ax_map.set_xlabel("Partidas Jogadas")
        ax_map.set_ylabel("Win %")
        ax_map.grid(alpha=0.3)

        for _, row in maps_df_agg.iterrows():
            ax_map.annotate(
                row["mapName"],
                xy=(row["matchesPlayed"], row["winPct"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )

        buf_map = io.BytesIO()
        fig_map.tight_layout()
        fig_map.savefig(buf_map, format="PNG", dpi=150)
        plt.close(fig_map)
        buf_map.seek(0)

        elements.append(Image(buf_map, width=500, height=280))
        elements.append(Spacer(1, 12))

    # --------------------------------------------------------------------------------
    # 7) Estatísticas de sides de equipe (detalhe em tabela)
    # --------------------------------------------------------------------------------
    if not sides_df_agg.empty:
        elements.append(Paragraph("📋 Tabela de Desempenho por Lado (Equipe)", estilos["subtitulo"]))
        tabela_data = [["Lado", "Partidas", "Win %", "Kills", "Deaths", "K/D"]]
        for _, row in sides_df_agg.iterrows():
            tabela_data.append(
                [
                    row["side"],
                    int(row["matchesPlayed"]),
                    f"{row['winPctSide']:.1f}%",
                    int(row.get("kills", 0)),
                    int(row.get("deaths", 0)),
                    f"{row.get('kdRatio', 0):.2f}",
                ]
            )

        t = Table(tabela_data, colWidths=[60, 60, 60, 60, 60, 60])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                    ("TOPPADDING", (0, 0), (-1, 0), 6),
                ]
            )
        )
        elements.append(t)
        elements.append(Spacer(1, 12))

    # --------------------------------------------------------------------------------
    # 8) Insira estatísticas gerais da equipe (por player) a partir de overview_raw_by_player
    # --------------------------------------------------------------------------------
    elements.append(Paragraph("📈 Estatísticas Gerais por Jogador", estilos["subtitulo"]))
    for player, df_over_json in overview_raw_by_player.items():
        # Parse novamente para DataFrame só para garantir a estrutura.
        df_over_df = parse_overview_to_df(df_over_json, player)
        if df_over_df.empty:
            elements.append(Paragraph(f"{player}: Sem dados gerais disponíveis.", estilos["normal"]))
            continue

        row = df_over_df.iloc[0]
        tex = (
            f"<b>{player}</b>: "
            f"Partidas Jogadas: {row.get('matchesPlayed', 0)}, "
            f"Vitórias: {row.get('matchesWon', 0)}, "
            f"Win%: {row.get('winPct', 0):.2f}%, "
            f"Kills: {row.get('kills', 0)}, "
            f"Deaths: {row.get('deaths', 0)}, "
            f"K/D: {row.get('kdRatio', 0):.2f}"
        )
        elements.append(Paragraph(tex, estilos["normal"]))
        elements.append(Spacer(1, 6))

    # --------------------------------------------------------------------------------
    # 9) Monta e retorna o PDF (tentando ficar em uma página, só quebrando se realmente necessário)
    # --------------------------------------------------------------------------------
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# ------------------------------------------------------------
# 7) Função principal de UI (Streamlit)
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="R6 Siege Team Analyzer", layout="wide")
    st.title("🏠 R6 Siege Analytics Dashboard")

    # --------------------------------------------------------
    # 7.1) Descobrir todos os jogadores na pasta "players/"
    # --------------------------------------------------------
    base_folder = "players"
    if not os.path.isdir(base_folder):
        st.error(f"A pasta `{base_folder}` não existe na raiz do projeto. Crie-a e coloque nela as pastas dos jogadores.")
        st.stop()

    all_players = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]

    jogadores_selecionados = st.sidebar.multiselect(
        "Selecione o(s) jogador(es) ou equipe:",
        options=all_players,
        default=all_players,
    )

    if not jogadores_selecionados:
        st.warning("Nenhum jogador selecionado. Selecione pelo menos um jogador para visualizar métricas.")
        st.stop()

    # --------------------------------------------------------
    # 7.2) Parâmetros de filtros mínimos (sidebar)
    # --------------------------------------------------------
    st.sidebar.markdown("---")
    min_operator_matches = st.sidebar.number_input(
        "🔢 Mínimo de partidas por operador para considerar:",
        min_value=1,
        value=100,
        step=1,
        help="Somente operadores com ≥ este número de partidas serão considerados nos Insights",
    )
    min_map_matches = st.sidebar.number_input(
        "🔢 Mínimo de partidas por mapa para considerar:",
        min_value=1,
        value=10,
        step=1,
        help="Somente mapas com ≥ este número de partidas serão considerados nos Insights",
    )

    # --------------------------------------------------------
    # 7.3) Carregar JSONs e transformar em DataFrames para cada jogador
    # --------------------------------------------------------
    overview_list = []
    maps_list = []
    ops_list = []

    overview_raw_by_player = {}

    for player in jogadores_selecionados:
        dir_player = os.path.join(base_folder, player)
        overview_json, maps_json, ops_json = load_player_data(dir_player)

        if overview_json is None:
            st.warning(f"O arquivo `overview.json` não foi encontrado para `{player}`.")
        else:
            overview_raw_by_player[player] = overview_json
            df_over = parse_overview_to_df(overview_json, player)
            overview_list.append(df_over)

        if maps_json is None:
            st.warning(f"O arquivo `maps.json` não foi encontrado para `{player}`.")
        else:
            df_maps = parse_maps_to_df(maps_json, player)
            maps_list.append(df_maps)

        if ops_json is None:
            st.warning(f"O arquivo `operators.json` não foi encontrado para `{player}`.")
        else:
            df_ops = parse_operators_to_df(ops_json, player)
            ops_list.append(df_ops)

    overview_df = pd.concat(overview_list, ignore_index=True) if overview_list else pd.DataFrame()
    maps_df = pd.concat(maps_list, ignore_index=True) if maps_list else pd.DataFrame()
    ops_df = pd.concat(ops_list, ignore_index=True) if ops_list else pd.DataFrame()

    # --------------------------------------------------------
    # 7.4) Agregação por equipe (somar todos os jogadores selecionados)
    # --------------------------------------------------------
    # 7.4.1) Overview agregado da equipe
    if not overview_df.empty:
        equipe_agg = {
            "matchesPlayed": overview_df["matchesPlayed"].sum(),
            "matchesWon": overview_df["matchesWon"].sum(),
            "matchesLost": overview_df["matchesLost"].sum(),
            "kills": overview_df["kills"].sum(),
            "deaths": overview_df["deaths"].sum(),
        }
        equipe_agg["winPct"] = (equipe_agg["matchesWon"] / equipe_agg["matchesPlayed"] * 100) if equipe_agg["matchesPlayed"] > 0 else 0
        equipe_agg["kdRatio"] = (equipe_agg["kills"] / equipe_agg["deaths"]) if equipe_agg["deaths"] > 0 else 0
        df_overview_equipe = pd.DataFrame([{"player": "Equipe"} | equipe_agg])
    else:
        df_overview_equipe = pd.DataFrame()

    # 7.4.2) Maps agregado
    if not maps_df.empty:
        maps_df_agg = maps_df.groupby("mapName").agg(
            {
                "matchesPlayed": "sum",
                "matchesWon": "sum",
                "kills": "sum",
                "deaths": "sum",
            }
        ).reset_index()
        maps_df_agg["matchesLost"] = maps_df_agg["matchesPlayed"] - maps_df_agg["matchesWon"]
        maps_df_agg["winPct"] = maps_df_agg.apply(
            lambda row: (row["matchesWon"] / row["matchesPlayed"] * 100) if row["matchesPlayed"] > 0 else 0,
            axis=1,
        )
        maps_df_agg["kdRatio"] = maps_df_agg.apply(
            lambda row: (row["kills"] / row["deaths"]) if row["deaths"] > 0 else 0,
            axis=1,
        )
    else:
        maps_df_agg = pd.DataFrame()

    # 7.4.3) Operadores agregado (e também separado por jogador)
    if not ops_df.empty:
        # **Agregado da equipe** (somando todos jogadores)
        ops_df_agg_team = ops_df.groupby(["operatorName", "side"]).agg(
            {
                "matchesPlayed": "sum",
                "matchesWon": "sum",
                "kills": "sum",
                "deaths": "sum",
            }
        ).reset_index()
        ops_df_agg_team["matchesLost"] = ops_df_agg_team["matchesPlayed"] - ops_df_agg_team["matchesWon"]
        ops_df_agg_team["winPct"] = ops_df_agg_team.apply(
            lambda row: (row["matchesWon"] / row["matchesPlayed"] * 100) if row["matchesPlayed"] > 0 else 0,
            axis=1,
        )
        ops_df_agg_team["kdRatio"] = ops_df_agg_team.apply(
            lambda row: (row["kills"] / row["deaths"]) if row["deaths"] > 0 else 0,
            axis=1,
        )
        ops_df_agg_team["killsPerMatch"] = ops_df_agg_team["kills"] / ops_df_agg_team["matchesPlayed"].replace(0, pd.NA)

        # **Agregado POR JOGADOR** (para usar na aba de Operadores e nos Insights)
        ops_df_agg_player = ops_df.groupby(["player", "operatorName", "side"]).agg(
            {
                "matchesPlayed": "sum",
                "matchesWon": "sum",
                "kills": "sum",
                "deaths": "sum",
            }
        ).reset_index()
        ops_df_agg_player["matchesLost"] = ops_df_agg_player["matchesPlayed"] - ops_df_agg_player["matchesWon"]
        ops_df_agg_player["winPct"] = ops_df_agg_player.apply(
            lambda row: (row["matchesWon"] / row["matchesPlayed"] * 100) if row["matchesPlayed"] > 0 else 0,
            axis=1,
        )
        ops_df_agg_player["kdRatio"] = ops_df_agg_player.apply(
            lambda row: (row["kills"] / row["deaths"]) if row["deaths"] > 0 else 0,
            axis=1,
        )
        ops_df_agg_player["killsPerMatch"] = ops_df_agg_player["kills"] / ops_df_agg_player["matchesPlayed"].replace(0, pd.NA)
    else:
        ops_df_agg_team = pd.DataFrame()
        ops_df_agg_player = pd.DataFrame()

    # --------------------------------------------------------
    # 7.5) Configurar as abas do Dashboard
    # --------------------------------------------------------
    tabs = st.tabs(
        [
            "📊 Resumo Geral",
            "🗺️ Desempenho por Mapa",
            "⚔️ Desempenho por Lado",
            "🔫 Operadores",
            "💡 Insights & Recomendações",
        ]
    )

    # -------------------------------------------
    # Aba 1: Resumo Geral (Overview)
    # -------------------------------------------
    with tabs[0]:
        st.header("✅ Resumo Geral")
        if not overview_df.empty:
            st.subheader("Por Jogador")
            st.dataframe(
                overview_df[["player", "matchesPlayed", "matchesWon", "matchesLost", "winPct", "kills", "deaths", "kdRatio"]],
                use_container_width=True,
            )

        if not df_overview_equipe.empty:
            st.subheader("Total da Equipe")
            st.dataframe(
                df_overview_equipe[["player", "matchesPlayed", "matchesWon", "matchesLost", "winPct", "kills", "deaths", "kdRatio"]],
                use_container_width=True,
            )

    # -------------------------------------------
    # Aba 2: Desempenho por Mapa
    # -------------------------------------------
    with tabs[1]:
        st.header("🗺️ Desempenho por Mapa")
        if maps_df_agg.empty:
            st.info("Não há dados de mapas disponíveis.")
        else:
            melhores5, piores5 = compute_best_worst_maps(maps_df_agg, min_map_matches)

            st.subheader("Top 5 Mapas com Maior Taxa de Vitória")
            st.dataframe(
                melhores5[["mapName", "matchesPlayed", "matchesWon", "matchesLost", "winPct", "kdRatio"]].head(5),
                use_container_width=True,
            )

            st.subheader("Top 5 Mapas com Menor Taxa de Vitória")
            st.dataframe(
                piores5[["mapName", "matchesPlayed", "matchesWon", "matchesLost", "winPct", "kdRatio"]].head(5),
                use_container_width=True,
            )

            # Gráfico comparativo: Top 10 mapas mais jogados x vitórias/derrotas
            st.subheader("📈 Top 10 Mapas Mais Jogados (Vitórias x Derrotas)")
            df_top10_mapa = maps_df_agg.sort_values("matchesPlayed", ascending=False).head(10).copy()
            df_plot_mapa = df_top10_mapa[["mapName", "matchesWon", "matchesLost"]].melt(
                id_vars=["mapName"],
                value_vars=["matchesWon", "matchesLost"],
                var_name="Resultado",
                value_name="Contagem",
            )
            df_plot_mapa["Resultado"] = df_plot_mapa["Resultado"].map(
                {
                    "matchesWon": "Vitórias",
                    "matchesLost": "Derrotas",
                }
            )

            chart_mapa = (
                alt.Chart(df_plot_mapa)
                .mark_bar()
                .encode(
                    x=alt.X("mapName:N", sort="-y", title="Mapa"),
                    y=alt.Y("Contagem:Q", title="Partidas"),
                    color=alt.Color("Resultado:N", title="Resultado", scale=alt.Scale(domain=["Vitórias", "Derrotas"], scheme="set1")),
                    tooltip=["mapName", "Resultado", "Contagem"],
                )
                .properties(height=350)
            )
            st.altair_chart(chart_mapa, use_container_width=True)

            with st.expander("Mostrar todos os mapas ordenados por taxa de vitória"):
                st.dataframe(
                    melhores5[["mapName", "matchesPlayed", "matchesWon", "matchesLost", "winPct", "kdRatio"]],
                    use_container_width=True,
                )

    # -------------------------------------------
    # Aba 3: Desempenho por Lado (ATK vs DEF)
    # -------------------------------------------
    with tabs[2]:
        st.header("⚔️ Desempenho por Lado (ATK vs DEF)")
        if ops_df_agg_team.empty:
            st.info("Não há dados de operadores para calcular desempenho por lado.")
        else:
            side_perf_team = compute_side_performance(ops_df_agg_team)
            st.subheader("Tabela de Desempenho por Lado")
            st.dataframe(
                side_perf_team[["side", "matchesPlayed", "matchesWon", "matchesLost", "winPctSide"]],
                use_container_width=True,
            )

            st.subheader("📊 Comparativo de Lados (Vitórias x Derrotas)")
            df_plot_side = side_perf_team.melt(
                id_vars=["side", "matchesPlayed"],
                value_vars=["matchesWon", "matchesLost"],
                var_name="Resultado",
                value_name="Contagem",
            )
            df_plot_side["Resultado"] = df_plot_side["Resultado"].map(
                {
                    "matchesWon": "Vitórias",
                    "matchesLost": "Derrotas",
                }
            )

            chart_side = (
                alt.Chart(df_plot_side)
                .mark_bar()
                .encode(
                    x=alt.X("side:N", title="Lado"),
                    y=alt.Y("Contagem:Q", title="Partidas"),
                    color=alt.Color("Resultado:N", title="Resultado", scale=alt.Scale(domain=["Vitórias", "Derrotas"], scheme="set1")),
                    tooltip=["side", "Resultado", "Contagem"],
                )
                .properties(height=350)
            )
            st.altair_chart(chart_side, use_container_width=True)

            st.markdown(
                """
                - **matchesPlayed**: número total de partidas jogadas por operadores de cada lado  
                - **matchesWon**: soma de vitórias de todos os operadores desse lado  
                - **matchesLost**: partidas jogadas menos partidas vencidas  
                - **winPctSide**: taxa de vitória do lado (`matchesWon / matchesPlayed * 100`)  
                """
            )

    # -------------------------------------------
    # Aba 4: Operadores (por Jogador)
    # -------------------------------------------
    with tabs[3]:
        st.header("🔫 Estatísticas de Operadores (por Jogador)")
        if ops_df_agg_player.empty:
            st.info("Não há dados de operadores disponíveis.")
        else:
            for player in jogadores_selecionados:
                st.subheader(f"📋 {player} – Operadores")
                df_jogador = ops_df_agg_player[ops_df_agg_player["player"] == player].copy()
                if df_jogador.empty:
                    st.write(f"Nenhum operador encontrado para `{player}`.")
                    continue

                df_filtrado = df_jogador[df_jogador["matchesPlayed"] >= min_operator_matches]
                if df_filtrado.empty:
                    st.write(f"Nenhum operador de `{player}` com ≥ {min_operator_matches} partidas.")
                    continue

                mais_jogados, maior_win, maior_kpm = compute_most_played_operators(df_jogador, min_operator_matches, top_n=10)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Top 10 Mais Jogados**")
                    st.dataframe(
                        mais_jogados[["operatorName", "side", "matchesPlayed", "winPct", "kdRatio"]],
                        use_container_width=True,
                    )
                    st.markdown("📊 Vitórias x Derrotas (Top 10 Mais Jogados)")
                    df_plot_ops = mais_jogados[["operatorName", "matchesWon", "matchesLost"]].melt(
                        id_vars=["operatorName"],
                        value_vars=["matchesWon", "matchesLost"],
                        var_name="Resultado",
                        value_name="Contagem",
                    )
                    df_plot_ops["Resultado"] = df_plot_ops["Resultado"].map(
                        {
                            "matchesWon": "Vitórias",
                            "matchesLost": "Derrotas",
                        }
                    )
                    chart_ops = (
                        alt.Chart(df_plot_ops)
                        .mark_bar()
                        .encode(
                            x=alt.X("operatorName:N", sort="-y", title="Operador"),
                            y=alt.Y("Contagem:Q", title="Partidas"),
                            color=alt.Color("Resultado:N", title="Resultado", scale=alt.Scale(domain=["Vitórias", "Derrotas"], scheme="set1")),
                            tooltip=["operatorName", "Resultado", "Contagem"],
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(chart_ops, use_container_width=True)

                with col2:
                    st.markdown("**Top 10 por Win %**")
                    st.dataframe(
                        maior_win[["operatorName", "side", "matchesPlayed", "winPct", "kdRatio"]],
                        use_container_width=True,
                    )
                    st.markdown("**Top 10 por Kills/Match**")
                    st.dataframe(
                        maior_kpm[["operatorName", "side", "matchesPlayed", "killsPerMatch", "kdRatio"]],
                        use_container_width=True,
                    )

                st.markdown("---")  # Separador antes do próximo jogador

    # -------------------------------------------
    # Aba 5: Insights & Recomendações
    # -------------------------------------------
    with tabs[4]:
        st.header("💡 Insights & Recomendações")

        if maps_df_agg.empty and ops_df_agg_player.empty:
            st.info("Não há dados suficientes para gerar insights.")
            st.stop()

        # ------------------------------------------------------------
        # 5.5.1) Melhor Mapa (apenas mapas com partidas >= min_map_matches)
        # ------------------------------------------------------------
        st.subheader("🏅 Melhor Mapa (― equipe)")
        if maps_df_agg.empty:
            st.write("Nenhum dado de mapa disponível.")
        else:
            df_maps_filtrado = maps_df_agg[maps_df_agg["matchesPlayed"] >= min_map_matches].copy()
            if df_maps_filtrado.empty:
                st.write(f"Nenhum mapa com ≥ {min_map_matches} partidas jogadas.")
            else:
                melhor_mapa = df_maps_filtrado.sort_values("winPct", ascending=False).iloc[0]
                st.markdown(
                    f"**Melhor mapa da equipe:** `{melhor_mapa['mapName']}` com _Win %_ de **{melhor_mapa['winPct']:.1f}%** "
                    f"em **{melhor_mapa['matchesPlayed']} partidas**."
                )
                st.markdown("#### Top 5 Mapas Filtrados (equipe)")
                st.dataframe(
                    df_maps_filtrado.sort_values("winPct", ascending=False)[
                        ["mapName", "matchesPlayed", "matchesWon", "matchesLost", "winPct", "kdRatio"]
                    ].head(5),
                    use_container_width=True,
                )
                st.markdown("📊 Distribuição de Mapas (Partidas x Win %)")
                chart_scatter_map = (
                    alt.Chart(df_maps_filtrado)
                    .mark_circle(size=80, opacity=0.7)
                    .encode(
                        x=alt.X("matchesPlayed:Q", title="Partidas Jogadas"),
                        y=alt.Y("winPct:Q", title="Win %"),
                        tooltip=["mapName", "matchesPlayed", "winPct", "kdRatio"],
                        color=alt.Color("winPct:Q", scale=alt.Scale(scheme="viridis"), title="Win %"),
                    )
                    .properties(height=350)
                )
                st.altair_chart(chart_scatter_map, use_container_width=True)

        st.markdown("---")

        # ------------------------------------------------------------
        # 5.5.2) Melhor Lado (attacker vs defender) – equipe
        # ------------------------------------------------------------
        st.subheader("⚔️ Melhor Lado (ATK × DEF) (― equipe)")
        if ops_df_agg_team.empty:
            st.write("Nenhum dado de operador disponível para calcular desempenho por lado.")
        else:
            side_perf_team = compute_side_performance(ops_df_agg_team)
            if side_perf_team.empty:
                st.write("Não há dados de lado.")
            else:
                best_side_row = side_perf_team[side_perf_team["matchesPlayed"] > 0].sort_values("winPctSide", ascending=False).iloc[0]
                st.markdown(
                    f"**Melhor lado da equipe:** `{best_side_row['side']}` com _Win %_ de **"
                    f"{best_side_row['winPctSide']:.1f}%** em **{best_side_row['matchesPlayed']} partidas**."
                )
                st.dataframe(
                    side_perf_team[["side", "matchesPlayed", "matchesWon", "matchesLost", "winPctSide"]],
                    use_container_width=True,
                )
                chart_side_pct = (
                    alt.Chart(side_perf_team)
                    .mark_bar()
                    .encode(
                        x=alt.X("side:N", title="Lado"),
                        y=alt.Y("winPctSide:Q", title="Win %"),
                        color=alt.Color("side:N", scale=alt.Scale(domain=["attacker", "defender"], scheme="set2")),
                        tooltip=["side", "matchesPlayed", "winPctSide"],
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart_side_pct, use_container_width=True)

        st.markdown("---")

        # ------------------------------------------------------------
        # 5.5.3) MELHOR OPERADOR POR LADO (por Jogador)
        # ------------------------------------------------------------
        st.subheader("🥇 Melhor Operador por Lado (por Jogador)")
        if ops_df_agg_player.empty:
            st.write("Nenhum dado de operador disponível.")
        else:
            for player in jogadores_selecionados:
                st.markdown(f"**🔹 {player}**")
                df_jogador = ops_df_agg_player[ops_df_agg_player["player"] == player].copy()
                df_jogador_filtrado = df_jogador[df_jogador["matchesPlayed"] >= min_operator_matches]
                if df_jogador_filtrado.empty:
                    st.write(f"  - Nenhum operador de `{player}` com ≥ {min_operator_matches} partidas.")
                    continue

                melhores_ops = []
                for lado in ["attacker", "defender"]:
                    df_lado = df_jogador_filtrado[df_jogador_filtrado["side"] == lado]
                    if not df_lado.empty:
                        melhor_op = df_lado.sort_values("winPct", ascending=False).iloc[0]
                        melhores_ops.append(
                            {
                                "player": player,
                                "side": lado,
                                "operatorName": melhor_op["operatorName"],
                                "matchesPlayed": int(melhor_op["matchesPlayed"]),
                                "winPct": round(melhor_op["winPct"], 1),
                                "killsPerMatch": round(melhor_op["kills"] / melhor_op["matchesPlayed"], 2),
                            }
                        )

                if not melhores_ops:
                    st.write(f"  - Nenhum operador de `{player}` com ≥ {min_operator_matches} partidas em nenhum dos lados.")
                else:
                    df_melhores_ops = pd.DataFrame(melhores_ops)
                    st.dataframe(
                        df_melhores_ops[["player", "side", "operatorName", "matchesPlayed", "winPct", "killsPerMatch"]],
                        use_container_width=True,
                    )

                    st.markdown("  📊 Distribuição de Operadores (Partidas x Win %) deste jogador")
                    chart_scatter_op = (
                        alt.Chart(df_jogador_filtrado)
                        .mark_circle(size=80, opacity=0.7)
                        .encode(
                            x=alt.X("matchesPlayed:Q", title="Partidas Jogadas"),
                            y=alt.Y("winPct:Q", title="Win %"),
                            color=alt.Color("side:N", title="Lado", scale=alt.Scale(domain=["attacker", "defender"], scheme="set1")),
                            tooltip=["operatorName", "side", "matchesPlayed", "winPct", "kdRatio"],
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(chart_scatter_op, use_container_width=True)

                st.markdown("---")  # Separador antes do próximo jogador

        st.markdown("---")

        # ------------------------------------------------------------
        # 5.5.4) Top Operadores por Lado (por Jogador)
        # ------------------------------------------------------------
        st.subheader("🎖️ Top Operadores por Lado (por Jogador)")
        if ops_df_agg_player.empty:
            st.write("Nenhum dado de operador disponível.")
        else:
            for player in jogadores_selecionados:
                st.markdown(f"**🔹 {player}**")
                df_jogador = ops_df_agg_player[ops_df_agg_player["player"] == player].copy()
                df_jogador_filtrado = df_jogador[df_jogador["matchesPlayed"] >= min_operator_matches]
                if df_jogador_filtrado.empty:
                    st.write(f"  - Nenhum operador de `{player}` com ≥ {min_operator_matches} partidas.")
                    continue

                for lado in ["attacker", "defender"]:
                    st.markdown(f"  • **Top 5 Operadores ({lado.capitalize()})**")
                    df_lado_top = df_jogador_filtrado[df_jogador_filtrado["side"] == lado].copy()
                    if df_lado_top.empty:
                        st.write(f"    - Nenhum operador {lado} de `{player}` com ≥ {min_operator_matches} partidas.")
                    else:
                        top_lado = df_lado_top.sort_values("winPct", ascending=False).head(5)
                        st.dataframe(
                            top_lado[
                                [
                                    "operatorName",
                                    "matchesPlayed",
                                    "winPct",
                                    "killsPerMatch",
                                    "kdRatio",
                                ]
                            ].reset_index(drop=True),
                            use_container_width=True,
                        )
                st.markdown("---")  # Separador antes do próximo jogador

        st.markdown("---")

        # ------------------------------------------------------------
        # 5.5.5) Melhor Estilo de Jogo (Playstyle) por Jogador
        # ------------------------------------------------------------
        st.subheader("🎯 Melhor Estilo de Jogo (Playstyle) por Jogador")
        for player in jogadores_selecionados:
            st.markdown(f"**🔹 {player}**")
            overview_json = overview_raw_by_player.get(player)
            if overview_json is None:
                st.write("  - Não foi possível extrair o playstyle (arquivo overview.json ausente).")
                continue

            playstyles = extract_playstyles(overview_json)
            if not playstyles:
                st.write("  - Nenhum dado de playstyle encontrado no overview.json.")
                continue

            melhor_estilo, uso_percent = playstyles[0]
            st.markdown(f"  - **Melhor Estilo:** `{melhor_estilo}` com **{uso_percent:.1f}%** de uso.")

            with st.expander(f"Ver todos os playstyles ({len(playstyles)}) para {player}"):
                df_play = pd.DataFrame(playstyles, columns=["Playstyle", "Uso (%)"])
                st.dataframe(df_play, use_container_width=True)

        st.markdown("---")
        st.markdown(
            """
            **Observações Finais**  
            - Ajuste `Mínimo de partidas por operador` e `Mínimo de partidas por mapa` na barra lateral para filtrar quais dados serão considerados no relatório.  
            - O PDF gerado agora leva em consideração exatamente aqueles valores mínimos e inclui gráficos internos.  
            """
        )

        # ------------------------------------------------------------
        # Botão para gerar relatório em PDF
        # ------------------------------------------------------------
        # 1) Garantir que sides_df_agg já esteja calculado
        if not ops_df_agg_team.empty:
            sides_df_agg = compute_side_performance(ops_df_agg_team)
        else:
            sides_df_agg = pd.DataFrame()

        st.subheader("📜 Gerar Relatório em PDF")
        if st.button("🖨️ Gerar Relatório PDF"):
            pdf_bytes = create_pdf_report(
                jogadores=jogadores_selecionados,
                overview_raw_by_player=overview_raw_by_player,
                maps_df_agg=maps_df_agg,
                ops_df_agg_player=ops_df_agg_player,
                ops_df_agg_team=ops_df_agg_team,
                sides_df_agg=sides_df_agg,
                min_operator_matches=min_operator_matches,
                min_map_matches=min_map_matches,
            )
            st.success("Relatório PDF gerado com sucesso! Clique abaixo para baixar.")
            st.download_button(
                label="⬇️ Baixar Relatório PDF",
                data=pdf_bytes,
                file_name="relatorio_r6_siege.pdf",
                mime="application/pdf",
            )


if __name__ == "__main__":
    main()
