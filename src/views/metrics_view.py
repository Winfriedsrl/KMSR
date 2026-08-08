import streamlit as st


def _render_metrics_formulas_view():
    st.title("Formule Metriche")

    # ------------------------------------------------------------------ #
    # Metriche di base                                                     #
    # ------------------------------------------------------------------ #
    st.markdown("## Metriche di base")
    st.markdown(
        "Siano $G$ l'insieme di tutti i periodi corretti per una query, "
        "$B_1, \\ldots, B_m \\subseteq G$ i blocchi del ground truth, "
        "e $\\hat{P}$ i periodi predetti dal retriever."
    )

    col_flat, col_block = st.columns(2, gap="large")

    with col_flat:
        st.markdown("#### Flat Recall")
        st.caption("Quanti periodi corretti sono stati recuperati, indipendentemente dal blocco di appartenenza.")
        st.latex(r"\mathrm{FlatRecall} = \frac{|G \cap \hat{P}|}{|G|}")

    with col_block:
        st.markdown("#### Block Recall")
        st.caption("Quanti blocchi sono stati recuperati interamente. Un blocco parziale vale zero.")
        st.latex(
            r"\mathrm{BlockRecall} = \frac{1}{m}\sum_{j=1}^{m}"
            r"\mathbf{1}\!\left[B_j \subseteq \hat{P}\right]"
        )

    st.divider()

    # ------------------------------------------------------------------ #
    # Score per tipo di query                                              #
    # ------------------------------------------------------------------ #
    st.markdown("## Score per tipo di query")
    st.caption(
        "Le query sono classificate in tre tipi (A, B, A+B). "
        "Per ciascun tipo si calcola la media delle metriche di base su tutte le query del tipo."
    )

    st.latex(r"S_A = \overline{\mathrm{FlatRecall}}_{\,A}")
    st.latex(r"S_B = \overline{\mathrm{BlockRecall}}_{\,B}")
    st.latex(r"S_{A+B} = \frac{\overline{\mathrm{FlatRecall}}_{\,A+B} + \overline{\mathrm{BlockRecall}}_{\,A+B}}{2}")

    st.divider()

    # ------------------------------------------------------------------ #
    # Macro Score e Weighted Score                                         #
    # ------------------------------------------------------------------ #
    st.markdown("## Macro Score e Weighted Score")

    col_macro, col_weighted = st.columns(2, gap="large")

    with col_macro:
        st.markdown("#### Macro Score")
        st.caption("Media semplice degli score per tipo. Ogni tipo pesa uguale.")
        st.latex(r"\mathrm{Macro} = \frac{S_A + S_B + S_{A+B}}{3}")

    with col_weighted:
        st.markdown("#### Weighted Score")
        st.caption("Media pesata: ogni tipo pesa in proporzione al numero di query che contiene.")
        st.latex(r"w_X = \frac{n_X}{n_A + n_B + n_{A+B}}")
        st.latex(r"\mathrm{Weighted} = w_A S_A + w_B S_B + w_{A+B} S_{A+B}")

    st.divider()

    # ------------------------------------------------------------------ #
    # Metriche di ranking                                                  #
    # ------------------------------------------------------------------ #
    st.markdown("## Metriche di ranking")
    st.markdown(
        "Flat Recall e Block Recall misurano **cosa** il retriever trova, ignorando l'ordine. "
        "Le metriche di ranking misurano anche **dove** lo trova: trovare il periodo giusto "
        "al posto 1 vale più che trovarlo al posto 20."
    )

    col_ndcg, col_bap = st.columns(2, gap="large")

    with col_ndcg:
        st.markdown("#### NDCG — per query di Tipo A")
        st.markdown(
            "**Idea:** ogni periodo corretto trovato alla posizione $r$ riceve un punteggio "
            "$1 / \\log_2(r+1)$. Posizione 1 vale 1.0, posizione 3 vale 0.5, posizione 7 vale 0.33, ecc. "
            "Il totale viene normalizzato sul caso ideale (tutti i corretti nelle primissime posizioni)."
        )
        st.markdown("**Notazione:**")
        st.markdown(
            "- $G$ = insieme dei periodi corretti (ground truth)\n"
            "- $p_r$ = periodo restituito dal retriever alla posizione $r$\n"
            "- $|G|$ = numero di periodi corretti attesi"
        )
        st.latex(r"\mathrm{DCG} = \sum_{r=1}^{k} \frac{\mathbf{1}[p_r \in G]}{\log_2(r+1)}")
        st.latex(r"\mathrm{IDCG} = \sum_{i=1}^{|G|} \frac{1}{\log_2(i+1)}")
        st.latex(r"\mathrm{NDCG} = \frac{\mathrm{DCG}}{\mathrm{IDCG}}")
        st.caption("IDCG è il DCG del caso perfetto: tutti i |G| periodi corretti nelle prime posizioni. NDCG = 1 significa ordinamento ideale.")

    with col_bap:
        st.markdown("#### Block AP — per query di Tipo B")
        st.markdown(
            "**Idea:** per ogni blocco, guarda a che posizione nella lista viene trovato il suo "
            "ultimo elemento — quella è la posizione in cui il blocco è **completo**. "
            "Più i blocchi si completano in alto, più il punteggio è alto. "
            "Il risultato è normalizzato su [0, 1] dividendo per il caso ideale."
        )

        st.markdown("**Variabili:**")
        st.markdown(
            "- $m$ = numero totale di blocchi nel ground truth\n"
            "- $m^*$ = numero di blocchi **completamente** recuperati ($m^* \\leq m$)\n"
            "- $r_{(j)}$ = posizione in cui il $j$-esimo blocco si completa, "
            "con i blocchi ordinati per posizione di completamento crescente\n"
            "- $s_j$ = numero di periodi nel $j$-esimo blocco (dimensione del blocco)\n"
            "- $R_j = \\sum_{i=1}^{j} s_{(i)}$ = posizione di completamento ideale del $j$-esimo blocco "
            "(blocchi ordinati dal più piccolo al più grande, senza spazi tra loro)"
        )

        st.markdown("**Passo 1 — Block AP grezzo** (somma sui soli blocchi trovati, divide per $m$ totale):")
        st.latex(r"\mathrm{BlockAP}_{\mathrm{raw}} = \frac{1}{m} \sum_{j=1}^{m^*} \frac{j}{r_{(j)}}")

        st.markdown("**Passo 2 — Block AP ideale** (caso in cui tutti gli $m$ blocchi sono trovati nelle prime posizioni possibili):")
        st.latex(r"\mathrm{IBAP} = \frac{1}{m} \sum_{j=1}^{m} \frac{j}{R_j}")

        st.markdown("**Passo 3 — Normalizzazione** (divide grezzo per ideale, risultato in $[0, 1]$):")
        st.latex(r"\mathrm{BlockAP} = \frac{\mathrm{BlockAP}_{\mathrm{raw}}}{\mathrm{IBAP}} = \frac{\displaystyle\sum_{j=1}^{m^*} \frac{j}{r_{(j)}}}{\displaystyle\sum_{j=1}^{m} \frac{j}{R_j}}")

        st.caption(
            "BlockAP = 0 se nessun blocco è recuperato interamente. "
            "BlockAP = 1 se tutti i blocchi sono recuperati e completati nelle posizioni più alte possibili. "
            "I blocchi parzialmente recuperati non contribuiscono (tutto-o-niente per blocco, coerente con Block Recall)."
        )

    st.divider()

    # ------------------------------------------------------------------ #
    # Valori di riferimento                                                #
    # ------------------------------------------------------------------ #
    st.markdown("## Valori di riferimento")
    st.markdown(
        "Tutte le metriche sono comprese tra **0** (nessun risultato corretto) e **1** (perfetto). "
        "I range seguenti sono indicativi per questo task; usali per confrontare modelli tra loro."
    )

    try:
        import plotly.graph_objects as go

        # Ordine bottom→top nel chart orizzontale (l'ultimo della lista appare in cima)
        metric_labels = ["Macro / Weighted", "Block AP", "Block Recall", "NDCG", "Flat Recall"]

        # Larghezza di ogni segmento (deve sommare a 1.0 per riga)
        scarso_w   = [0.30, 0.20, 0.20, 0.40, 0.40]
        discreto_w = [0.20, 0.20, 0.20, 0.20, 0.20]
        buono_w    = [0.20, 0.20, 0.20, 0.20, 0.20]
        ottimo_w   = [0.30, 0.40, 0.40, 0.20, 0.20]

        scarso_lbl   = ["0.00–0.29", "0.00–0.19", "0.00–0.19", "0.00–0.39", "0.00–0.39"]
        discreto_lbl = ["0.30–0.49", "0.20–0.39", "0.20–0.39", "0.40–0.59", "0.40–0.59"]
        buono_lbl    = ["0.50–0.69", "0.40–0.59", "0.40–0.59", "0.60–0.79", "0.60–0.79"]
        ottimo_lbl   = ["0.70–1.00", "0.60–1.00", "0.60–1.00", "0.80–1.00", "0.80–1.00"]

        fig = go.Figure()
        for name, widths, labels, color in [
            ("Scarso",   scarso_w,   scarso_lbl,   "#d9534f"),
            ("Discreto", discreto_w, discreto_lbl, "#f0ad4e"),
            ("Buono",    buono_w,    buono_lbl,    "#5cb85c"),
            ("Ottimo",   ottimo_w,   ottimo_lbl,   "#1a7340"),
        ]:
            fig.add_trace(go.Bar(
                name=name,
                y=metric_labels,
                x=widths,
                orientation="h",
                marker_color=color,
                text=labels,
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(size=11, color="white"),
                hovertemplate="<b>%{y}</b><br>" + name + ": %{text}<extra></extra>",
            ))

        fig.update_layout(
            barmode="stack",
            xaxis=dict(range=[0, 1], tickformat=".1f", title="Score"),
            yaxis=dict(title=""),
            height=260,
            margin=dict(l=0, r=0, t=10, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

    st.info(
        "**Come leggere i numeri**\n\n"
        "- Confronta sempre modelli sullo **stesso dataset** e con lo **stesso k**.\n"
        "- Un miglioramento di 0.05 su Weighted Score è già significativo a parità di configurazione.\n"
        "- Se Flat Recall è alta ma NDCG è bassa, il modello trova i periodi giusti ma li posiziona in fondo alla lista.\n"
        "- Se Block Recall è molto inferiore a Flat Recall, il retriever recupera periodi isolati ma non completa i blocchi."
    )
