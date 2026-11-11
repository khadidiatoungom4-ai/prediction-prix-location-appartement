import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image as PILImage

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocessing as pp

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PDF_PATH = REPORTS_DIR / "rapport_preprocessing.pdf"

PRIMARY = colors.HexColor("#1F3864")
ACCENT = colors.HexColor("#2E86AB")
LIGHT = colors.HexColor("#EAF2F8")
LIGHT_ALT = colors.HexColor("#F3F6FA")
BORDER = colors.HexColor("#C9D2DE")
TEXT = colors.HexColor("#222222")
GRAY = colors.HexColor("#555555")

MPL_PRIMARY = "#1F3864"
MPL_ACCENT = "#2E86AB"
MPL_GRAY = "#6B7280"

RAW_STATS = {
    "rows": 99492,
    "columns": 22,
    "duplicates": 84,
    "missing_total": 168860,
    "missing": [
        ("address", 91549),
        ("pets_allowed", 60424),
        ("amenities", 16044),
        ("cityname", 302),
        ("state", 302),
        ("bedrooms", 124),
        ("bathrooms", 63),
        ("latitude", 25),
        ("longitude", 25),
        ("price", 1),
        ("price_display", 1),
    ],
    "states": 51,
    "cities": 2972,
    "price_max": 52500,
}

FILTER_STEPS = [
    ("Données brutes", 99492, None),
    ("Doublons exacts", 99408, 84),
    ("Catégorie appartement", 99347, 61),
    ("Prix mensuel", 99343, 4),
    ("Prix strictement positif", 99342, 1),
    ("Surface strictement positive", 99342, 0),
    ("Surface inférieure ou égale à 10 000", 99337, 5),
    ("Prix entre 200 et 15 000", 99314, 23),
    ("Prix au pied carré entre 0,3 et 10", 99285, 29),
    ("Chambres entre 0 et 6", 99278, 7),
    ("Salles de bain entre 0 et 6", 99278, 0),
]

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"


def register_fonts():
    pdfmetrics.registerFont(TTFont("DejaVuSans", str(FONT_DIR / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuSans-Oblique", str(FONT_DIR / "DejaVuSans-Oblique.ttf")))
    pdfmetrics.registerFont(
        TTFont("DejaVuSans-BoldOblique", str(FONT_DIR / "DejaVuSans-BoldOblique.ttf"))
    )
    registerFontFamily(
        "DejaVuSans",
        normal="DejaVuSans",
        bold="DejaVuSans-Bold",
        italic="DejaVuSans-Oblique",
        boldItalic="DejaVuSans-BoldOblique",
    )


def fmt_int(value):
    return f"{int(value):,}".replace(",", " ")


def fmt_dec(value, decimals=2):
    text = f"{float(value):,.{decimals}f}"
    return text.replace(",", " ").replace(".", ",")


def fmt_pct(value, decimals=1):
    return f"{fmt_dec(value, decimals)} %"


def file_size_mb(path):
    if not path.exists():
        return "-"
    return f"{path.stat().st_size / (1024 * 1024):.1f}".replace(".", ",") + " Mo"


def build_styles():
    styles = {}
    styles["Title"] = ParagraphStyle(
        "Title",
        fontName="DejaVuSans-Bold",
        fontSize=24,
        leading=30,
        textColor=PRIMARY,
        spaceAfter=6,
    )
    styles["Subtitle"] = ParagraphStyle(
        "Subtitle",
        fontName="DejaVuSans",
        fontSize=12,
        leading=17,
        textColor=GRAY,
        spaceAfter=12,
    )
    styles["H1"] = ParagraphStyle(
        "H1",
        fontName="DejaVuSans-Bold",
        fontSize=15,
        leading=20,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=8,
    )
    styles["H1NoToc"] = ParagraphStyle(
        "H1NoToc",
        fontName="DejaVuSans-Bold",
        fontSize=15,
        leading=20,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=8,
    )
    styles["H2"] = ParagraphStyle(
        "H2",
        fontName="DejaVuSans-Bold",
        fontSize=11.5,
        leading=16,
        textColor=ACCENT,
        spaceBefore=10,
        spaceAfter=5,
    )
    styles["Body"] = ParagraphStyle(
        "Body",
        fontName="DejaVuSans",
        fontSize=9.8,
        leading=14.5,
        textColor=TEXT,
        alignment=TA_JUSTIFY,
        spaceAfter=7,
    )
    styles["Bullet"] = ParagraphStyle(
        "Bullet",
        fontName="DejaVuSans",
        fontSize=9.6,
        leading=14,
        textColor=TEXT,
        alignment=TA_JUSTIFY,
        leftIndent=6,
    )
    styles["Caption"] = ParagraphStyle(
        "Caption",
        fontName="DejaVuSans-Oblique",
        fontSize=8.2,
        leading=11,
        textColor=GRAY,
        alignment=TA_CENTER,
        spaceBefore=4,
        spaceAfter=10,
    )
    styles["Metric"] = ParagraphStyle(
        "Metric",
        fontName="DejaVuSans",
        fontSize=15,
        leading=19,
        textColor=PRIMARY,
        alignment=TA_CENTER,
    )
    styles["MetricLabel"] = ParagraphStyle(
        "MetricLabel",
        fontName="DejaVuSans",
        fontSize=8.5,
        leading=11,
        textColor=GRAY,
        alignment=TA_CENTER,
    )
    styles["TableCell"] = ParagraphStyle(
        "TableCell",
        fontName="DejaVuSans",
        fontSize=8.3,
        leading=11,
        textColor=TEXT,
        alignment=TA_LEFT,
    )
    styles["TableHeader"] = ParagraphStyle(
        "TableHeader",
        fontName="DejaVuSans-Bold",
        fontSize=8.3,
        leading=11,
        textColor=colors.white,
        alignment=TA_CENTER,
    )
    styles["Note"] = ParagraphStyle(
        "Note",
        fontName="DejaVuSans",
        fontSize=8.2,
        leading=11.5,
        textColor=GRAY,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    styles["TOC1"] = ParagraphStyle(
        "TOC1",
        fontName="DejaVuSans-Bold",
        fontSize=10.2,
        leading=15,
        textColor=PRIMARY,
        spaceBefore=4,
    )
    styles["TOC2"] = ParagraphStyle(
        "TOC2",
        fontName="DejaVuSans",
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor("#333333"),
        leftIndent=16,
    )
    return styles


def cell(text, style):
    return Paragraph(text, style)


def report_image(path, width_cm):
    with PILImage.open(path) as image:
        pixel_width, pixel_height = image.size
    width = width_cm * cm
    height = width * pixel_height / pixel_width
    return Image(str(path), width=width, height=height)


def make_table(rows, col_widths, styles, header=True, font_size=8.3):
    table = Table(rows, colWidths=col_widths, repeatRows=1 if header else 0, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "DejaVuSans"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 3),
        ("TEXTCOLOR", (0, 0), (-1, -1), TEXT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
    ]
    if header:
        style.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "DejaVuSans-Bold"),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ]
        )
        for row in range(1, len(rows)):
            if row % 2 == 0:
                style.append(("BACKGROUND", (0, row), (-1, row), LIGHT_ALT))
    table.setStyle(TableStyle(style))
    return table


def bullet_list(items, styles):
    return ListFlowable(
        [
            ListItem(Paragraph(item, styles["Bullet"]), leftIndent=12)
            for item in items
        ],
        bulletType="bullet",
        start="•",
        leftIndent=14,
        bulletFontName="DejaVuSans",
        bulletFontSize=8,
        spaceBefore=2,
        spaceAfter=6,
    )


def make_figures(cleaned, missing_counts):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font="DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.titleweight": "bold",
            "figure.dpi": 150,
            "savefig.dpi": 150,
        }
    )

    missing_labels = [label for label, _ in missing_counts]
    missing_values = [value for _, value in missing_counts]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    sns.barplot(x=missing_values, y=missing_labels, ax=ax, color=MPL_ACCENT)
    ax.set_xlabel("Nombre de valeurs manquantes")
    ax.set_ylabel("")
    ax.set_title("Valeurs manquantes dans le dataset 100K avant nettoyage")
    max_value = max(missing_values) if missing_values else 1
    for index, value in enumerate(missing_values):
        ax.text(
            value + max_value * 0.01,
            index,
            fmt_int(value),
            va="center",
            fontsize=8,
            color=MPL_PRIMARY,
        )
    ax.set_xlim(0, max_value * 1.18)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_missing.png", bbox_inches="tight")
    plt.close(fig)

    clean_price = cleaned["price"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    sns.histplot(clean_price, bins=55, ax=axes[0], color=MPL_PRIMARY)
    axes[0].axvline(clean_price.median(), color=MPL_ACCENT, linestyle="--", linewidth=1.2)
    axes[0].set_title("Distribution du prix après nettoyage")
    axes[0].set_xlabel("Prix mensuel (USD)")
    axes[0].set_ylabel("Nombre d'annonces")
    axes[0].text(
        0.97,
        0.93,
        f"Médiane : {fmt_int(clean_price.median())} USD\nMoyenne : {fmt_int(clean_price.mean())} USD",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#D0D7E2"),
    )

    sns.boxplot(x=clean_price, ax=axes[1], color=MPL_ACCENT)
    axes[1].set_title("Boîte à moustaches du prix")
    axes[1].set_xlabel("Prix mensuel (USD)")
    axes[1].set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_price.png", bbox_inches="tight")
    plt.close(fig)

    sample = cleaned.sample(min(5000, len(cleaned)), random_state=42)
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    sns.scatterplot(
        data=sample,
        x="square_feet",
        y="price",
        alpha=0.28,
        s=14,
        color=MPL_ACCENT,
        edgecolor="none",
        ax=ax,
    )
    ax.set_xlim(0, sample["square_feet"].quantile(0.99))
    ax.set_ylim(0, sample["price"].quantile(0.99))
    ax.set_xlabel("Surface (pieds carrés)")
    ax.set_ylabel("Prix mensuel (USD)")
    ax.set_title("Relation entre la surface et le prix (échantillon de 5 000 annonces)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_scatter.png", bbox_inches="tight")
    plt.close(fig)

    corr_columns = [
        "price",
        "square_feet",
        "bedrooms",
        "bathrooms",
        "latitude",
        "longitude",
        "amenities_count",
        "total_rooms",
        "city_freq",
        "source_freq",
        "listing_month",
    ]
    corr_columns = [column for column in corr_columns if column in cleaned.columns]
    correlation = cleaned[corr_columns].corr()
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.75},
        ax=ax,
        annot_kws={"fontsize": 7.5},
    )
    ax.set_title("Matrice de corrélation des variables numériques")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_corr.png", bbox_inches="tight")
    plt.close(fig)

    state_stats = cleaned.groupby("state")["price"].agg(["median", "count"])
    state_stats = state_stats[state_stats["count"] >= 100]
    state_stats = state_stats.sort_values("median", ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    sns.barplot(x=state_stats["median"].values, y=state_stats.index, ax=ax, color=MPL_PRIMARY)
    ax.set_xlabel("Prix médian mensuel (USD)")
    ax.set_ylabel("État")
    ax.set_title("Top 12 des États par prix médian (au moins 100 annonces)")
    ax.set_xlim(0, state_stats["median"].max() * 1.38)
    for index, (median, count) in enumerate(zip(state_stats["median"], state_stats["count"])):
        ax.text(
            median + state_stats["median"].max() * 0.02,
            index,
            f"{fmt_int(median)} USD (n={fmt_int(count)})",
            va="center",
            fontsize=7.8,
            color=MPL_PRIMARY,
        )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_states.png", bbox_inches="tight")
    plt.close(fig)

    amenity_columns = [column for column in cleaned.columns if column.startswith("amenity_")]
    amenity_pct = cleaned[amenity_columns].mean().sort_values(ascending=False).head(12) * 100
    amenity_labels = [
        column.replace("amenity_", "").replace("_", " ").title() for column in amenity_pct.index
    ]
    pets = cleaned["pets_allowed"].value_counts()
    fig, axes = plt.subplots(
        1, 2, figsize=(11.0, 4.4), gridspec_kw={"width_ratios": [1.45, 1.0]}
    )
    sns.barplot(x=amenity_pct.values, y=amenity_labels, ax=axes[0], color=MPL_ACCENT)
    axes[0].set_xlabel("Part des annonces (%)")
    axes[0].set_ylabel("")
    axes[0].set_title("Équipements les plus fréquents")
    axes[0].set_xlim(0, amenity_pct.max() * 1.2)
    for index, value in enumerate(amenity_pct.values):
        axes[0].text(
            value + amenity_pct.max() * 0.015,
            index,
            fmt_pct(value),
            va="center",
            fontsize=7.6,
            color=MPL_PRIMARY,
        )
    sns.barplot(x=pets.values, y=pets.index, ax=axes[1], color=MPL_PRIMARY)
    axes[1].set_xlabel("Nombre d'annonces")
    axes[1].set_ylabel("")
    axes[1].set_title("Animaux autorisés")
    for index, value in enumerate(pets.values):
        axes[1].text(
            value + pets.max() * 0.015,
            index,
            fmt_int(value),
            va="center",
            fontsize=7.8,
            color=MPL_PRIMARY,
        )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_amenities_pets.png", bbox_inches="tight")
    plt.close(fig)


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename, styles, **kwargs):
        super().__init__(filename, **kwargs)
        self.styles = styles
        width, height = A4
        cover_frame = Frame(2 * cm, 2 * cm, width - 4 * cm, height - 4 * cm, id="cover")
        body_frame = Frame(2 * cm, 2 * cm, width - 4 * cm, height - 4.2 * cm, id="body")
        self.addPageTemplates(
            [
                PageTemplate(id="Cover", frames=[cover_frame], onPage=self.draw_cover),
                PageTemplate(id="Body", frames=[body_frame], onPage=self.draw_body),
            ]
        )

    def draw_cover(self, canvas, doc):
        width, height = A4
        canvas.saveState()
        canvas.setFillColor(PRIMARY)
        canvas.rect(0, height - 10 * cm, width, 10 * cm, fill=1, stroke=0)
        canvas.setFillColor(ACCENT)
        canvas.rect(0, height - 10 * cm, width, 0.28 * cm, fill=1, stroke=0)
        canvas.setFillColor(colors.white)
        canvas.setFont("DejaVuSans-Bold", 26)
        canvas.drawString(2 * cm, height - 4.2 * cm, "Rapport de prétraitement")
        canvas.setFont("DejaVuSans", 14.5)
        canvas.drawString(2 * cm, height - 5.3 * cm, "Prédiction du prix de location d'un appartement")
        canvas.setFont("DejaVuSans", 10.5)
        canvas.drawString(2 * cm, height - 6.3 * cm, "Projet Machine Learning – Orange Digital Center")
        canvas.drawString(
            2 * cm,
            height - 7.1 * cm,
            "Datasets : apartments_for_rent_classified_10K et 100K",
        )

        canvas.setFillColor(LIGHT_ALT)
        canvas.roundRect(2 * cm, height - 15.2 * cm, width - 4 * cm, 4.4 * cm, 0.25 * cm, fill=1, stroke=0)
        canvas.setFillColor(PRIMARY)
        canvas.setFont("DejaVuSans-Bold", 12)
        canvas.drawString(2.5 * cm, height - 11.6 * cm, "Informations clés")
        canvas.setFillColor(TEXT)
        canvas.setFont("DejaVuSans", 9.8)
        canvas.drawString(
            2.5 * cm,
            height - 12.55 * cm,
            "Responsable du prétraitement : Baye Serigne Seck",
        )
        canvas.drawString(2.5 * cm, height - 13.35 * cm, "Date : 10 novembre 2026")
        canvas.drawString(
            2.5 * cm,
            height - 14.15 * cm,
            "Scripts : src/preprocessing.py et src/generate_report.py",
        )

        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.6)
        canvas.line(2 * cm, 2.8 * cm, width - 2 * cm, 2.8 * cm)
        canvas.setFont("DejaVuSans", 8.5)
        canvas.setFillColor(GRAY)
        canvas.drawString(
            2 * cm,
            2.2 * cm,
            "Rapport généré automatiquement par le script src/generate_report.py.",
        )
        canvas.drawRightString(width - 2 * cm, 2.2 * cm, "Page 1")
        canvas.restoreState()

    def draw_body(self, canvas, doc):
        width, height = A4
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D0D7E2"))
        canvas.setLineWidth(0.5)
        canvas.line(2 * cm, height - 1.6 * cm, width - 2 * cm, height - 1.6 * cm)
        canvas.setFont("DejaVuSans", 7.8)
        canvas.setFillColor(GRAY)
        canvas.drawString(
            2 * cm,
            height - 1.3 * cm,
            "Prétraitement des données – Prédiction du prix de location",
        )
        canvas.drawRightString(width - 2 * cm, height - 1.3 * cm, "Orange Digital Center")
        canvas.line(2 * cm, 1.6 * cm, width - 2 * cm, 1.6 * cm)
        canvas.drawString(2 * cm, 1.1 * cm, "Novembre 2026")
        canvas.drawCentredString(width / 2, 1.1 * cm, f"Page {doc.page}")
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            style_name = flowable.style.name
            if style_name == "H1":
                self.notify("TOCEntry", (0, flowable.getPlainText(), self.page))
            elif style_name == "H2":
                self.notify("TOCEntry", (1, flowable.getPlainText(), self.page))


def build_story(styles, cleaned, features, stats, steps, c10_shape, x10_columns, raw_stats):
    story = []
    story.append(Spacer(1, 1))
    story.append(NextPageTemplate("Body"))
    story.append(PageBreak())

    story.append(Paragraph("Synthèse", styles["H1"]))
    metric_rows = [
        [
            Paragraph(f"<b>{fmt_int(len(cleaned))}</b>", styles["Metric"]),
            Paragraph(f"<b>{features.shape[1]}</b>", styles["Metric"]),
        ],
        [
            Paragraph("annonces nettoyées (100K)", styles["MetricLabel"]),
            Paragraph("variables explicatives", styles["MetricLabel"]),
        ],
        [
            Paragraph("<b>0</b>", styles["Metric"]),
            Paragraph(f"<b>{fmt_int(stats['duplicates_removed'])}</b>", styles["Metric"]),
        ],
        [
            Paragraph("valeur manquante", styles["MetricLabel"]),
            Paragraph("doublons supprimés", styles["MetricLabel"]),
        ],
    ]
    metric_table = Table(metric_rows, colWidths=[8.5 * cm, 8.5 * cm], hAlign="LEFT")
    metric_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.white),
                ("INNERGRID", (0, 0), (-1, -1), 0.6, colors.white),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story.append(metric_table)
    story.append(Spacer(1, 10))

    raw_missing = raw_stats["missing_total"]
    story.append(
        Paragraph(
            f"Le jeu de données 100K contient {fmt_int(raw_stats['rows'])} annonces brutes et "
            f"{fmt_int(raw_missing)} valeurs manquantes. Après nettoyage, {fmt_int(len(cleaned))} "
            f"annonces sont conservées, soit {fmt_pct(len(cleaned) / raw_stats['rows'] * 100)} des données "
            f"initiales. Les {fmt_int(raw_stats['rows'] - len(cleaned))} lignes écartées correspondent à des "
            "doublons, des annonces non résidentielles ou non mensuelles, des prix ou surfaces "
            "incohérents et des valeurs extrêmes. Toutes les valeurs manquantes ont été traitées et "
            "les variables catégorielles ont été encodées pour la modélisation.",
            styles["Body"],
        )
    )

    story.append(Paragraph("Sommaire", styles["H1NoToc"]))
    toc = TableOfContents()
    toc.levelStyles = [styles["TOC1"], styles["TOC2"]]
    toc.dotsMinLevel = 0
    story.append(toc)
    story.append(PageBreak())

    story.append(Paragraph("1. Contexte et objectif", styles["H1"]))
    story.append(
        Paragraph(
            "Ce rapport documente l'étape de prétraitement du projet de prédiction du prix de "
            "location d'un appartement. L'objectif est de transformer les données brutes initiales "
            "en un jeu de données propre, cohérent et directement exploitable par "
            "les modèles de Machine Learning.",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "Le travail couvre la gestion des valeurs manquantes, la suppression des doublons, le "
            "traitement des valeurs aberrantes, l'encodage des variables catégorielles, la création "
            "de variables dérivées et la préparation des variables explicatives X. "
            "Deux jeux de données sont traités : le jeu 10K, utilisé pour les essais rapides, et le "
            "jeu 100K, qui constitue la base principale du projet.",
            styles["Body"],
        )
    )

    story.append(Paragraph("2. Données sources", styles["H2"]))
    story.append(
        Paragraph(
            "Les fichiers sources sont au format CSV, séparés par des points-virgules et encodés "
            "en Latin-1. Le jeu 100K contient 99 492 lignes et 22 colonnes. Le jeu 10K est un "
            "échantillon de 10 000 lignes issu du même schéma.",
            styles["Body"],
        )
    )
    source_rows = [
        ["Dataset", "Fichier source", "Lignes", "Colonnes", "Encodage"],
        ["10K", "apartments_for_rent_classified_10K.csv", "10 000", "22", "Latin-1"],
        ["100K", "apartments_for_rent_classified_100K.csv", "99 492", "22", "Latin-1"],
    ]
    story.append(make_table(source_rows, [2.2 * cm, 7.6 * cm, 2.4 * cm, 2.2 * cm, 2.6 * cm], styles))
    story.append(Paragraph("Tableau 1. Description des fichiers sources.", styles["Caption"]))

    group_rows = [
        ["Groupe", "Variables"],
        ["Identifiant", "id"],
        ["Cible", "price"],
        ["Numériques", "bathrooms, bedrooms, square_feet, latitude, longitude"],
        ["Catégorielles", "category, currency, fee, price_type, has_photo, pets_allowed, cityname, state, source"],
        ["Texte", "title, body, amenities"],
        ["Temporelle", "time (timestamp Unix)"],
    ]
    story.append(make_table(group_rows, [4.0 * cm, 13.0 * cm], styles))
    story.append(Paragraph("Tableau 2. Groupes de variables du fichier brut.", styles["Caption"]))

    story.append(
        report_image(FIGURES_DIR / "fig_missing.png", 16.2)
    )
    story.append(
        Paragraph(
            "Figure 1. Les variables les plus incomplètes sont l'adresse, les animaux autorisés et "
            "les équipements. Les colonnes cityname, state, latitude et longitude ne présentent que "
            "quelques valeurs manquantes.",
            styles["Caption"],
        )
    )

    story.append(Paragraph("3. Méthodologie de prétraitement", styles["H1"]))
    story.append(Paragraph("3.1 Nettoyage de base", styles["H2"]))
    story.append(
        Paragraph(
            "Le nettoyage commence par la suppression des doublons exacts, puis par le filtrage des "
            "annonces qui ne correspondent pas au problème étudié. Seules les annonces de la "
            "catégorie appartement, avec un prix mensuel, un prix strictement positif et une surface "
            "strictement positive, sont conservées. Les valeurs extrêmes sont ensuite retirées selon "
            "les seuils présentés ci-dessous.",
            styles["Body"],
        )
    )
    filter_rows = [["Étape", "Lignes restantes", "Supprimées"]]
    for label, remaining, removed in steps:
        filter_rows.append(
            [label, fmt_int(remaining), "-" if removed is None else fmt_int(removed)]
        )
    story.append(make_table(filter_rows, [9.4 * cm, 3.8 * cm, 3.8 * cm], styles))
    story.append(Paragraph("Tableau 3. Filtres appliqués séquentiellement au dataset 100K.", styles["Caption"]))

    story.append(Paragraph("3.2 Gestion des valeurs manquantes", styles["H2"]))
    story.append(
        Paragraph(
            "Les valeurs manquantes numériques sont imputées par la médiane afin de limiter "
            "l'influence des distributions asymétriques. Les valeurs manquantes géographiques sont "
            "imputées par la médiane de l'État, puis par la médiane globale lorsque l'État est "
            "inconnu. Les variables catégorielles reçoivent une modalité explicite.",
            styles["Body"],
        )
    )
    impute_rows = [
        ["Variable", "Règle appliquée"],
        [
            "amenities",
            "Valeur manquante conservée comme information absente ; création de l'indicateur amenities_provided.",
        ],
        ["bedrooms", "Imputation par la médiane globale."],
        [
            "bathrooms",
            "Imputation par la médiane conditionnelle au nombre de chambres, puis par la médiane globale.",
        ],
        ["cityname, state", "Remplacement par la modalité Unknown."],
        ["latitude, longitude", "Imputation par la médiane de l'État, puis par la médiane globale."],
        ["pets_allowed", "Remplacement par No pets après normalisation des valeurs None."],
        ["has_photo", "Remplacement par No."],
    ]
    impute_table_rows = [[impute_rows[0][0], impute_rows[0][1]]]
    for left, right in impute_rows[1:]:
        impute_table_rows.append([cell(left, styles["TableCell"]), cell(right, styles["TableCell"])])
    header_row = [
        cell(impute_rows[0][0], styles["TableHeader"]),
        cell(impute_rows[0][1], styles["TableHeader"]),
    ]
    impute_table_rows[0] = header_row
    story.append(make_table(impute_table_rows, [4.4 * cm, 12.6 * cm], styles))
    story.append(Paragraph("Tableau 4. Règles d'imputation des valeurs manquantes.", styles["Caption"]))

    story.append(Paragraph("3.3 Traitement des valeurs aberrantes", styles["H2"]))
    story.append(
        Paragraph(
            "Les seuils ci-dessous visent à conserver les annonces plausibles tout en supprimant "
            "les valeurs manifestement erronées ou extrêmes. Le prix au pied carré est utilisé "
            "uniquement comme filtre de cohérence : il est supprimé ensuite pour éviter toute fuite "
            "d'information provenant de la cible.",
            styles["Body"],
        )
    )
    outlier_rows = [
        ["Variable", "Plage conservée"],
        ["Prix mensuel", "200 à 15 000 USD"],
        ["Surface", "Inférieure ou égale à 10 000 pieds carrés"],
        ["Prix au pied carré", "0,3 à 10 USD"],
        ["Chambres", "0 à 6"],
        ["Salles de bain", "0 à 6"],
    ]
    story.append(make_table(outlier_rows, [7.0 * cm, 10.0 * cm], styles))
    story.append(Paragraph("Tableau 5. Seuils de détection des valeurs aberrantes.", styles["Caption"]))

    story.append(Paragraph("3.4 Création de variables", styles["H2"]))
    story.append(
        bullet_list(
            [
                "<b>Équipements :</b> 27 indicateurs binaires (parking, piscine, salle de sport, etc.), "
                "ainsi que amenities_count et amenities_provided.",
                "<b>Animaux :</b> has_cats et has_dogs à partir de la variable pets_allowed.",
                "<b>Dates :</b> listing_date, listing_year, listing_month et listing_dayofweek à partir du timestamp.",
                "<b>Variables dérivées :</b> is_studio et total_rooms (chambres plus salles de bain).",
                "<b>Localisation :</b> latitude, longitude, encodage one-hot de l'État et fréquence de la ville.",
                "<b>Source :</b> encodage par fréquence de la variable source.",
            ],
            styles,
        )
    )

    story.append(Paragraph("3.5 Encodage et standardisation", styles["H2"]))
    story.append(
        Paragraph(
            "Les variables catégorielles à faible cardinalité sont encodées en one-hot, notamment "
            "l'État et le type de photo. La ville, qui possède une forte cardinalité, est encodée par "
            "sa fréquence d'apparition et complétée par la latitude, la longitude et l'État. La source "
            "est également encodée par fréquence. Une version standardisée des variables numériques "
            "est fournie pour les modèles sensibles à l'échelle, tandis que les indicateurs binaires "
            "restent inchangés.",
            styles["Body"],
        )
    )

    story.append(Paragraph("4. Résultats du prétraitement", styles["H1"]))
    story.append(Paragraph("4.1 Bilan du nettoyage", styles["H2"]))
    summary_rows = [
        ["Indicateur", "Avant", "Après"],
        ["Lignes", fmt_int(raw_stats["rows"]), fmt_int(len(cleaned))],
        ["Colonnes après nettoyage", str(raw_stats["columns"]), str(cleaned.shape[1])],
        ["Variables explicatives X", "-", str(features.shape[1])],
        ["Valeurs manquantes", fmt_int(raw_missing), "0"],
        ["Doublons exacts", fmt_int(stats["duplicates_removed"]), "0"],
        ["États distincts", fmt_int(raw_stats["states"]), fmt_int(cleaned["state"].nunique())],
        ["Villes distinctes", fmt_int(raw_stats["cities"]), fmt_int(cleaned["cityname"].nunique())],
    ]
    story.append(make_table(summary_rows, [7.0 * cm, 5.0 * cm, 5.0 * cm], styles))
    story.append(Paragraph("Tableau 6. Comparaison avant et après prétraitement (dataset 100K).", styles["Caption"]))
    story.append(
        Paragraph(
            "Note : la modalité Unknown ajoutée pour les valeurs manquantes de cityname et state "
            "explique l'augmentation du nombre d'États et de villes distincts après nettoyage.",
            styles["Caption"],
        )
    )

    story.append(Paragraph("4.2 Distribution du prix", styles["H2"]))
    story.append(report_image(FIGURES_DIR / "fig_price.png", 17.0))
    story.append(
        Paragraph(
            "Figure 2. Distribution du prix après nettoyage. La distribution reste asymétrique à "
            "droite, avec une médiane de 1 350 USD et une moyenne de 1 520 USD.",
            styles["Caption"],
        )
    )

    story.append(Paragraph("4.3 Relation entre surface et prix", styles["H2"]))
    story.append(report_image(FIGURES_DIR / "fig_scatter.png", 16.2))
    story.append(
        Paragraph(
            "Figure 3. Le prix tend à augmenter avec la surface, mais la dispersion reste importante "
            "à surface égale, ce qui justifie l'utilisation d'autres variables explicatives.",
            styles["Caption"],
        )
    )

    story.append(Paragraph("4.4 Corrélations", styles["H2"]))
    story.append(report_image(FIGURES_DIR / "fig_corr.png", 15.4))
    story.append(
        Paragraph(
            "Figure 4. La surface, le nombre de chambres et le nombre de salles de bain présentent "
            "les corrélations positives les plus fortes avec le prix. La latitude et la longitude "
            "apportent une information complémentaire sur la localisation.",
            styles["Caption"],
        )
    )

    story.append(Paragraph("4.5 Analyse par État", styles["H2"]))
    story.append(report_image(FIGURES_DIR / "fig_states.png", 16.2))
    story.append(
        Paragraph(
            "Figure 5. Les prix médians varient fortement selon l'État. Cette variable est donc "
            "pertinente pour la modélisation et a été encodée en one-hot.",
            styles["Caption"],
        )
    )

    story.append(Paragraph("4.6 Équipements et animaux", styles["H2"]))
    story.append(report_image(FIGURES_DIR / "fig_amenities_pets.png", 17.0))
    story.append(
        Paragraph(
            "Figure 6. Le parking, la piscine et la salle de sport sont les équipements les plus "
            "fréquents. La majorité des annonces ne précise pas d'animaux autorisés.",
            styles["Caption"],
        )
    )

    story.append(Paragraph("4.7 Statistiques descriptives", styles["H2"]))
    numeric_columns = ["price", "square_feet", "bedrooms", "bathrooms", "amenities_count", "total_rooms"]
    describe = cleaned[numeric_columns].describe().T
    describe_rows = [["Variable", "Moyenne", "Médiane", "Écart-type", "Min", "Max"]]
    for column in numeric_columns:
        row = describe.loc[column]
        decimals = 0 if column in {"price", "square_feet"} else 2
        describe_rows.append(
            [
                column,
                fmt_dec(row["mean"], decimals),
                fmt_dec(row["50%"], decimals),
                fmt_dec(row["std"], decimals),
                fmt_dec(row["min"], decimals),
                fmt_dec(row["max"], decimals),
            ]
        )
    story.append(make_table(describe_rows, [4.6 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.4 * cm, 2.5 * cm], styles))
    story.append(Paragraph("Tableau 7. Statistiques descriptives des principales variables après nettoyage.", styles["Caption"]))

    story.append(Paragraph("5. Variables disponibles pour la modélisation", styles["H1"]))
    story.append(
        Paragraph(
            "La matrice X contient uniquement des variables numériques et ne contient pas la cible. "
            "Les variables sont réparties en trois groupes : les variables numériques continues, les "
            "indicateurs binaires et les modalités one-hot des États.",
            styles["Body"],
        )
    )
    feature_rows = [
        ["Groupe", "Nombre", "Exemples"],
        ["Variables numériques", "12", "square_feet, bathrooms, latitude, city_freq, source_freq"],
        ["Indicateurs binaires", "34", "amenity_parking, has_cats, is_studio, photo_Yes"],
        ["États en one-hot", "51", "state_CA, state_NY, state_TX"],
        ["Total", "97", "X_100K.csv.gz"],
    ]
    story.append(make_table(feature_rows, [5.4 * cm, 2.4 * cm, 9.2 * cm], styles))
    story.append(Paragraph("Tableau 8. Composition de la matrice des variables explicatives (100K).", styles["Caption"]))
    story.append(
        Paragraph(
            f"Pour le jeu 10K, la matrice X contient {x10_columns} variables et {fmt_int(c10_shape[0])} "
            "lignes. Les fichiers correspondants suivent la même organisation que ceux du jeu 100K.",
            styles["Body"],
        )
    )

    story.append(Paragraph("6. Fichiers générés", styles["H1"]))
    story.append(
        Paragraph(
            "Les livrables du prétraitement sont les matrices de variables explicatives X. Les "
            "fichiers de données nettoyées complètes et les fichiers de cible y ne font pas partie "
            "des livrables. Les matrices sont compressées au format .csv.gz afin de réduire l'espace "
            "occupé et se lisent directement avec pandas.",
            styles["Body"],
        )
    )
    file_rows = [
        ["Fichier", "Contenu", "Dimensions", "Taille"],
        [
            "X_100K.csv.gz",
            cell("Variables explicatives encodées (100K)", styles["TableCell"]),
            f"{fmt_int(features.shape[0])} x {features.shape[1]}",
            file_size_mb(PROCESSED_DIR / "X_100K.csv.gz"),
        ],
        [
            "X_100K_scaled.csv.gz",
            cell("Variables explicatives standardisées (100K)", styles["TableCell"]),
            f"{fmt_int(features.shape[0])} x {features.shape[1]}",
            file_size_mb(PROCESSED_DIR / "X_100K_scaled.csv.gz"),
        ],
        [
            "X_10K.csv.gz",
            cell("Variables explicatives encodées (10K)", styles["TableCell"]),
            f"{fmt_int(c10_shape[0])} x {x10_columns}",
            file_size_mb(PROCESSED_DIR / "X_10K.csv.gz"),
        ],
        [
            "X_10K_scaled.csv.gz",
            cell("Variables explicatives standardisées (10K)", styles["TableCell"]),
            f"{fmt_int(c10_shape[0])} x {x10_columns}",
            file_size_mb(PROCESSED_DIR / "X_10K_scaled.csv.gz"),
        ],
    ]
    header = [cell(value, styles["TableHeader"]) for value in file_rows[0]]
    body_rows = [header]
    for row in file_rows[1:]:
        body_rows.append(
            [
                row[0],
                row[1],
                row[2],
                row[3],
            ]
        )
    story.append(make_table(body_rows, [5.0 * cm, 6.2 * cm, 3.2 * cm, 2.6 * cm], styles))
    story.append(
        Paragraph(
            "Tableau 9. Fichiers de variables explicatives produits par le script de prétraitement.",
            styles["Caption"],
        )
    )
    story.append(
        Paragraph(
            "Exemple de lecture : pd.read_csv('data/processed/X_100K.csv.gz').",
            styles["Note"],
        )
    )

    story.append(Paragraph("7. Limites et recommandations", styles["H1"]))
    story.append(
        bullet_list(
            [
                "<b>Équipements manquants :</b> une valeur manquante ne signifie pas que l'équipement est absent. "
                "L'indicateur amenities_provided permet de distinguer les deux situations.",
                "<b>Forte cardinalité :</b> la ville est encodée par fréquence et complétée par la latitude, "
                "la longitude et l'État. Un encodage par cible pourra être testé avec validation croisée.",
                "<b>Seuils d'aberrations :</b> les seuils sont fixes et adaptés au marché américain. Ils pourront "
                "être ajustés selon les objectifs métier.",
                "<b>Asymétrie de la cible :</b> une transformation logarithmique du prix pourra être testée "
                "lors de la modélisation.",
                "<b>Standardisation :</b> la version standardisée est calculée sur l'ensemble des données. Pour une "
                "évaluation rigoureuse, le scaler devra être ajusté uniquement sur l'ensemble d'entraînement.",
                "<b>Fuite de cible :</b> le prix au pied carré a été utilisé uniquement pour filtrer les incohérences, "
                "puis supprimé de la matrice X.",
                "<b>Choix des modèles :</b> les modèles à arbres peuvent utiliser X_100K.csv.gz directement. Les "
                "modèles linéaires utiliseront plutôt X_100K_scaled.csv.gz.",
            ],
            styles,
        )
    )

    story.append(Paragraph("8. Conclusion", styles["H1"]))
    story.append(
        Paragraph(
            "Le prétraitement aboutit à un jeu de données propre, sans valeur manquante, avec des "
            "variables catégorielles encodées et des variables dérivées pertinentes. Le jeu 100K "
            "conserve 99 278 annonces sur 99 492, soit un taux de conservation très élevé. Les fichiers "
            "générés sont prêts à être utilisés pour la séparation entraînement/test, l'entraînement "
            "des modèles de régression et l'évaluation des performances.",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            "Les prochaines étapes consistent à effectuer le découpage entraînement/test, à entraîner "
            "plusieurs modèles de régression, à comparer leurs performances avec les métriques MAE, "
            "MSE, RMSE et R², puis à identifier le modèle le plus performant.",
            styles["Body"],
        )
    )

    return story


def main():
    register_fonts()
    styles = build_styles()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    cleaned = pd.read_csv(PROCESSED_DIR / "apartments_clean_100K.csv")
    features, _ = pp.build_feature_matrix(cleaned)

    cleaned10 = pd.read_csv(PROCESSED_DIR / "apartments_clean_10K.csv")
    features10, _ = pp.build_feature_matrix(cleaned10)
    c10_shape = cleaned10.shape
    x10_columns = features10.shape[1]

    stats = {"duplicates_removed": RAW_STATS["duplicates"]}
    steps = FILTER_STEPS
    make_figures(cleaned, RAW_STATS["missing"])

    story = build_story(
        styles,
        cleaned,
        features,
        stats,
        steps,
        c10_shape,
        x10_columns,
        RAW_STATS,
    )

    doc = ReportDocTemplate(
        str(PDF_PATH),
        styles,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2.2 * cm,
        bottomMargin=2 * cm,
        title="Rapport de prétraitement - Prédiction du prix de location",
        author="Baye Serigne Seck",
        subject="Prétraitement des données immobilières",
    )
    doc.multiBuild(story)
    print(f"PDF généré : {PDF_PATH}")


if __name__ == "__main__":
    main()
