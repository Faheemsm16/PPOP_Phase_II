# utils/reporting.py
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import io

def build_pdf_report(patient_meta, snapshot, horizon, enforce_decay,
                     trend_png_bytes=None, shap_png_bytes=None, notes=None):
    """
    Build a PDF report with patient inputs, predictions, risk assessment,
    and optionally embed the prediction trend and SHAP explainability plots.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # ---------------- Title ----------------
    story.append(Paragraph("Hemophilia Decision Support Report", styles["Title"]))
    story.append(Spacer(1, 12))

    # ---------------- Patient Info ----------------
    story.append(Paragraph("<b>Patient Profile</b>", styles["Heading2"]))
    patient_table_data = [
        ["Name", patient_meta.get("patient_name", "NA")],
        ["Age", patient_meta.get("age", "NA")],
        ["Sex", patient_meta.get("sex", "NA")],
        ["Weight (kg)", patient_meta.get("weight_kg", "NA")],
        ["Height (cm)", patient_meta.get("height_cm", "NA")],
    ]
    story.append(Table(patient_table_data, hAlign="LEFT", style=[
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
    ]))
    story.append(Spacer(1, 12))

    # ---------------- Hemophilia Profile ----------------
    story.append(Paragraph("<b>Hemophilia Profile</b>", styles["Heading2"]))
    hemo_table_data = [
        ["Factor Type", patient_meta.get("factor_type", "NA")],
        ["Severity", patient_meta.get("severity", "NA")],
        ["Inhibitor", patient_meta.get("inhibitor", "NA")],
    ]
    story.append(Table(hemo_table_data, hAlign="LEFT", style=[
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
    ]))
    story.append(Spacer(1, 12))

    # ---------------- Infusion Details ----------------
    story.append(Paragraph("<b>Infusion Details</b>", styles["Heading2"]))
    infusion_table_data = [
        ["Brand", patient_meta.get("infusion_brand", "NA")],
        ["Route", patient_meta.get("route", "NA")],
        ["Dose (IU)", patient_meta.get("infusion_dose_IU", "NA")],
        ["Time since last infusion (hr)", patient_meta.get("time_since_last_infusion_hr", "NA")],
    ]
    story.append(Table(infusion_table_data, hAlign="LEFT", style=[
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.black),
    ]))
    story.append(Spacer(1, 12))

    # ---------------- Predictions ----------------
    story.append(Paragraph("<b>Predictions</b>", styles["Heading2"]))
    story.append(Paragraph(f"Predicted Factor Level (Now): <b>{snapshot.get('factor_now', 0):.2f} IU/dL</b>", styles["Normal"]))
    story.append(Paragraph(f"Bleed Risk: <b>{snapshot.get('risk_label', 'NA')}</b> "
                           f"(p = {snapshot.get('risk_p', 0)*100:.1f}%)", styles["Normal"]))
    story.append(Paragraph(f"Prediction Horizon: {horizon} hours", styles["Normal"]))
    story.append(Paragraph(f"Decay Assumption: {'Monotonic PK decay' if enforce_decay else 'Hybrid PK+LSTM'}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # ---------------- Prediction Graph ----------------
    if trend_png_bytes:
        story.append(Paragraph("<b>Prediction Trend</b>", styles["Heading2"]))
        img = Image(io.BytesIO(trend_png_bytes))
        img._restrictSize(450, 250)
        story.append(img)
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "Blue curve shows predicted decline in factor level. "
            "Red line = safety threshold (50 IU/dL). "
            "When the blue curve falls below red, risk of bleeding rises.",
            styles["Normal"]
        ))
        story.append(Spacer(1, 12))


    # ---------------- Explainability ----------------
    if shap_png_bytes:
        story.append(Paragraph("<b>Explainability (SHAP)</b>", styles["Heading2"]))
        img = Image(io.BytesIO(shap_png_bytes))
        img._restrictSize(450, 250)
        story.append(img)
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "Bars above zero indicate features that increased bleed risk. "
            "Bars below zero indicate features that reduced risk. "
            "Longer bars mean stronger influence on prediction.",
            styles["Normal"]
        ))
        story.append(Spacer(1, 12))

    # ---------------- Notes ----------------
    if notes:
        story.append(Paragraph("<b>Clinician Notes</b>", styles["Heading2"]))
        story.append(Paragraph(notes, styles["Normal"]))
        story.append(Spacer(1, 12))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer