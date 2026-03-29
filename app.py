import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, render_template_string, jsonify, send_file
from sentence_transformers import SentenceTransformer
from catboost import CatBoostClassifier, Pool
import json
import time
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.platypus.flowables import Flowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics import renderPDF
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import base64

# --- 1. Load Model and Assets ---
ASSETS_DIR = "."
print("Loading model assets from the current folder...")
try:
    model = CatBoostClassifier()
    model.load_model(os.path.join(ASSETS_DIR, "catboost_hybrid.cbm"))
    assets = joblib.load(os.path.join(ASSETS_DIR, "assets.pkl"))
    st_model = SentenceTransformer(assets["embed_model_name"])
    print("✅ Assets loaded successfully.")
except Exception as e:
    print(f"❌ ERROR: Could not load model files. Make sure 'catboost_hybrid.cbm' and 'assets.pkl' are in the script's folder.")
    exit()

# --- 2. ML Prediction Logic ---
def analyze_cookie_data(df: pd.DataFrame):
    """Runs the CatBoost model to get a risk probability for each cookie."""
    rename_map = {"name": "Cookie / Data Key name", "domain": "Domain", "hostOnly": "Wildcard match"}
    df.rename(columns=rename_map, inplace=True)
    if "Wildcard match" in df.columns:
        df["Wildcard match"] = ~df["Wildcard match"].astype(bool)
    else:
        df["Wildcard match"] = False
    if "retention_hours" not in df.columns:
        if "expirationDate" in df.columns:
            df["retention_hours"] = (pd.to_numeric(df["expirationDate"], errors="coerce") - time.time()).fillna(0) / 3600
            df.loc[df["retention_hours"] < 0, "retention_hours"] = 0
        else:
            df["retention_hours"] = 0.0
    if "Platform" not in df.columns: df["Platform"] = "unknown"
    if "Description" not in df.columns: df["Description"] = ""
    text_cols = ["Platform", "Cookie / Data Key name", "Domain", "Description"]
    df["combined_text"] = df[text_cols].fillna("").agg(" ".join, axis=1)
    embeddings = st_model.encode(df["combined_text"].tolist(), normalize_embeddings=True)
    emb_df = pd.DataFrame(embeddings, columns=assets["emb_cols"], index=df.index)
    X_full = pd.concat([emb_df, df[assets["numeric_cols"] + assets["categorical_cols"]]], axis=1)
    X_full = pd.get_dummies(X_full, columns=assets["categorical_cols"], dtype=float)
    X_reordered = X_full.reindex(columns=assets["feature_order"], fill_value=0.0)
    pool = Pool(X_reordered)
    probabilities = model.predict_proba(pool)[:, 1]
    return probabilities

# --- 3. Weighted Rule-Based Scoring Engine ---
WEIGHTS = {
    "ml_model_score": 40, "http_only_flag": 10, "secure_flag": 8,
    "same_site_policy": 8, "secure_prefixes": 7, "http_strict_transport_security": 6,
    "content_security_policy": 6, "third_party_status": 6, "is_persistent": 5,
    "subresource_integrity": 4, "domain_scope": 2, "path_scope": 2,
}

CRITERIA_DESCRIPTIONS = {
    "ml_model_score": "Machine learning model confidence score based on cookie characteristics",
    "http_only_flag": "Prevents client-side scripts from accessing the cookie (XSS protection)",
    "secure_flag": "Ensures cookie is only sent over HTTPS connections",
    "same_site_policy": "Controls when cookies are sent with cross-site requests (CSRF protection)",
    "secure_prefixes": "Cookie names starting with __Host- or __Secure- for additional security",
    "http_strict_transport_security": "Enforces HTTPS connections for the domain",
    "content_security_policy": "Prevents XSS attacks by restricting resources browser can load",
    "third_party_status": "Whether cookie is first-party (same domain) or third-party",
    "is_persistent": "Session cookies are safer than persistent cookies with long expiration",
    "subresource_integrity": "Ensures resources haven't been tampered with",
    "domain_scope": "Host-only cookies are more secure than domain-wide cookies",
    "path_scope": "Cookies scoped to specific paths limit exposure"
}

def evaluate_cookie_safety(cookie, page_context):
    """Evaluates a single cookie against the rule-based criteria."""
    scores = {}
    scores["http_only_flag"] = WEIGHTS["http_only_flag"] if cookie.get("httpOnly", False) else 0
    scores["secure_flag"] = WEIGHTS["secure_flag"] if cookie.get("secure", False) else 0
    samesite = cookie.get("sameSite", "none").lower()
    if samesite == "strict": scores["same_site_policy"] = WEIGHTS["same_site_policy"]
    elif samesite == "lax": scores["same_site_policy"] = WEIGHTS["same_site_policy"] * 0.7
    else: scores["same_site_policy"] = 0
    name = cookie.get("name", "")
    if name.startswith("__Host-"): scores["secure_prefixes"] = WEIGHTS["secure_prefixes"]
    elif name.startswith("__Secure-"): scores["secure_prefixes"] = WEIGHTS["secure_prefixes"] * 0.6
    else: scores["secure_prefixes"] = 0
    page_domain = page_context.get("hostname", "")
    cookie_domain = cookie.get("domain", "")
    scores["third_party_status"] = WEIGHTS["third_party_status"] if page_domain in cookie_domain or cookie.get("hostOnly") else 0
    scores["is_persistent"] = WEIGHTS["is_persistent"] if cookie.get("session", True) else 0
    scores["domain_scope"] = WEIGHTS["domain_scope"] if cookie.get("hostOnly", True) else 0
    scores["path_scope"] = WEIGHTS["path_scope"] if cookie.get("path", "/") != "/" else 0
    scores["content_security_policy"] = WEIGHTS["content_security_policy"] if page_context.get("hasCSP", False) else 0
    scores["http_strict_transport_security"] = WEIGHTS["http_strict_transport_security"] if page_context.get("hasHSTS", False) else 0
    scores["subresource_integrity"] = WEIGHTS["subresource_integrity"] * page_context.get("sriCoverage", 0)
    return scores

def format_criteria_name(name):
    """Makes the criteria names user-friendly for display."""
    return name.replace("_", " ").title()

# --- 4. PDF Report Generation ---
class SecurityScoreMeter(Flowable):
    """Custom flowable for security score meter"""
    def __init__(self, score, width=400, height=80):
        Flowable.__init__(self)
        self.score = score
        self.width = width
        self.height = height

    def draw(self):
        canvas = self.canv
        # Draw background
        canvas.setFillColor(colors.lightgrey)
        canvas.roundRect(0, 0, self.width, self.height, 10, fill=1)
        
        # Draw progress
        progress_width = (self.score / 100) * self.width
        if self.score >= 75:
            color = colors.green
        elif self.score >= 50:
            color = colors.orange
        else:
            color = colors.red
            
        canvas.setFillColor(color)
        canvas.roundRect(0, 0, progress_width, self.height, 10, fill=1)
        
        # Draw text
        canvas.setFillColor(colors.black)
        canvas.setFont("Helvetica-Bold", 16)
        text = f"Overall Security Score: {self.score:.1f}/100"
        text_width = canvas.stringWidth(text, "Helvetica-Bold", 16)
        canvas.drawString((self.width - text_width) / 2, self.height / 2 - 8, text)

def generate_pdf_report(analysis_data, cookies_data, filename="cookie_security_report.pdf"):
    """Generate a professional PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='CoverTitle',
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        alignment=1,
        spaceAfter=30
    ))
    styles.add(ParagraphStyle(
        name='SectionTitle',
        fontName='Helvetica-Bold',
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12
    ))
    styles.add(ParagraphStyle(
        name='SubsectionTitle',
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=6
    ))
    
    story = []
    
    # Cover Page
    cover_elements = []
    cover_elements.append(Spacer(1, 2*inch))
    
    # Title
    title_style = ParagraphStyle(
        name='CoverTitle',
        fontName='Helvetica-Bold',
        fontSize=28,
        textColor=colors.HexColor('#2c3e50'),
        alignment=1,
        spaceAfter=20
    )
    cover_elements.append(Paragraph("COOKIE SECURITY ANALYSIS REPORT", title_style))
    
    # Subtitle
    subtitle_style = ParagraphStyle(
        name='CoverSubtitle',
        fontName='Helvetica',
        fontSize=16,
        textColor=colors.HexColor('#7f8c8d'),
        alignment=1,
        spaceAfter=40
    )
    cover_elements.append(Paragraph("Comprehensive Security Assessment & Recommendations", subtitle_style))
    
    # Report Details
    details_style = ParagraphStyle(
        name='CoverDetails',
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#95a5a6'),
        alignment=1
    )
    cover_elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", details_style))
    cover_elements.append(Paragraph(f"Total Cookies Analyzed: {len(cookies_data)}", details_style))
    
    cover_elements.append(Spacer(1, 3*inch))
    
    # Confidential Notice
    confidential_style = ParagraphStyle(
        name='Confidential',
        fontName='Helvetica-Oblique',
        fontSize=9,
        textColor=colors.HexColor('#e74c3c'),
        alignment=1
    )
    cover_elements.append(Paragraph("CONFIDENTIAL - FOR INTERNAL USE ONLY", confidential_style))
    
    story.extend(cover_elements)
    story.append(PageBreak())
    
    # Table of Contents
    story.append(Paragraph("Table of Contents", styles['SectionTitle']))
    story.append(Spacer(1, 0.2*inch))
    
    toc_items = [
        "1. Executive Summary",
        "2. Security Overview",
        "3. Risk Distribution Analysis",
        "4. Security Features Compliance",
        "5. Detailed Cookie Analysis",
        "6. Recommendations & Action Plan"
    ]
    
    for item in toc_items:
        story.append(Paragraph(item, styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
    
    story.append(PageBreak())
    
    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary", styles['SectionTitle']))
    story.append(Spacer(1, 0.2*inch))
    
    exec_summary = f"""
    This comprehensive security analysis examined {len(cookies_data)} cookies to assess their security posture 
    and identify potential vulnerabilities. The assessment combines machine learning predictions with rule-based 
    security evaluation to provide a holistic view of cookie security.
    """
    story.append(Paragraph(exec_summary, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Key Metrics Table
    metrics_data = [
        ['Metric', 'Value', 'Status'],
        ['Overall Security Score', f"{analysis_data['average_score']:.1f}/100", 
         'Excellent' if analysis_data['average_score'] >= 80 else 'Good' if analysis_data['average_score'] >= 60 else 'Needs Improvement'],
        ['Total Cookies', str(len(cookies_data)), 'Analyzed'],
        ['Safe Cookies', str(analysis_data['safe_cookies']), f"{(analysis_data['safe_cookies']/len(cookies_data)*100):.1f}%"],
        ['Risky Cookies', str(analysis_data['risky_cookies']), f"{(analysis_data['risky_cookies']/len(cookies_data)*100):.1f}%"],
        ['Analysis Date', datetime.now().strftime('%Y-%m-%d'), 'Completed']
    ]
    
    metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 2*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
    ]))
    story.append(metrics_table)
    
    story.append(PageBreak())
    
    # 2. Security Overview
    story.append(Paragraph("2. Security Overview", styles['SectionTitle']))
    story.append(Spacer(1, 0.2*inch))
    
    # Security Score Meter
    score_meter = SecurityScoreMeter(analysis_data['average_score'], width=400, height=60)
    story.append(score_meter)
    story.append(Spacer(1, 0.3*inch))
    
    overview_text = f"""
    The overall security score of {analysis_data['average_score']:.1f}/100 indicates the collective security posture 
    of all analyzed cookies. This score is calculated based on multiple security criteria including encryption, 
    access controls, and compliance with security best practices.
    """
    story.append(Paragraph(overview_text, styles['Normal']))
    
    story.append(PageBreak())
    
    # 3. Risk Distribution Analysis
    story.append(Paragraph("3. Risk Distribution Analysis", styles['SectionTitle']))
    story.append(Spacer(1, 0.2*inch))
    
    # Create risk distribution chart
    drawing = Drawing(400, 200)
    pie = Pie()
    pie.x = 150
    pie.y = 50
    pie.width = 150
    pie.height = 150
    pie.data = [analysis_data['safe_cookies'], analysis_data['medium_risk_cookies'], analysis_data['risky_cookies']]
    pie.labels = ['Safe', 'Medium Risk', 'High Risk']
    pie.slices.strokeWidth = 1
    pie.slices[0].fillColor = colors.green
    pie.slices[1].fillColor = colors.orange
    pie.slices[2].fillColor = colors.red
    drawing.add(pie)
    story.append(drawing)
    
    risk_text = f"""
    The risk distribution shows how cookies are categorized based on their security characteristics:
    
    • Safe Cookies ({analysis_data['safe_cookies']}): Properly configured with security flags and best practices
    • Medium Risk ({analysis_data['medium_risk_cookies']}): Missing some security features but not critical
    • High Risk ({analysis_data['risky_cookies']}): Missing essential security controls requiring immediate attention
    """
    story.append(Paragraph(risk_text, styles['Normal']))
    
    story.append(PageBreak())
    
    # 4. Security Features Compliance
    story.append(Paragraph("4. Security Features Compliance", styles['SectionTitle']))
    story.append(Spacer(1, 0.2*inch))
    
    # Security features table
    security_data = [
        ['Security Feature', 'Implementation Rate', 'Importance'],
        ['HTTPS Only (Secure Flag)', f"{analysis_data['security_flags']['Secure']:.1f}%", 'Critical'],
        ['HTTPOnly Flag', f"{analysis_data['security_flags']['HttpOnly']:.1f}%", 'High'],
        ['SameSite Policy', f"{analysis_data['security_flags']['SameSite']:.1f}%", 'High'],
        ['Session Cookies', f"{(analysis_data['lifetime_stats']['Session']/len(cookies_data)*100):.1f}%", 'Medium']
    ]
    
    security_table = Table(security_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    security_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6'))
    ]))
    story.append(security_table)
    story.append(Spacer(1, 0.3*inch))
    
    compliance_text = """
    Security feature compliance indicates how well cookies implement essential security controls. 
    Higher implementation rates correlate with better protection against common web vulnerabilities 
    like cross-site scripting (XSS) and cross-site request forgery (CSRF).
    """
    story.append(Paragraph(compliance_text, styles['Normal']))
    
    story.append(PageBreak())
    
    # 5. Detailed Cookie Analysis (Top 20 for brevity)
    story.append(Paragraph("5. Detailed Cookie Analysis", styles['SectionTitle']))
    story.append(Spacer(1, 0.2*inch))
    
    # Sample of cookies for the report
    sample_cookies = cookies_data[:20]  # Show first 20 cookies
    
    cookie_headers = ['Cookie Name', 'Domain', 'Risk Level', 'Security Score']
    cookie_rows = [cookie_headers]
    
    for cookie in sample_cookies:
        risk_level = "High" if cookie['ml_prediction'] == 'Risky' else "Low"
        score_color = "🟢" if cookie['final_score'] >= 75 else "🟡" if cookie['final_score'] >= 50 else "🔴"
        cookie_rows.append([
            cookie['name'][:30] + '...' if len(cookie['name']) > 30 else cookie['name'],
            cookie['domain'][:25] + '...' if len(cookie['domain']) > 25 else cookie['domain'],
            risk_level,
            f"{score_color} {cookie['final_score']:.1f}"
        ])
    
    cookie_table = Table(cookie_rows, colWidths=[1.8*inch, 1.8*inch, 1*inch, 1.2*inch])
    cookie_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ffffff')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
    ]))
    story.append(cookie_table)
    
    if len(cookies_data) > 20:
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(f"Note: Showing 20 of {len(cookies_data)} total cookies. Full analysis available in digital format.", 
                             styles['Italic']))
    
    story.append(PageBreak())
    
    # 6. Recommendations & Action Plan
    story.append(Paragraph("6. Recommendations & Action Plan", styles['SectionTitle']))
    story.append(Spacer(1, 0.2*inch))
    
    recommendations = [
        {
            'priority': 'HIGH',
            'action': 'Implement Secure Flag',
            'description': 'Ensure all cookies are marked with Secure flag to prevent transmission over unencrypted connections',
            'impact': 'Prevents session hijacking'
        },
        {
            'priority': 'HIGH',
            'action': 'Enable HTTPOnly Flag',
            'description': 'Set HTTPOnly flag on sensitive cookies to prevent client-side script access',
            'impact': 'Mitigates XSS attacks'
        },
        {
            'priority': 'MEDIUM',
            'action': 'Configure SameSite Policy',
            'description': 'Implement SameSite=Lax or SameSite=Strict to prevent CSRF attacks',
            'impact': 'Reduces CSRF vulnerability'
        },
        {
            'priority': 'MEDIUM',
            'action': 'Review Cookie Lifetimes',
            'description': 'Implement appropriate expiration times and prefer session cookies for sensitive data',
            'impact': 'Reduces attack surface'
        }
    ]
    
    for rec in recommendations:
        # Priority indicator
        priority_color = colors.red if rec['priority'] == 'HIGH' else colors.orange if rec['priority'] == 'MEDIUM' else colors.blue
        priority_style = ParagraphStyle(
            name='Priority',
            fontName='Helvetica-Bold',
            fontSize=10,
            textColor=priority_color,
            backColor=colors.white,
            borderPadding=5,
            borderColor=priority_color,
            borderWidth=1
        )
        
        story.append(Paragraph(f"Priority: {rec['priority']}", priority_style))
        story.append(Paragraph(f"Action: {rec['action']}", styles['SubsectionTitle']))
        story.append(Paragraph(f"Description: {rec['description']}", styles['Normal']))
        story.append(Paragraph(f"Security Impact: {rec['impact']}", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
    
    # Conclusion
    conclusion_text = f"""
    This analysis provides a comprehensive overview of cookie security posture. Immediate attention should be given 
    to high-risk cookies and implementation of critical security flags. Regular security audits and continuous 
    monitoring are recommended to maintain a strong security posture.
    
    For questions or additional analysis, please contact your security team.
    """
    story.append(Paragraph(conclusion_text, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- 5. Flask Web Application ---
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ANALYZER_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cookie Security Analyzer Pro</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Exo+2:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --neon-blue: #00f3ff;
            --neon-purple: #b300ff;
            --neon-pink: #ff00c8;
            --neon-green: #00ff9d;
            --dark-bg: #0a0a16;
            --dark-card: rgba(18, 18, 40, 0.7);
            --glass-border: rgba(255, 255, 255, 0.1);
            --text-primary: #e0e0ff;
            --text-secondary: #a0a0c0;
            --shadow-glow-blue: 0 0 15px rgba(0, 243, 255, 0.7);
            --shadow-glow-purple: 0 0 15px rgba(179, 0, 255, 0.7);
            --gradient-primary: linear-gradient(135deg, var(--neon-blue) 0%, var(--neon-purple) 100%);
            --gradient-secondary: linear-gradient(135deg, var(--neon-pink) 0%, var(--neon-purple) 100%);
            --gradient-success: linear-gradient(135deg, var(--neon-green) 0%, #00cc7a 100%);
            --gradient-warning: linear-gradient(135deg, #ffcc00 0%, #ff9900 100%);
            --gradient-danger: linear-gradient(135deg, #ff3366 0%, #cc0066 100%);
            --transition-slow: all 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            --transition-fast: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Exo 2', sans-serif;
            background-color: var(--dark-bg);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
            overflow-x: hidden;
            position: relative;
        }

        /* Animated background */
        .cyber-bg {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            overflow: hidden;
        }

        .grid-lines {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                linear-gradient(rgba(0, 243, 255, 0.1) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 243, 255, 0.1) 1px, transparent 1px);
            background-size: 50px 50px;
            animation: gridMove 20s linear infinite;
        }

        @keyframes gridMove {
            0% { transform: translate(0, 0); }
            100% { transform: translate(50px, 50px); }
        }

        .floating-shapes {
            position: absolute;
            width: 100%;
            height: 100%;
        }

        .shape {
            position: absolute;
            border-radius: 50%;
            filter: blur(40px);
            opacity: 0.3;
            animation: float 15s infinite ease-in-out;
        }

        .shape-1 {
            width: 300px;
            height: 300px;
            background: var(--neon-blue);
            top: 10%;
            left: 5%;
            animation-delay: 0s;
        }

        .shape-2 {
            width: 400px;
            height: 400px;
            background: var(--neon-purple);
            top: 60%;
            right: 10%;
            animation-delay: 5s;
        }

        .shape-3 {
            width: 250px;
            height: 250px;
            background: var(--neon-pink);
            bottom: 10%;
            left: 20%;
            animation-delay: 10s;
        }

        @keyframes float {
            0%, 100% { transform: translateY(0) scale(1); }
            50% { transform: translateY(-20px) scale(1.05); }
        }

        /* App Container */
        .app-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            position: relative;
            z-index: 1;
        }

        /* Header Styles */
        .cyber-header {
            text-align: center;
            margin-bottom: 3rem;
            padding: 4rem 2rem;
            background: rgba(10, 10, 30, 0.5);
            backdrop-filter: blur(10px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            box-shadow: 
                0 10px 30px rgba(0, 0, 0, 0.5),
                var(--shadow-glow-blue);
            position: relative;
            overflow: hidden;
            animation: fadeInDown 1s ease-out;
        }

        .cyber-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, rgba(0, 243, 255, 0.1) 0%, rgba(179, 0, 255, 0.1) 100%);
            z-index: -1;
        }

        .cyber-header h1 {
            font-family: 'Orbitron', sans-serif;
            font-weight: 900;
            font-size: 4rem;
            margin-bottom: 1rem;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 0 0 20px rgba(0, 243, 255, 0.5);
            letter-spacing: 2px;
            animation: textGlow 3s ease-in-out infinite alternate;
        }

        @keyframes textGlow {
            0% { text-shadow: 0 0 10px rgba(0, 243, 255, 0.5); }
            100% { text-shadow: 0 0 20px rgba(0, 243, 255, 0.8), 0 0 30px rgba(179, 0, 255, 0.5); }
        }

        .cyber-header p {
            font-size: 1.3rem;
            max-width: 700px;
            margin: 0 auto;
            color: var(--text-secondary);
        }

        /* Upload Card Styles */
        .cyber-card {
            background: var(--dark-card);
            backdrop-filter: blur(15px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 2.5rem;
            margin-bottom: 2.5rem;
            box-shadow: 
                0 10px 30px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            transition: var(--transition-slow);
            position: relative;
            overflow: hidden;
            animation: fadeInUp 0.8s ease-out 0.2s both;
        }

        .cyber-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: var(--gradient-primary);
        }

        .cyber-card:hover {
            transform: translateY(-10px) rotateX(5deg);
            box-shadow: 
                0 15px 40px rgba(0, 0, 0, 0.4),
                var(--shadow-glow-blue),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }

        .upload-icon {
            font-size: 5rem;
            margin-bottom: 1.5rem;
            display: inline-block;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: iconPulse 2s infinite ease-in-out;
        }

        @keyframes iconPulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }

        .upload-area {
            border: 2px dashed rgba(0, 243, 255, 0.3);
            border-radius: 15px;
            padding: 3rem 2rem;
            text-align: center;
            transition: var(--transition-fast);
            cursor: pointer;
            margin-bottom: 2rem;
            background: rgba(0, 0, 0, 0.2);
            position: relative;
            overflow: hidden;
        }

        .upload-area::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(0, 243, 255, 0.1), transparent);
            transition: 0.5s;
        }

        .upload-area:hover {
            border-color: var(--neon-blue);
            box-shadow: var(--shadow-glow-blue);
        }

        .upload-area:hover::before {
            left: 100%;
        }

        .upload-area.dragover {
            border-color: var(--neon-blue);
            background-color: rgba(0, 243, 255, 0.05);
            transform: scale(1.02);
        }

        /* Button Styles */
        .cyber-button {
            background: var(--gradient-primary);
            border: none;
            padding: 1rem 2.5rem;
            font-weight: 600;
            border-radius: 50px;
            color: white;
            position: relative;
            overflow: hidden;
            transition: var(--transition-fast);
            box-shadow: 0 5px 15px rgba(0, 243, 255, 0.4);
            font-family: 'Exo 2', sans-serif;
            letter-spacing: 1px;
        }

        .cyber-button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
            transition: 0.5s;
        }

        .cyber-button:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 243, 255, 0.6);
        }

        .cyber-button:hover::before {
            left: 100%;
        }

        .cyber-button:active {
            transform: translateY(-2px);
        }

        /* Report Download Button */
        .report-button {
            background: var(--gradient-success);
            border: none;
            padding: 1rem 2rem;
            font-weight: 600;
            border-radius: 50px;
            color: white;
            position: relative;
            overflow: hidden;
            transition: var(--transition-fast);
            box-shadow: 0 5px 15px rgba(0, 255, 157, 0.4);
            font-family: 'Exo 2', sans-serif;
            letter-spacing: 1px;
            font-size: 1.1rem;
            margin: 2rem auto;
            display: block;
            text-decoration: none;
            text-align: center;
            max-width: 300px;
        }

        .report-button:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 255, 157, 0.6);
            color: white;
        }

        /* White outline button for details */
        .btn-outline-white {
            background: transparent;
            border: 2px solid var(--text-primary);
            color: var(--text-primary) !important;
            padding: 0.5rem 1rem;
            font-weight: 600;
            border-radius: 50px;
            transition: var(--transition-fast);
            font-size: 0.85rem;
        }

        .btn-outline-white:hover {
            background: var(--text-primary);
            color: var(--dark-bg) !important;
            transform: translateY(-2px);
            box-shadow: var(--shadow-glow-blue);
        }

        /* Results Card Styles */
        .results-card {
            opacity: 0;
            transform: translateY(30px);
            animation: fadeInUp 0.8s ease-out forwards;
        }

        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }

        .stat-card {
            background: var(--dark-card);
            backdrop-filter: blur(10px);
            border: 1px solid var(--glass-border);
            border-radius: 15px;
            padding: 2rem 1.5rem;
            text-align: center;
            transition: var(--transition-fast);
            position: relative;
            overflow: hidden;
            animation: statReveal 0.8s ease-out forwards;
            opacity: 0;
            transform: translateY(20px);
        }

        .stat-card:nth-child(1) { animation-delay: 0.1s; }
        .stat-card:nth-child(2) { animation-delay: 0.2s; }
        .stat-card:nth-child(3) { animation-delay: 0.3s; }
        .stat-card:nth-child(4) { animation-delay: 0.4s; }

        @keyframes statReveal {
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, rgba(0, 243, 255, 0.1) 0%, rgba(179, 0, 255, 0.1) 100%);
            z-index: -1;
            opacity: 0;
            transition: var(--transition-fast);
        }

        .stat-card:hover::before {
            opacity: 1;
        }

        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3), var(--shadow-glow-blue);
        }

        .stat-value {
            font-size: 3.5rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            line-height: 1;
            font-family: 'Orbitron', sans-serif;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .stat-label {
            font-size: 1rem;
            color: var(--text-secondary);
            font-weight: 500;
        }

        /* Chart Grid */
        .chart-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }

        .chart-card {
            background: var(--dark-card);
            backdrop-filter: blur(10px);
            border: 1px solid var(--glass-border);
            border-radius: 15px;
            padding: 1.5rem;
            transition: var(--transition-fast);
            height: 100%;
            display: flex;
            flex-direction: column;
            position: relative;
            overflow: hidden;
        }

        .chart-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, rgba(0, 243, 255, 0.05) 0%, rgba(179, 0, 255, 0.05) 100%);
            z-index: -1;
            opacity: 0;
            transition: var(--transition-fast);
        }

        .chart-card:hover::before {
            opacity: 1;
        }

        .chart-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3), var(--shadow-glow-blue);
        }

        .chart-header {
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--text-primary);
            font-size: 1.2rem;
            display: flex;
            align-items: center;
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 1px;
        }

        .chart-header i {
            margin-right: 0.5rem;
            color: var(--neon-blue);
        }

        .chart-container {
            flex: 1;
            min-height: 300px;
            position: relative;
        }

        /* Table Styles */
        .table-container {
            background: var(--dark-card);
            backdrop-filter: blur(10px);
            border: 1px solid var(--glass-border);
            border-radius: 15px;
            overflow: hidden;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }

        .table-header {
            background: linear-gradient(90deg, rgba(0, 243, 255, 0.2) 0%, rgba(179, 0, 255, 0.2) 100%);
            padding: 1.2rem 1.5rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 1px;
        }

        .table-header i {
            margin-right: 0.5rem;
            color: var(--neon-blue);
        }

        .data-table {
            width: 100%;
            border-collapse: collapse;
        }

        .data-table th {
            background-color: rgba(0, 0, 0, 0.3);
            padding: 1.2rem 1rem;
            text-align: left;
            font-weight: 600;
            color: var(--text-primary);
            border-bottom: 1px solid var(--glass-border);
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 1px;
            font-size: 0.9rem;
        }

        .data-table td {
            padding: 1.2rem 1rem;
            border-bottom: 1px solid var(--glass-border);
            vertical-align: middle;
            transition: var(--transition-fast);
            color: var(--text-primary) !important;
        }

        .data-table tr:last-child td {
            border-bottom: none;
        }

        .data-table tr:hover td {
            background-color: rgba(0, 243, 255, 0.05);
        }

        /* Badge Styles */
        .score-badge {
            font-size: 0.9rem;
            font-weight: 700;
            color: white;
            padding: 0.6rem 1.2rem;
            border-radius: 50px;
            display: inline-block;
            text-align: center;
            min-width: 70px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
            transition: var(--transition-fast);
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 1px;
        }

        .score-high {
            background: var(--gradient-success);
            box-shadow: 0 4px 15px rgba(0, 255, 157, 0.4);
        }

        .score-medium {
            background: var(--gradient-warning);
            box-shadow: 0 4px 15px rgba(255, 204, 0, 0.4);
        }

        .score-low {
            background: var(--gradient-danger);
            box-shadow: 0 4px 15px rgba(255, 51, 102, 0.4);
        }

        .score-badge:hover {
            transform: scale(1.05);
        }

        .prediction-badge {
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 1px;
        }

        /* Criteria Status Styles */
        .criteria-item {
            position: relative;
            cursor: help;
        }

        .criteria-status {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
            transition: var(--transition-fast);
        }

        .criteria-fulfilled {
            background: rgba(0, 255, 157, 0.2);
            color: var(--neon-green);
            border: 1px solid rgba(0, 255, 157, 0.3);
        }

        .criteria-not-fulfilled {
            background: rgba(255, 51, 102, 0.2);
            color: #ff3366;
            border: 1px solid rgba(255, 51, 102, 0.3);
        }

        .criteria-status i {
            font-size: 0.8rem;
        }

        .criteria-tooltip {
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(10, 10, 30, 0.95);
            backdrop-filter: blur(10px);
            border: 1px solid var(--neon-blue);
            border-radius: 8px;
            padding: 0.8rem 1rem;
            font-size: 0.8rem;
            color: var(--text-primary);
            white-space: nowrap;
            z-index: 1000;
            opacity: 0;
            visibility: hidden;
            transition: var(--transition-fast);
            box-shadow: var(--shadow-glow-blue);
        }

        .criteria-item:hover .criteria-tooltip {
            opacity: 1;
            visibility: visible;
            transform: translateX(-50%) translateY(-5px);
        }

        /* Modal Styles */
        .cyber-modal .modal-content {
            background: var(--dark-card);
            backdrop-filter: blur(15px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            box-shadow: 
                0 15px 40px rgba(0, 0, 0, 0.5),
                var(--shadow-glow-purple);
            color: var(--text-primary);
        }

        .cyber-modal .modal-header {
            background: linear-gradient(90deg, rgba(179, 0, 255, 0.2) 0%, rgba(255, 0, 200, 0.2) 100%);
            border-bottom: 1px solid var(--glass-border);
            border-radius: 20px 20px 0 0;
            padding: 1.5rem;
        }

        .cyber-modal .modal-title {
            font-family: 'Orbitron', sans-serif;
            font-weight: 700;
            color: var(--text-primary);
        }

        .cyber-modal .btn-close {
            filter: invert(1);
        }

        .cyber-modal .modal-body {
            padding: 1.5rem;
        }

        /* Progress bars in modal */
        .cyber-progress {
            height: 10px;
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 5px;
            overflow: hidden;
        }

        .cyber-progress-bar {
            height: 100%;
            border-radius: 5px;
            transition: width 1s ease-in-out;
            position: relative;
            overflow: hidden;
        }

        .cyber-progress-bar::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
            animation: progressShine 2s infinite;
        }

        @keyframes progressShine {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }

        /* Lazy loading styles */
        .lazy-load {
            opacity: 0;
            transform: translateY(20px);
            transition: opacity 0.6s ease, transform 0.6s ease;
        }

        .lazy-load.loaded {
            opacity: 1;
            transform: translateY(0);
        }

        /* Loading spinner */
        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: var(--neon-blue);
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* CHANGES MADE HERE: Make text white */
        .text-muted {
            color: var(--text-primary) !important;
        }

        /* Report Section Styles */
        .report-section {
            background: var(--dark-card);
            backdrop-filter: blur(15px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 2.5rem;
            margin: 2rem 0;
            text-align: center;
            animation: fadeInUp 0.8s ease-out;
        }

        .report-icon {
            font-size: 4rem;
            margin-bottom: 1.5rem;
            display: inline-block;
            background: var(--gradient-success);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .report-description {
            font-size: 1.1rem;
            margin-bottom: 2rem;
            color: var(--text-secondary);
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
        }

        /* Responsive Adjustments */
        @media (max-width: 768px) {
            .app-container {
                padding: 15px;
            }
            
            .cyber-header {
                padding: 2.5rem 1rem;
            }
            
            .cyber-header h1 {
                font-size: 2.5rem;
            }
            
            .cyber-card {
                padding: 1.5rem;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
            }
            
            .chart-grid {
                grid-template-columns: 1fr;
            }
            
            .data-table {
                font-size: 0.85rem;
            }
            
            .data-table th, .data-table td {
                padding: 0.8rem 0.5rem;
            }
            
            .report-section {
                padding: 1.5rem;
            }
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(10, 10, 30, 0.5);
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb {
            background: var(--gradient-primary);
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--gradient-secondary);
        }
    </style>
</head>
<body>
    <!-- Animated Background -->
    <div class="cyber-bg">
        <div class="grid-lines"></div>
        <div class="floating-shapes">
            <div class="shape shape-1"></div>
            <div class="shape shape-2"></div>
            <div class="shape shape-3"></div>
        </div>
    </div>
    
    <div class="app-container">
        <header class="cyber-header">
            <h1>COOKIE SECURITY ANALYZER <span class="text-gradient">PRO</span></h1>
            <p>Advanced hybrid analysis using machine learning and security rule evaluation</p>
        </header>
        
        <div class="cyber-card">
            <div class="text-center mb-4">
                <div class="upload-icon">
                    <i class="fas fa-cloud-upload-alt"></i>
                </div>
                <h2>UPLOAD COOKIE DATA</h2>
                <p class="text-muted">Drag & drop your cookies.json file or click to browse</p>
            </div>
            
            <form action="/analyze" method="post" enctype="multipart/form-data" class="text-center" id="uploadForm">
                <div class="upload-area" id="dropZone">
                    <div id="upload-prompt">
                        <i class="fas fa-file-upload mb-3" style="font-size: 3rem;"></i>
                        <h5>DROP YOUR FILE HERE</h5>
                        <p class="text-muted">or click to select from your device</p>
                    </div>
                    <div id="file-selected-state" style="display: none;"></div>
                    <input type="file" name="cookie_file" id="cookieFile" accept=".json" required hidden>
                </div>
                <button type="submit" class="cyber-button mt-3" id="analyzeBtn">
                    <i class="fas fa-search me-2"></i>ANALYZE COOKIES
                </button>
            </form>
        </div>
        
        <div id="results-container"></div>
    </div>
    
    <!-- Score Details Modal -->
    <div class="modal fade cyber-modal" id="scoreModal" tabindex="-1">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="scoreModalLabel"></h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body" id="scoreModalBody"></div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // DOM Elements
        const uploadForm = document.getElementById('uploadForm');
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('cookieFile');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const resultsContainer = document.getElementById('results-container');
        
        // Store analysis data for PDF generation
        let currentAnalysisData = null;
        let currentCookiesData = null;
        
        // Lazy loading observer
        const lazyLoadObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('loaded');
                    lazyLoadObserver.unobserve(entry.target);
                }
            });
        }, {
            rootMargin: '50px',
            threshold: 0.1
        });

        // Initialize lazy loading
        function initializeLazyLoading() {
            document.querySelectorAll('.lazy-load').forEach(element => {
                lazyLoadObserver.observe(element);
            });
        }

        // Event Listeners
        uploadForm.addEventListener('submit', handleFormSubmit);
        fileInput.addEventListener('change', handleFileSelect);
        
        // Drag and Drop functionality
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(event => {
            dropZone.addEventListener(event, preventDefaults, false);
        });
        
        ['dragenter', 'dragover'].forEach(event => {
            dropZone.addEventListener(event, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(event => {
            dropZone.addEventListener(event, unhighlight, false);
        });
        
        dropZone.addEventListener('drop', handleDrop, false);
        dropZone.addEventListener('click', () => fileInput.click());
        
        // Functions
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        function highlight() {
            dropZone.classList.add('dragover');
        }
        
        function unhighlight() {
            dropZone.classList.remove('dragover');
        }
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            fileInput.files = files;
            handleFileSelect();
        }
        
        function handleFileSelect() {
            const file = fileInput.files[0];
            if (file) {
                document.getElementById('upload-prompt').style.display = 'none';
                const selectedStateDiv = document.getElementById('file-selected-state');
                selectedStateDiv.innerHTML = `
                    <i class="fas fa-file-check mb-2" style="font-size: 2.5rem; color: var(--neon-green);"></i>
                    <h5>${file.name}</h5>
                    <p class="text-muted">Ready to analyze</p>
                `;
                selectedStateDiv.style.display = 'block';
            }
        }
        
        function handleFormSubmit(e) {
            e.preventDefault();
            
            if (!fileInput.files || fileInput.files.length === 0) {
                showAlert('Please select a file first.', 'danger');
                return;
            }
            
            // Update button state
            analyzeBtn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> ANALYZING...';
            analyzeBtn.disabled = true;
            
            // Clear previous results
            resultsContainer.innerHTML = '';
            
            // Submit form
            const formData = new FormData(uploadForm);
            
            fetch('/analyze', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    return response.text().then(text => { throw new Error(text) });
                }
                return response.json();
            })
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }
                
                // Store data for PDF generation
                currentAnalysisData = data.viz_data;
                currentCookiesData = data.cookies_data;
                
                // Display results
                resultsContainer.innerHTML = data.html;
                
                // Add PDF report section
                const reportSection = document.createElement('div');
                reportSection.className = 'report-section lazy-load';
                reportSection.innerHTML = `
                    <div class="report-icon">
                        <i class="fas fa-file-pdf"></i>
                    </div>
                    <h2>Professional Security Report</h2>
                    <p class="report-description">
                        Generate a comprehensive PDF report with executive summary, detailed analysis, 
                        and actionable recommendations. Perfect for stakeholders and compliance documentation.
                    </p>
                    <button class="report-button" onclick="generatePDFReport()">
                        <i class="fas fa-download me-2"></i>DOWNLOAD PDF REPORT
                    </button>
                `;
                resultsContainer.appendChild(reportSection);
                
                // Initialize lazy loading for new content
                initializeLazyLoading();
                
                // Initialize visualizations with lazy loading
                initializeVisualizations(data.viz_data);
                
                // Scroll to results
                document.querySelector('.results-card').scrollIntoView({ 
                    behavior: 'smooth',
                    block: 'start'
                });
            })
            .catch(error => {
                showAlert(error.message, 'danger');
                console.error('Error:', error);
            })
            .finally(() => {
                // Reset button state
                analyzeBtn.innerHTML = '<i class="fas fa-search me-2"></i>ANALYZE COOKIES';
                analyzeBtn.disabled = false;
            });
        }
        
        function showAlert(message, type) {
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
            alertDiv.innerHTML = `
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            
            resultsContainer.innerHTML = '';
            resultsContainer.appendChild(alertDiv);
        }
        
        // Generate PDF Report
        function generatePDFReport() {
            if (!currentAnalysisData) {
                showAlert('No analysis data available. Please analyze cookies first.', 'warning');
                return;
            }
            
            // Show loading state
            const reportButton = document.querySelector('.report-button');
            const originalText = reportButton.innerHTML;
            reportButton.innerHTML = '<span class="spinner-border spinner-border-sm"></span> GENERATING REPORT...';
            reportButton.disabled = true;
            
            fetch('/generate-pdf', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    analysis_data: currentAnalysisData,
                    cookies_data: currentCookiesData
                })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to generate PDF');
                }
                return response.blob();
            })
            .then(blob => {
                // Create download link
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = `cookie_security_report_${new Date().toISOString().split('T')[0]}.pdf`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                
                // Show success message
                showAlert('PDF report generated successfully!', 'success');
            })
            .catch(error => {
                console.error('Error generating PDF:', error);
                showAlert('Failed to generate PDF report. Please try again.', 'danger');
            })
            .finally(() => {
                // Restore button state
                reportButton.innerHTML = originalText;
                reportButton.disabled = false;
            });
        }
        
        // Event delegation for dynamically created elements
        document.addEventListener('click', function(e) {
            if (e.target.matches('.view-details-btn')) {
                const button = e.target;
                document.getElementById('scoreModalLabel').textContent = `Security Score Breakdown: ${button.dataset.cookieName}`;
                
                const breakdown = JSON.parse(button.dataset.breakdown);
                const descriptions = JSON.parse(button.dataset.descriptions || '{}');
                
                let content = '<div class="container-fluid">';
                
                for (const key in breakdown) {
                    const item = breakdown[key];
                    const scorePercent = (item.score / item.weight * 100).toFixed(0);
                    const isFulfilled = item.score > 0;
                    const description = descriptions[key] || 'No description available';
                    
                    content += `
                        <div class="row align-items-center py-3 border-bottom border-secondary">
                            <div class="col-6 criteria-item">
                                <div class="d-flex align-items-center gap-2">
                                    <span>${item.name}</span>
                                    <span class="criteria-status ${isFulfilled ? 'criteria-fulfilled' : 'criteria-not-fulfilled'}">
                                        <i class="fas ${isFulfilled ? 'fa-check' : 'fa-times'}"></i>
                                        ${isFulfilled ? 'Fulfilled' : 'Not Fulfilled'}
                                    </span>
                                </div>
                                <div class="criteria-tooltip">${description}</div>
                            </div>
                            <div class="col-6">
                                <div class="d-flex align-items-center">
                                    <div class="cyber-progress flex-grow-1 me-2">
                                        <div class="cyber-progress-bar" role="progressbar" style="width: ${scorePercent}%; background: ${isFulfilled ? 'var(--gradient-success)' : 'var(--gradient-danger)'};" 
                                             aria-valuenow="${scorePercent}" aria-valuemin="0" aria-valuemax="100"></div>
                                    </div>
                                    <span class="text-nowrap"><strong>${item.score.toFixed(1)}</strong> / ${item.weight}</span>
                                </div>
                            </div>
                        </div>
                    `;
                }
                
                content += '</div>';
                document.getElementById('scoreModalBody').innerHTML = content;
            }
        });
        
        function initializeVisualizations(data) {
            // Lazy load charts with intersection observer
            const chartContainers = document.querySelectorAll('.chart-container');
            
            chartContainers.forEach(container => {
                const lazyLoadChart = new IntersectionObserver((entries) => {
                    entries.forEach(entry => {
                        if (entry.isIntersecting) {
                            const canvas = entry.target.querySelector('canvas');
                            if (canvas && !canvas.chart) {
                                createChart(canvas.id, data);
                            }
                            lazyLoadChart.unobserve(entry.target);
                        }
                    });
                });
                
                lazyLoadChart.observe(container);
            });
        }
        
        function createChart(canvasId, data) {
            const canvas = document.getElementById(canvasId);
            if (!canvas) return;
            
            switch(canvasId) {
                case 'scoreGaugeChart':
                    createScoreGauge(canvas, data);
                    break;
                case 'criteriaImpactChart':
                    createCriteriaImpact(canvas, data);
                    break;
                case 'riskDistributionChart':
                    createRiskDistribution(canvas, data);
                    break;
                case 'securityRadarChart':
                    createSecurityRadar(canvas, data);
                    break;
            }
        }
        
        function createScoreGauge(canvas, data) {
            const score = data.average_score;
            const gaugeData = {
                datasets: [{
                    data: [score, 100 - score],
                    backgroundColor: [
                        score >= 75 ? '#00ff9d' : score >= 50 ? '#ffcc00' : '#ff3366', 
                        'rgba(255, 255, 255, 0.1)'
                    ],
                    borderWidth: 0,
                    circumference: 180,
                    rotation: 270,
                }]
            };
            
            const chart = new Chart(canvas, {
                type: 'doughnut',
                data: gaugeData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '80%',
                    plugins: {
                        legend: { display: false },
                        tooltip: { enabled: false }
                    },
                    animation: {
                        animateRotate: true,
                        animateScale: true
                    }
                }
            });
            
            canvas.chart = chart;
            
            // Add score text
            const scoreText = document.createElement('div');
            scoreText.className = 'position-absolute top-50 start-50 translate-middle text-center';
            scoreText.innerHTML = `
                <div style="font-size: 2.5rem; font-weight: 800; font-family: 'Orbitron', sans-serif; background: var(--gradient-primary); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">${Math.round(score)}</div>
                <div style="font-size: 1rem; color: var(--text-secondary);">/ 100</div>
            `;
            canvas.parentNode.appendChild(scoreText);
        }

        function createCriteriaImpact(canvas, data) {
            if (!data.criteria_impact || data.criteria_impact.length === 0) return;
            
            const chart = new Chart(canvas, {
                type: 'polarArea',
                data: {
                    labels: data.criteria_impact.map(d => d.name),
                    datasets: [{
                        label: 'Score Impact',
                        data: data.criteria_impact.map(d => d.lost_score),
                        backgroundColor: [
                            'rgba(255, 51, 102, 0.7)',
                            'rgba(255, 204, 0, 0.7)',
                            'rgba(0, 243, 255, 0.7)',
                            'rgba(0, 255, 157, 0.7)',
                            'rgba(179, 0, 255, 0.7)'
                        ],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'right',
                            labels: {
                                color: 'white',
                                font: {
                                    family: 'Exo 2, sans-serif'
                                }
                            }
                        }
                    },
                    scales: {
                        r: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                display: false
                            }
                        }
                    },
                    animation: {
                        animateRotate: true,
                        animateScale: true
                    }
                }
            });
            
            canvas.chart = chart;
        }

        function createRiskDistribution(canvas, data) {
            const chart = new Chart(canvas, {
                type: 'doughnut',
                data: {
                    labels: ['Safe', 'Medium Risk', 'High Risk'],
                    datasets: [{
                        data: [
                            data.safe_cookies || 0,
                            data.medium_risk_cookies || 0,
                            data.high_risk_cookies || 0
                        ],
                        backgroundColor: [
                            'rgba(0, 255, 157, 0.7)',
                            'rgba(255, 204, 0, 0.7)',
                            'rgba(255, 51, 102, 0.7)'
                        ],
                        borderWidth: 0,
                        hoverOffset: 15
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'right',
                            labels: {
                                color: 'white',
                                font: {
                                    family: 'Exo 2, sans-serif'
                                }
                            }
                        }
                    },
                    animation: {
                        animateRotate: true,
                        animateScale: true
                    }
                }
            });
            
            canvas.chart = chart;
        }

        function createSecurityRadar(canvas, data) {
            if (!data.security_flags) return;
            
            const chart = new Chart(canvas, {
                type: 'radar',
                data: {
                    labels: Object.keys(data.security_flags),
                    datasets: [{
                        label: 'Adoption Rate (%)',
                        data: Object.values(data.security_flags),
                        backgroundColor: 'rgba(0, 243, 255, 0.2)',
                        borderColor: 'rgba(0, 243, 255, 1)',
                        pointBackgroundColor: 'rgba(0, 243, 255, 1)',
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: 'rgba(0, 243, 255, 1)',
                        borderWidth: 2,
                        pointRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        r: {
                            beginAtZero: true,
                            max: 100,
                            ticks: {
                                stepSize: 20,
                                display: false
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            angleLines: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            pointLabels: {
                                color: 'white',
                                font: {
                                    family: 'Exo 2, sans-serif',
                                    size: 11
                                }
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    animation: {
                        duration: 2000,
                        easing: 'easeOutQuart'
                    }
                }
            });
            
            canvas.chart = chart;
        }

        // Initialize lazy loading on page load
        document.addEventListener('DOMContentLoaded', function() {
            initializeLazyLoading();
        });
    </script>
</body>
</html>
"""

RESULTS_FRAGMENT_HTML = """
<div class="results-card lazy-load">
    <h2 class="mb-4 text-center"><i class="fas fa-chart-line me-2"></i>ANALYSIS DASHBOARD</h2>
    
    <!-- Stats Overview -->
    <div class="stats-grid">
        <div class="stat-card lazy-load">
            <div class="stat-value">{{ average_score|round|int }}</div>
            <div class="stat-label">Overall Security Score</div>
        </div>
        <div class="stat-card lazy-load">
            <div class="stat-value">{{ (cookies|selectattr('ml_prediction', 'equalto', 'Safe')|list|length) }}</div>
            <div class="stat-label">Safe Cookies</div>
        </div>
        <div class="stat-card lazy-load">
            <div class="stat-value">{{ (cookies|selectattr('ml_prediction', 'equalto', 'Risky')|list|length) }}</div>
            <div class="stat-label">Risky Cookies</div>
        </div>
        <div class="stat-card lazy-load">
            <div class="stat-value">{{ cookies|length }}</div>
            <div class="stat-label">Total Cookies</div>
        </div>
    </div>
    
    <!-- Visualizations Grid -->
    <div class="chart-grid">
        <div class="chart-card lazy-load">
            <div class="chart-header"><i class="fas fa-gauge-high"></i> Security Score</div>
            <div class="chart-container">
                <canvas id="scoreGaugeChart"></canvas>
            </div>
        </div>
        <div class="chart-card lazy-load">
            <div class="chart-header"><i class="fas fa-bullseye"></i> Criteria Impact</div>
            <div class="chart-container">
                <canvas id="criteriaImpactChart"></canvas>
            </div>
        </div>
        <div class="chart-card lazy-load">
            <div class="chart-header"><i class="fas fa-shield-alt"></i> Risk Distribution</div>
            <div class="chart-container">
                <canvas id="riskDistributionChart"></canvas>
            </div>
        </div>
        <div class="chart-card lazy-load">
            <div class="chart-header"><i class="fas fa-flag"></i> Security Flags</div>
            <div class="chart-container">
                <canvas id="securityRadarChart"></canvas>
            </div>
        </div>
    </div>
    
    <!-- Detailed Cookie Analysis -->
    <h2 class="mt-5 mb-4 text-center"><i class="fas fa-table me-2"></i>DETAILED COOKIE ANALYSIS</h2>
    
    <div class="table-container lazy-load">
        <div class="table-header">
            <div><i class="fas fa-cookie-bite"></i> Cookie Details</div>
            <div>{{ cookies|length }} items</div>
        </div>
        
        <div class="table-responsive">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Cookie Name</th>
                        <th>Domain</th>
                        <th>ML Prediction</th>
                        <th class="text-center">Safety Score</th>
                        <th class="text-center">Details</th>
                    </tr>
                </thead>
                <tbody>
                    {% for cookie in cookies %}
                    <tr class="lazy-load">
                        <td><strong>{{ cookie.name }}</strong></td>
                        <td>{{ cookie.domain }}</td>
                        <td>
                            <span class="prediction-badge {{ 'text-bg-success' if cookie.ml_prediction == 'Safe' else 'text-bg-danger' }}">
                                <i class="fas {{ 'fa-shield-alt' if cookie.ml_prediction == 'Safe' else 'fa-exclamation-triangle' }} me-1"></i>
                                {{ cookie.ml_prediction }}
                            </span>
                        </td>
                        <td class="text-center">
                            <span class="score-badge {{ 'score-high' if cookie.final_score >= 75 else 'score-medium' if cookie.final_score >= 50 else 'score-low' }}">
                                {{ cookie.final_score|round|int }}
                            </span>
                        </td>
                        <td class="text-center">
                            <button class="btn btn-outline-white btn-sm view-details-btn" data-bs-toggle="modal" data-bs-target="#scoreModal" 
                                data-cookie-name="{{ cookie.name }}" 
                                data-breakdown='{{ cookie.score_breakdown_json }}'
                                data-descriptions='{{ cookie.criteria_descriptions_json }}'>
                                <i class="fas fa-chart-pie me-1"></i>Breakdown
                            </button>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
</div>
"""

@app.route('/')
def index():
    return render_template_string(ANALYZER_PAGE_HTML)

@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    if 'cookie_file' not in request.files or not request.files['cookie_file'].filename:
        return jsonify({"error": "No file selected."}), 400
    
    file = request.files['cookie_file']
    try:
        data = json.loads(file.read().decode('utf-8'))
        cookies_data = data.get("cookies", data)
        page_context = data.get("pageSecurityContext", {})
        
        if not isinstance(cookies_data, list) or not cookies_data:
            return jsonify({"error": "JSON does not contain a valid list of cookies."}), 400

        df = pd.DataFrame(cookies_data)
        ml_risk_probabilities = analyze_cookie_data(df.copy())
        
        processed_cookies = []
        total_score_sum = 0
        lifetime_stats = {"Session": 0, "Short-Term (< 1 mo)": 0, "Long-Term (>= 1 mo)": 0}
        flag_counts = {"HttpOnly": 0, "Secure": 0, "SameSite=Strict": 0}
        criteria_total_lost = {key: 0 for key in WEIGHTS if key != "ml_model_score"}
        
        safe_cookies = 0
        risky_cookies = 0

        for i, cookie in enumerate(cookies_data):
            criteria_scores = evaluate_cookie_safety(cookie, page_context)
            rule_based_score = sum(criteria_scores.values())
            ml_risk_prob = ml_risk_probabilities[i]
            ml_safety_score = (1 - ml_risk_prob) * WEIGHTS["ml_model_score"]
            final_score = ml_safety_score + rule_based_score
            total_score_sum += final_score

            # Count safe vs risky cookies
            if ml_risk_prob <= 0.5:
                safe_cookies += 1
            else:
                risky_cookies += 1

            # Aggregate data for visualizations
            for key, score in criteria_scores.items():
                criteria_total_lost[key] += (WEIGHTS[key] - score)

            if cookie.get("session", True): lifetime_stats["Session"] += 1
            elif cookie.get("expirationDate", 0) - time.time() < 2592000: lifetime_stats["Short-Term (< 1 mo)"] += 1
            else: lifetime_stats["Long-Term (>= 1 mo)"] += 1

            if cookie.get("httpOnly"): flag_counts["HttpOnly"] += 1
            if cookie.get("secure"): flag_counts["Secure"] += 1
            if cookie.get("sameSite", "").lower() == "strict": flag_counts["SameSite=Strict"] += 1
            
            score_breakdown = {
                "ml_model_score": {"name": "ML Model Confidence", "score": ml_safety_score, "weight": WEIGHTS["ml_model_score"]},
                **{k: {"name": format_criteria_name(k), "score": v, "weight": WEIGHTS[k]} for k, v in criteria_scores.items()}
            }
            
            # Include criteria descriptions for tooltips
            criteria_descriptions = {k: CRITERIA_DESCRIPTIONS.get(k, "No description available") for k in criteria_scores.keys()}
            criteria_descriptions["ml_model_score"] = CRITERIA_DESCRIPTIONS["ml_model_score"]

            processed_cookies.append({
                "name": cookie.get("name", "N/A"), 
                "domain": cookie.get("domain", "N/A"),
                "ml_prediction": "Risky" if ml_risk_prob > 0.5 else "Safe",
                "final_score": final_score,
                "score_breakdown_json": json.dumps(score_breakdown),
                "criteria_descriptions_json": json.dumps(criteria_descriptions)
            })
            
        # Prepare visualization data
        total_cookies = len(cookies_data)
        security_flags = {
            "HttpOnly": (flag_counts["HttpOnly"] / total_cookies) * 100 if total_cookies > 0 else 0,
            "Secure": (flag_counts["Secure"] / total_cookies) * 100 if total_cookies > 0 else 0,
            "SameSite": (flag_counts["SameSite=Strict"] / total_cookies) * 100 if total_cookies > 0 else 0
        }
        
        # Filter and sort criteria impact for clarity in the chart
        criteria_impact = sorted(
            [{"name": format_criteria_name(k), "lost_score": v} for k, v in criteria_total_lost.items() if v > 0],
            key=lambda item: item["lost_score"],
            reverse=True
        )[:5]  # Limit to top 5 for better visualization

        viz_data = {
            "average_score": total_score_sum / total_cookies if total_cookies > 0 else 0,
            "criteria_impact": criteria_impact,
            "lifetime_stats": lifetime_stats,
            "security_flags": security_flags,
            "safe_cookies": safe_cookies,
            "risky_cookies": risky_cookies,
            "medium_risk_cookies": total_cookies - safe_cookies - risky_cookies  # Placeholder
        }

        html = render_template_string(
            RESULTS_FRAGMENT_HTML,
            cookies=processed_cookies,
            average_score=viz_data["average_score"]
        )
        
        # Include cookies data for PDF generation
        return jsonify({
            "html": html, 
            "viz_data": viz_data,
            "cookies_data": processed_cookies
        })

    except Exception as e:
        print(f"ERROR during analysis: {e}")
        return jsonify({"error": f"An error occurred during analysis: {e}"}), 500

@app.route('/generate-pdf', methods=['POST'])
def generate_pdf():
    """Generate and download PDF report"""
    try:
        data = request.get_json()
        analysis_data = data.get('analysis_data')
        cookies_data = data.get('cookies_data')
        
        if not analysis_data or not cookies_data:
            return jsonify({"error": "No analysis data available"}), 400
        
        # Generate PDF
        pdf_buffer = generate_pdf_report(analysis_data, cookies_data)
        
        # Return PDF as download
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=f"cookie_security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mimetype='application/pdf'
        )
        
    except Exception as e:
        print(f"ERROR generating PDF: {e}")
        return jsonify({"error": f"Failed to generate PDF: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)