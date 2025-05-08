
import tkinter as tk
from tkinter import filedialog
from tkinterweb import HtmlFrame
from PIL import Image, ImageTk, ImageDraw
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from datetime import datetime
import json
import re
from llm import get_gemini_response_legacy

def export_to_pdf(report_text):
    file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
    if not file_path:
        return

    doc = SimpleDocTemplate(file_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    custom_style = ParagraphStyle(
        'Custom',
        parent=styles['BodyText'],
        fontSize=16,  # Increased from 14
        leading=18,   # Increased from 16
        spaceAfter=12,
        textColor=colors.darkslategray
    )

    story = []
    for line in report_text.split('\n'):
        if line.startswith("## "):
            p = Paragraph(line.replace("## ", "<b>").replace(":", "</b>"), styles['Heading2'])
        else:
            p = Paragraph(line, custom_style)
        story.append(p)
        story.append(Spacer(1, 8))

    doc.build(story)

def clean_text(text):
    return text.replace("..", ".").replace(" .", ".").replace(" ,", ",").replace("#", "").replace("*", "").replace("Here's a psychological behavior report based on the data you provided:", "").replace("Based on your data, here are the scores:(leans towards Emotional)(leans towards Innovative)(leans towards Idealistic)", "")

def create_gradient_bg(width, height):
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    for y in range(height):
        position = y / height
        # More vibrant gradient colors
        if position < 0.33:
            r = int(100 + (255 - 100) * position * 3)
            g = int(180 + (255 - 180) * position * 3)
            b = int(220 + (255 - 220) * position * 3)
        elif position < 0.66:
            adjusted = (position - 0.33) * 3
            r = int(255 + (255 - 255) * adjusted)
            g = int(255 + (200 - 255) * adjusted)
            b = int(255 + (150 - 255) * adjusted)
        else:
            adjusted = (position - 0.66) * 3
            r = int(255 + (255 - 255) * adjusted)
            g = int(200 + (230 - 200) * adjusted)
            b = int(150 + (230 - 150) * adjusted)

        draw.line([(0, y), (width, y)], fill=(r, g, b))
    return ImageTk.PhotoImage(img)

def get_score_description(score):
    if score <= 3:
        return "Weak", "#e74c3c"  # Red for weak
    elif score <= 6:
        return "Moderate", "#f39c12"  # Orange for moderate
    elif score <= 8:
        return "Strong", "#2ecc71"  # Green for strong
    else:
        return "Exceptional", "#3498db"  # Blue for exceptional

def create_score_display_html(title, score, max_score=10):
    description, color = get_score_description(score)
    
    return f"""
    <div style="margin: 15px 0; padding: 15px; background: rgba(255,255,255,0.85); border-radius: 12px; box-shadow: 0 3px 10px rgba(0,0,0,0.1); border-left: 5px solid {color};">
        <div style="display: flex; align-items: center; justify-content: space-between; gap: 15px;">
            <!-- Title -->
            <div style="font-size: 24px; font-weight: 600; color: #2c3e50; min-width: 100px; font-family: 'Roboto', sans-serif;">
                {title}
            </div>
            
            <!-- Score and Description -->
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="display: flex; align-items: baseline; gap: 5px;">
                    <span style="font-size: 38px; font-weight: bold; color: {color}; font-family: 'Abadi', sans-serif;">{score}</span>
                    <span style="font-size: 22px; color: #7f8c8d; font-family: 'Roboto', sans-serif;">/ {max_score}</span>
                </div>
                <div style="font-size: 22px; font-weight: 600; color: {color}; padding: 6px 14px; 
                          border-radius: 20px; background: rgba(0,0,0,0.05); font-family: 'Roboto', sans-serif;">
                    {description}
                </div>
            </div>
        </div>
    </div>
    """

def extract_scores_from_text(text):
    patterns = {
        "defensive_avoidant": r"(?:Defensive Avoidant Behavior|\*\*Defensive Avoidant Behavior:\*\*).*?(\d+)(?:/10)?",
        "expressive_positivity": r"(?:Expressive Positivity|\*\*Expressive Positivity:\*\*).*?(\d+)(?:/10)?",
        "confrontational_reactivity": r"(?:Confrontational Reactivity Behavior|\*\*Confrontational Reactivity Behavior:\*\*).*?(\d+)(?:/10)?",
        "startle_responsiveness": r"(?:Startle Responsiveness Behavior|\*\*Startle Responsiveness Behavior:\*\*).*?(\d+)(?:/10)?"
    }
    
    scores = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        scores[key] = int(match.group(1)) if match else 5
    
    return (scores["defensive_avoidant"], scores["expressive_positivity"], 
            scores["confrontational_reactivity"], scores["startle_responsiveness"])

def display_report_from_json(json_path="report.json"):
    try:
        with open(json_path, "r") as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"Failed to load JSON data: {e}")
        return

    report_text = get_gemini_response_legacy(json_data)
    report_text = clean_text(report_text)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Extract scores
    (defensive_avoidant_score, expressive_positivity_score,
     confrontational_reactivity_score, startle_responsiveness_score) = extract_scores_from_text(report_text)

    # Create score displays
    defensive_avoidant_display = create_score_display_html("Defensive Avoidant", defensive_avoidant_score)
    expressive_positivity_display = create_score_display_html("Expressive Positivity", expressive_positivity_score)
    confrontational_reactivity_display = create_score_display_html("Confrontational Reactivity", confrontational_reactivity_score)
    startle_responsiveness_display = create_score_display_html("Startle Responsiveness", startle_responsiveness_score)

    scores_html = f"""
    <div style="display: flex; flex-direction: column; gap: 15px; margin: 0 20px;">
        {defensive_avoidant_display}
        {expressive_positivity_display}
        {confrontational_reactivity_display}
        {startle_responsiveness_display}
    </div>
    """

    # Remove score information from the report text
    patterns = [
        r"\*\*Defensive Avoidant Behavior:\*\*\s*\d+(?:/10)?",
        r"Defensive Avoidant Behavior:\s*\d+(?:/10)?",
        r"\*\*Expressive Positivity:\*\*\s*\d+(?:/10)?",
        r"Expressive Positivity:\s*\d+(?:/10)?",
        r"\*\*Confrontational Reactivity:\*\*\s*\d+(?:/10)?",
        r"Confrontational Reactivity:\s*\d+(?:/10)?",
        r"\*\*Confrontational Reactivity Behavior:\*\*\s*\d+(?:/10)?",
        r"Confrontational Reactivity Behavior:\s*\d+(?:/10)?",
        r"\*\*Startle Responsiveness:\*\*\s*\d+(?:/10)?",
        r"Startle Responsiveness:\s*\d+(?:/10)?",
        r"\*\*Startle Responsiveness Behavior:\*\*\s*\d+(?:/10)?",
        r"Startle Responsiveness Behavior:\s*\d+(?:/10)?"
    ]

    for pattern in patterns:
        report_text = re.sub(pattern, "", report_text, flags=re.IGNORECASE)

    # Define section colors
    section_colors = {
        "profile": "#3498db",
        "patterns": "#2ecc71",
        "traits": "#9b59b6",
        "recommendations": "#e74c3c"
    }

    # Format the report text with larger fonts and better spacing
    formatted_report = report_text.replace("\n\n", "</div><div style='margin-bottom: 25px;'>")
    formatted_report = formatted_report.replace("\n", "<br>")
    
    # Add more vibrant section headers
    formatted_report = formatted_report.replace(
        "1. Emotional Behavior Profile", 
        f"""<h2 style='color: {section_colors["profile"]}; font-size: 30px; margin: 35px 0 20px 0; 
            padding: 12px 18px; border-radius: 8px; background: rgba(52,152,219,0.1); 
            font-family: "Roboto", sans-serif; font-weight: 700;'>
            <span style='background: {section_colors["profile"]}; color: white; padding: 6px 12px; border-radius: 5px;'>1</span>
            Emotional Behavior Profile
        </h2>"""
    ).replace(
        "2. Overall Behavioral Patterns", 
        f"""<h2 style='color: {section_colors["patterns"]}; font-size: 30px; margin: 35px 0 20px 0; 
            padding: 12px 18px; border-radius: 8px; background: rgba(46,204,113,0.1); 
            font-family: "Roboto", sans-serif; font-weight: 700;'>
            <span style='background: {section_colors["patterns"]}; color: white; padding: 6px 12px; border-radius: 5px;'>2</span>
            Behavioral Patterns
        </h2>"""
    ).replace(
        "3. Behavioral Tendencies & Underlying Psychological Traits", 
        f"""<h2 style='color: {section_colors["traits"]}; font-size: 30px; margin: 35px 0 20px 0; 
            padding: 12px 18px; border-radius: 8px; background: rgba(155,89,182,0.1); 
            font-family: "Roboto", sans-serif; font-weight: 700;'>
            <span style='background: {section_colors["traits"]}; color: white; padding: 6px 12px; border-radius: 5px;'>3</span>
            Psychological Traits
        </h2>"""
    ).replace(
        "4. Concerns or Red Flags", 
        f"""<h2 style='color: {section_colors["recommendations"]}; font-size: 30px; margin: 35px 0 20px 0; 
            padding: 12px 18px; border-radius: 8px; background: rgba(231,76,60,0.1); 
            font-family: "Roboto", sans-serif; font-weight: 700;'>
            <span style='background: {section_colors["recommendations"]}; color: white; padding: 6px 12px; border-radius: 5px;'>4</span>
            Recommendations
        </h2>"""
    ).replace(
        "* ", 
        """<div style='margin: 15px 0; padding-left: 30px; position: relative; font-size: 20px; 
            line-height: 1.8; color: #34495e; font-family: "Roboto", sans-serif;'>
            <span style='position: absolute; left: 0; color: #3498db; font-size: 24px;'>•</span> """
    )

    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&family=Abadi+MT+Condensed&display=swap" rel="stylesheet">
    </head>
    <body style="font-family: 'Roboto', sans-serif; color: #34495e; line-height: 1.8; background: #f9f9f9;">
        <div style="max-width: 1000px; margin: 0 auto; padding: 20px;">
            <!-- Header -->
            <div style="background: linear-gradient(135deg, #3498db, #2c3e50); 
                        padding: 35px; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.15); 
                        margin-bottom: 30px; text-align: center;">
                <h1 style="color: #1a237e; font-size: 44px; font-weight: 700; margin: 0; 
                           font-family: 'Abadi', sans-serif; letter-spacing: 1px; text-shadow: 1px 1px 2px rgba(255,255,255,0.5);">
                    Emotional Behavior Analysis
                </h1>
                <div style="color: rgba(26,35,126,0.9); margin-top: 15px; font-size: 20px; 
                            font-family: 'Roboto', sans-serif; font-weight: 500;">
                    Generated by EmoQuest v1.0 • {timestamp}
                </div>
            </div>

            <!-- Scores Section -->
            <div style="background: white; padding: 30px; border-radius: 16px; 
                        margin-bottom: 30px; box-shadow: 0 8px 25px rgba(0,0,0,0.08);">
                <h2 style='color: #2c3e50; font-size: 34px; margin: 0 0 25px 0; padding-bottom: 12px; 
                           border-bottom: 3px solid #3498db; text-align: center; font-weight: 700; 
                           font-family: "Abadi", sans-serif;'>
                    Your Emotional Profile Scores
                </h2>
                {scores_html}
            </div>

            <!-- Report Content -->
            <div style="background: white; padding: 35px; border-radius: 16px; 
                        margin-bottom: 30px; box-shadow: 0 8px 25px rgba(0,0,0,0.08);">
                {formatted_report}
            </div>

            <!-- Footer -->
            <div style='text-align: center; color: #7f8c8d; margin-top: 30px; font-size: 18px; 
                        border-top: 1px solid rgba(0,0,0,0.1); padding-top: 20px; 
                        font-family: "Roboto", sans-serif;'>
                Confidential Report • {timestamp}<br>
                <span style="color: #3498db; font-weight: 500;">EmoQuest Behavioral Analytics Suite v1.0</span>
            </div>
        </div>
    </body>
    </html>
    """

    # GUI setup
    root = tk.Tk()
    root.title("Emotional Behavior Report")
    root.geometry("1200x900")
    root.configure(bg='#f0f3f6')

    # Dynamic gradient background
    bg_image = create_gradient_bg(1200, 900)
    bg_label = tk.Label(root, image=bg_image)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    # Container for content
    container = tk.Frame(root, bg='white', bd=0)
    container.place(relx=0.05, rely=0.05, relwidth=0.9, relheight=0.8)
    container.configure(highlightthickness=0)

    frame = tk.Frame(container, bg='white')
    frame.pack(fill="both", expand=True)

    # Use HtmlFrame for better rendering
    html_frame = HtmlFrame(frame, messages_enabled=False, vertical_scrollbar=True)
    html_frame.load_html(html_report)
    html_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Enhanced Export button
    export_btn = tk.Button(
        root, 
        text="📄 Export Report", 
        font=("Roboto", 14, "bold"),
        bg='#3498db', 
        fg='white',
        activebackground='#2980b9',
        borderwidth=0,
        relief='flat',
        padx=25,
        pady=12,
        cursor="hand2",
        command=lambda: export_to_pdf(report_text)
    )
    export_btn.place(relx=0.5, rely=0.94, anchor='center', width=200, height=50)

    # Hover effects
    def on_enter(e):
        export_btn['background'] = '#2980b9'
        export_btn['fg'] = 'white'
    
    def on_leave(e):
        export_btn['background'] = '#3498db'
        export_btn['fg'] = 'white'
    
    export_btn.bind("<Enter>", on_enter)
    export_btn.bind("<Leave>", on_leave)

    root.mainloop()

if __name__ == "__main__":
    display_report_from_json()