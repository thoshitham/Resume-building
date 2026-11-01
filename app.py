
import streamlit as st
import re
import io
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from xml.sax.saxutils import escape

# Page configuration
st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="🎯",
    layout="wide"
)

# Title
st.title("🎯 AI Resume Matcher & Career Recommendation System")
st.write("Upload your resume and paste a job description.")

# Skills list
SKILLS = [
    "python", "java", "c++", "c", "sql", "mysql",
    "html", "css", "javascript", "machine learning",
    "deep learning", "artificial intelligence",
    "data analysis", "pandas", "numpy", "tensorflow",
    "pytorch", "nlp", "computer vision", "power bi",
    "excel", "git", "communication", "aws",
    "flask", "django", "statistics", "generative ai"
]

# Extract text from PDF
def extract_resume_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text.strip()

# Find skills
def find_skills(text):
    text = text.lower()
    found = []

    for skill in SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, text):
            found.append(skill)

    return found

# Calculate similarity
def calculate_match(resume, job):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume, job])

    score = cosine_similarity(vectors[0], vectors[1])[0][0]

    return round(score * 100, 2)

# Generate PDF report
def generate_report(score, matched, missing, job_title):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("AI Resume Analysis Report", styles["Title"]))
    content.append(Spacer(1, 20))

    content.append(Paragraph(
        "Job Role: " + escape(job_title), styles["Normal"]
    ))

    content.append(Paragraph(
        f"Resume Match Percentage: {score}%", styles["Heading2"]
    ))

    content.append(Spacer(1, 15))

    content.append(Paragraph(
        "Matching Skills", styles["Heading2"]
    ))

    content.append(Paragraph(
        escape(", ".join(matched) if matched else "None"),
        styles["Normal"]
    ))

    content.append(Spacer(1, 15))

    content.append(Paragraph(
        "Missing Skills", styles["Heading2"]
    ))

    content.append(Paragraph(
        escape(", ".join(missing) if missing else "None"),
        styles["Normal"]
    ))

    content.append(Spacer(1, 15))

    content.append(Paragraph(
        "Resume Improvement Tips", styles["Heading2"]
    ))

    tips = [
        "Highlight relevant technical skills.",
        "Explain your projects clearly.",
        "Add measurable achievements when available.",
        "Include job keywords that match your real experience."
    ]

    for tip in tips:
        content.append(Paragraph(escape(tip), styles["Normal"]))
        content.append(Spacer(1, 8))

    doc.build(content)
    buffer.seek(0)

    return buffer.getvalue()


# User inputs
resume_file = st.file_uploader(
    "Upload your Resume (PDF)",
    type=["pdf"]
)

job_title = st.text_input("Enter Target Job Role")

job_description = st.text_area(
    "Paste Job Description",
    height=200
)

# Analyze button
if st.button("Analyze Resume", type="primary"):

    if resume_file is None:
        st.error("Please upload your resume.")

    elif not job_description.strip():
        st.error("Please enter a job description.")

    else:
        try:
            resume_text = extract_resume_text(resume_file)

            if not resume_text:
                st.error("No text found in the PDF.")
                st.stop()

            if not job_description.strip():
                st.error("Please enter a job description.")
                st.stop()

            # Calculate match score
            score = calculate_match(
                resume_text,
                job_description
            )

            # Extract skills
            resume_skills = find_skills(resume_text)
            job_skills = find_skills(job_description)

            matched_skills = [
                skill for skill in job_skills
                if skill in resume_skills
            ]

            missing_skills = [
                skill for skill in job_skills
                if skill not in resume_skills
            ]

            # Display results
            st.success("Resume Analysis Completed!")

            st.subheader("Resume Match Score")
            st.progress(score / 100)
            st.metric("Match Percentage", f"{score}%")

            st.divider()

            st.subheader("Matching Skills")

            if matched_skills:
                st.write(", ".join(matched_skills))
            else:
                st.write("No matching skills detected.")

            st.subheader("Missing Skills")

            if missing_skills:
                st.write(", ".join(missing_skills))
            else:
                st.write("No missing skills detected.")

            st.divider()

            st.header("Career Recommendations")

            st.write(
                f"Based on your selected job role: "
                f"{job_title if job_title else 'Target Job'}"
            )

            if missing_skills:
                st.write(
                    "Focus on learning: "
                    + ", ".join(missing_skills)
                )
            else:
                st.write("Your listed skills match the job skills.")

            st.subheader("Career Learning Plan")

            st.write("Week 1: Learn the basics.")
            st.write("Week 2: Practise with exercises.")
            st.write("Week 3: Build a small project.")
            st.write("Week 4: Add the project to your resume.")

            st.subheader("Resume Improvement Tips")

            st.write("1. Highlight relevant technical skills.")
            st.write("2. Explain your projects clearly.")
            st.write("3. Add measurable achievements when available.")
            st.write("4. Include job keywords that match your real experience.")

            # Download report
            pdf_report = generate_report(
                score,
                matched_skills,
                missing_skills,
                job_title if job_title else "Target Job"
            )

            st.download_button(
                label="Download Analysis Report",
                data=pdf_report,
                file_name="Resume_Analysis_Report.pdf",
                mime="application/pdf"
            )

        except Exception as e:
            st.error(f"Something went wrong: {e}")
