---
title: AI-Powered Manuscript Reviewer
emoji: 📄
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.25.0
app_file: app.py
pinned: true
---

# 📄 AI-Powered Manuscript Reviewer

An advanced web app for academic peer review assistance, built for **Mesopotamian Academic Press** and **Peninsula Publishing Press**. It uses OpenAI GPT models to analyze uploaded manuscripts and recommend article citations from publisher archives.

---

## 🚀 Features

- 🧠 AI-assisted peer review generation using GPT-4
- 📄 PDF title and text extraction (via PyMuPDF)
- 🔍 Semantic article ranking using MiniLM embeddings
- 🔗 DOI-based citation recommendations
- 🏷️ Multi-publisher filtering
- 📦 Cached article metadata for faster searches
- ✅ Manual article search and selection
- 📑 Scientific justification for citations (20–25 words)

---

## 🛠️ How to Run Locally

```bash
pip install -r requirements.txt
python app.py
