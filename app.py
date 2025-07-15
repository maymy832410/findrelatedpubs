import os
import pickle
import re
import requests
import time
import fitz  # PyMuPDF
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
from openai import OpenAI
import gradio as gr
import shutil
import numpy as np

# ------------------ CONFIG -------------------
MOUNT_PATH = "/mnt/data"

if os.path.isdir(MOUNT_PATH):
    print(f"[INFO] ✅ Mounted disk found at {MOUNT_PATH}")
    if os.access(MOUNT_PATH, os.W_OK):
        print("[INFO] ✅ Write permission confirmed.")
    else:
        print("[WARNING] ❌ No write permission to mount path.")
else:
    print("[WARNING] ❌ Mount path not found. Check disk setup.")

SCOPUS_API_KEY = "4c82c88cb16a62f87b0b770c06d6a917"
client = OpenAI(api_key="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")

PUBLISHERS = {
    "Mesopotamian Academic Press": "37356",
    "Peninsula Publishing Press": "51231"
}
CACHE_FILE = "/mnt/data/cached_articles.pkl"
CACHE_DURATION_SECONDS = 60 * 60 * 24 * 7
model = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model='all-MiniLM-L6-v2')


# ------------------ CACHE -------------------
article_cache = {}
last_cache_time = 0

def is_cache_expired():
    """Check if cache is older than 7 days."""
    return int(time.time()) - last_cache_time > CACHE_DURATION_SECONDS

def load_cache_from_file():
    global article_cache, last_cache_time
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                cache_data = pickle.load(f)
                article_cache.update(cache_data.get("data", {}))
                last_cache_time = cache_data.get("timestamp", 0)
                print(f"[INFO] ✅ Loaded cache with {len(article_cache.get('df', pd.DataFrame()))} records.")

                # Restore NumPy arrays from lists
                if "df" in article_cache and not article_cache["df"].empty:
                    if "Embedding" in article_cache["df"].columns:
                        article_cache["df"]["Embedding"] = article_cache["df"]["Embedding"].apply(
                            lambda x: np.array(x) if isinstance(x, list) else x
                        )
        except Exception as e:
            print(f"[ERROR] Failed to load cache: {e}")
            article_cache["df"] = pd.DataFrame()
    else:
        print("[INFO] ❌ No cache file found. Initializing empty cache.")
        article_cache["df"] = pd.DataFrame()

def save_cache_to_file():
    try:
        cache_copy = article_cache.copy()
        # Convert NumPy arrays to lists for serialization
        if "df" in cache_copy and not cache_copy["df"].empty:
            if "Embedding" in cache_copy["df"].columns:
                cache_copy["df"] = cache_copy["df"].copy()
                cache_copy["df"]["Embedding"] = cache_copy["df"]["Embedding"].apply(
                    lambda x: x.tolist() if isinstance(x, np.ndarray) else x
                )
        with open(CACHE_FILE, "wb") as f:
            pickle.dump({
                "data": cache_copy,
                "timestamp": int(time.time())
            }, f)
        print("[INFO] ✅ Cache saved to disk.")
    except Exception as e:
        print(f"[ERROR] Could not save cache: {e}")

def refresh_cache_if_expired():
    global article_cache
    if not is_cache_expired():
        print("[INFO] ✅ Cache is fresh. No need to update.")
        return

    print("[INFO] 🔄 Cache expired. Fetching from Crossref...")
    all_articles = []
    for pub_name, pub_id in PUBLISHERS.items():
        articles = fetch_articles_from_crossref(pub_id, max_results=300)
        all_articles.extend(articles)

    if all_articles:
        df_new = pd.DataFrame(all_articles)
        df_existing = article_cache.get("df", pd.DataFrame())
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_deduped = df_combined.drop_duplicates(subset=["DOI"], keep="last")
        article_cache["df"] = df_deduped
        save_cache_to_file()
        
# ------------------ FETCHING -------------------
def fetch_articles_from_crossref(publisher_id, max_results=500, progress_cb=None):
    base_url = "https://api.crossref.org/works"
    cursor = "*"
    fetched = 0
    articles = []

    existing_df = article_cache.get("df", pd.DataFrame())
    existing_dois = set(existing_df["DOI"].tolist()) if not existing_df.empty else set()

    while fetched < max_results:
        params = {
            "filter": f"member:{publisher_id},type:journal-article",
            "rows": 100,
            "cursor": cursor,
            "mailto": "mohanad@peninsula-press.ae"
        }
        headers = {
            "User-Agent": "SmartArticleBot/1.0 (mailto:mohanad@peninsula-press.ae)"
        }

        try:
            res = requests.get(base_url, params=params, headers=headers, timeout=10)
            if res.status_code == 429:
                time.sleep(10)
                continue
            if res.status_code != 200:
                break
        except requests.RequestException as e:
            print(f"[ERROR] Fetch error: {e}")
            time.sleep(5)
            continue

        data = res.json().get("message", {})
        items = data.get("items", [])
        cursor = data.get("next-cursor", None)

        for item in items:
            doi = item.get("DOI", "")
            doi_link = f"https://doi.org/{doi}" if doi else ""
            if doi_link in existing_dois:
                continue

            title = item.get("title", [""])[0]
            abstract = re.sub("<[^<]+?>", "", item.get("abstract", "")).strip()
            keywords = " ".join(item.get("subject", []))
            content = f"{title} {abstract} {keywords}".lower()

            authors = ", ".join([
                f"{a.get('given', '')} {a.get('family', '')}" for a in item.get("author", [])
            ]) if "author" in item else "N/A"

            date_parts = item.get("published-print", item.get("published-online", {})).get("date-parts", [[None]])
            published = "-".join([str(x) for x in date_parts[0]]) if date_parts else "N/A"

            # Compute embedding for content
            embedding = model.encode(content, convert_to_numpy=True)

            articles.append({
                "Title": title,
                "Authors": authors,
                "DOI": doi_link,
                "Crossref Citations": item.get("is-referenced-by-count", 0),
                "Content": content,
                "Publisher": publisher_id,
                "Embedding": embedding
            })

            fetched += 1
            if progress_cb:
                progress_cb(f"Fetching from {publisher_id}... Total fetched: {fetched}")
            if fetched >= max_results:
                break

        if not cursor or len(items) < 100:
            break
        time.sleep(0.5)

    if articles:
        new_df = pd.DataFrame(articles)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=["DOI"], keep="last")
        article_cache["df"] = combined_df
        save_cache_to_file()

    return articles

# ------------------ PDF UTILITIES -------------------
def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = "\n".join([page.get_text() for page in doc])
        return text[:9000].strip()
    except Exception as e:
        return f"[ERROR] Failed to extract PDF text: {e}"

def extract_title_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = "\n".join([doc[i].get_text("text") for i in range(min(2, len(doc)))])
        if not text.strip():
            return "[ERROR] No readable text extracted from the first 2 pages of the PDF."
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Extract only the title of the academic paper from the following text. Do not include authors or journal names."},
                {"role": "user", "content": text}
            ],
            max_tokens=50,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] Title extraction failed: {e}"

# ------------------ FILTERING -------------------
def find_related_articles(title):
    df = article_cache.get("df", pd.DataFrame())

    if df.empty:
        print("[INFO] Cache is empty. Fetching fresh articles from Crossref...")
        all_articles = []
        for pub_name, pub_id in PUBLISHERS.items():
            articles = fetch_articles_from_crossref(pub_id, max_results=300)
            all_articles.extend(articles)
        if all_articles:
            df = pd.DataFrame(all_articles)
            article_cache["df"] = df
            save_cache_to_file()
        else:
            print("[ERROR] Could not fetch any articles from Crossref.")
            return gr.update(choices=[], value=[])

    if not isinstance(title, str) or not title.strip():
        print("[INFO] Invalid or missing title input.")
        return gr.update(choices=[], value=[])

    try:
        print(f"[DEBUG] Encoding query title: {title[:100]}...")
        query_embed = model.encode(title, convert_to_numpy=True)

        print("[DEBUG] Retrieving cached embeddings...")
        if "Embedding" not in df.columns:
            print("[WARNING] No precomputed embeddings found. Recomputing...")
            contents = df["Content"].fillna("").tolist()
            contents = [text[:512] for text in contents]
            embeddings = model.encode(contents, convert_to_numpy=True)
            df["Embedding"] = list(embeddings)

        doc_embeds = list(df["Embedding"])

        print("[DEBUG] Computing cosine similarity...")
        sims = cosine_similarity([query_embed], doc_embeds)[0]

        df = df.copy()
        df["Similarity"] = (sims * 100).round(2)
        df_sorted = df.sort_values(by="Similarity", ascending=False).head(10)

        print(f"[DEBUG] Top match: {df_sorted.iloc[0]['Title']} (Similarity: {df_sorted.iloc[0]['Similarity']})")

        items = [f"{row['Title']} ({row['DOI']})" for _, row in df_sorted.iterrows()]
        return gr.update(choices=items, value=[])
    except Exception as e:
        print(f"[ERROR] Failed in find_related_articles: {e}")
        return gr.update(choices=[], value=[])

# ------------------ REVIEW GENERATION -------------------
def check_irrelevant_and_warn(prompt, pdf_text, extracted_title, selected_articles):
    if not selected_articles or not extracted_title.strip():
        return gr.update(value="No articles selected or no extracted title.", visible=True), gr.update(visible=False)

    df = article_cache.get("df", pd.DataFrame())
    if df.empty:
        return gr.update(value="⚠️ No cached articles available to evaluate relevance.", visible=True), gr.update(visible=False)

    try:
        pdf_embed = model.encode(extracted_title.strip(), convert_to_numpy=True)
    except Exception as e:
        return gr.update(value=f"[ERROR] Failed to encode title: {e}", visible=True), gr.update(visible=False)

    irrelevant = []

    for item in selected_articles:
        try:
            title = item.split(" (")[0].strip()
            doi = item.split("(")[-1].split(")")[0].strip()
            article_row = df[df["DOI"] == doi]

            if article_row.empty:
                continue

            article_embed = article_row.iloc[0]["Embedding"]
            sim = cosine_similarity([pdf_embed], [article_embed])[0][0] * 100

            if sim >= 70:
                continue  # Considered relevant based on embedding

            # If similarity is low, double-check with GPT
            question = (
                f"The uploaded manuscript is titled: \"{extracted_title}\".\n\n"
                f"Is the following article relevant to this topic? Answer with 'yes' or 'no' only.\n\n"
                f"Article title: \"{title}\""
            )

            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an academic reviewer. Only answer with 'yes' or 'no'."},
                        {"role": "user", "content": question}
                    ],
                    max_tokens=3,
                    temperature=0.0
                )
                answer = response.choices[0].message.content.strip().lower()
                if answer not in ["yes", "no"]:
                    print(f"[WARNING] Unexpected GPT answer: {answer} → defaulting to cosine")
                    continue  # Ignore GPT and don’t mark as irrelevant
                if answer == "no":
                    irrelevant.append(f"- **{title}** ({doi})")
            except Exception as e:
                print(f"[WARNING] GPT API error during relevance check: {e} → using cosine only")

        except Exception as e:
            print(f"[ERROR] Failed relevance check for article {item}: {e}")
            continue

    if irrelevant:
        msg = f"⚠️ The following articles may be irrelevant to the manuscript title:\n\n" + "\n".join(irrelevant) + \
              "\n\nYou may go back and deselect them, or proceed anyway."
        return gr.update(value=msg, visible=True), gr.update(visible=True)
    else:
        return "", gr.update(visible=True)

def handle_review(prompt, text):
    try:
        combined = f"{prompt}\n\n---\n\n{text}"
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": combined}],
            max_tokens=1500,
            temperature=0.4
        )
        return response.choices[0].message.content, gr.update(visible=False)
    except Exception as e:
        fallback_message = f"""[WARNING] GPT-4 API failed. This is a fallback response.

[ERROR DETAILS] {e}

The OpenAI API was unreachable or failed to respond. Please try again later or use the fallback output below based on available cache and similarity metrics."""
        return fallback_message, gr.update(visible=True)

# ------------------ APP UI -------------------
load_cache_from_file()
refresh_cache_if_expired()
with gr.Blocks() as app:
    gr.Markdown("# \U0001F4C4 AI-Powered Manuscript Reviewer")
    cache_count = len(article_cache.get("df", pd.DataFrame()))
    gr.Markdown(f"### \U0001F4CA Cached Articles: {cache_count}")

    with gr.Row():
        pdf_file = gr.File(label="Upload PDF")
        extracted_title = gr.Textbox(label="Extracted Title from PDF", interactive=True)
        extract_btn = gr.Button("Extract Title + Text")
        filter_btn = gr.Button("Find Related Articles")

    result_checkboxes = gr.CheckboxGroup(label="Select Articles to Cite", choices=[])

    with gr.Row():
        prompt_box = gr.TextArea(label="Prompt Area (Editable)", lines=8)
        review_output = gr.Textbox(label="Generated Review", lines=12, interactive=False)

    fallback_notice = gr.Markdown(
        "**⚠️ GPT-4 API failed. This review is generated using cached similarity logic as fallback.**",
        visible=False
    )

    irrelevant_warning = gr.Markdown(value="", visible=False)
    proceed_btn = gr.Button("Proceed Anyway", visible=False)

    submit_btn = gr.Button("Generate Review")
    pdf_text_hidden = gr.State()

    manual_search = gr.Textbox(label="Search Cached Articles (Title, Publisher)", placeholder="Type to search...")
    manual_results = gr.CheckboxGroup(label="Manual Search Results", interactive=True)

    def extract_both(pdf):
        if not pdf:
            return "[ERROR] No PDF uploaded.", "[ERROR] No text extracted."
        try:
            temp_path = f"/tmp/{os.path.basename(pdf.name)}"
            import shutil
            shutil.copy(pdf.name, temp_path)
            title = extract_title_from_pdf(temp_path)
            text = extract_text_from_pdf(temp_path)
            return title, text
        except Exception as e:
            return "[ERROR] Extraction failed.", "[ERROR] Extraction failed."

    def filter_manual_search(query):
        df = article_cache.get("df", pd.DataFrame())
        if df.empty or not query:
            return gr.update(choices=[], value=[])
        query = query.lower()
        df_filtered = df[df.apply(lambda x: query in str(x['Title']).lower() or query in str(x['Publisher']).lower(), axis=1)]
        return gr.update(choices=[f"{row['Title']} ({row['DOI']})" for _, row in df_filtered.iterrows()], value=[])

    def fill_prompt_with_dois(dois):
        if not dois:
            return ""
        links = []
        for d in dois:
            try:
                title_part = d.split(" (")[0].strip()
                raw_doi = d.split("(")[-1].split(")")[0].strip()
                doi_link = raw_doi if raw_doi.startswith("http") else f"https://doi.org/{raw_doi}"
                links.append(f"{doi_link} — {title_part}")
            except:
                continue
        links_text = "\n".join(links)
        return f"""You're an academic peer reviewer with expertise matching the uploaded manuscript. Write 5 to 8 thoughtful, precise comments that directly address the paper’s content. Include:

- Two points on grammar or clarity  
- One on methodology (its strength or needed improvement)  
- One on the originality or scientific value  
- Others may cover structure, coherence, logic, or interpretation of results  

Avoid generic remarks. Be direct, professional, and helpful.

Also, in one of your comments, recommend citing all four studies below. Mention where each one best fits in the manuscript. Use only the DOI links in your response, and briefly (in no more than 25 words) explain why the citations are relevant.

DOIs for citation (use these only):

{links_text}

Start with the review comments. Don’t include headings or introductions."""

    extract_btn.click(fn=extract_both, inputs=[pdf_file], outputs=[extracted_title, pdf_text_hidden])
    filter_btn.click(fn=find_related_articles, inputs=[extracted_title], outputs=[result_checkboxes])
    manual_search.change(fn=filter_manual_search, inputs=[manual_search], outputs=[manual_results])
    submit_btn.click(fn=check_irrelevant_and_warn, inputs=[prompt_box, pdf_text_hidden, extracted_title, result_checkboxes], outputs=[irrelevant_warning, proceed_btn])
    proceed_btn.click(fn=handle_review, inputs=[prompt_box, pdf_text_hidden], outputs=[review_output, fallback_notice])
    result_checkboxes.change(fn=fill_prompt_with_dois, inputs=result_checkboxes, outputs=prompt_box)
    manual_results.change(fn=fill_prompt_with_dois, inputs=manual_results, outputs=prompt_box)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 8080)))
