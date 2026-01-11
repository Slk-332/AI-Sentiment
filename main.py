import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from google import genai
import warnings

warnings.filterwarnings("ignore")

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

API_KEY = "AIzaSyAXK2mDf6SlSBtIEwUrtIfMVkWNTfQM7ZY"
client = genai.Client(api_key=API_KEY)

def predict_sentiment(text):
    """วิเคราะห์อารมณ์จากข้อความ"""
    inputs = tokenizer(str(text), return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return label_map[torch.argmax(outputs.logits, dim=1).item()]

def generate_executive_summary(df):
    """สรุปปัญหาเชิงลบด้วย Gemini 1.5-Flash (ประหยัดโควต้าและแม่นยำ)"""
    neg_samples = df[df.sentiment == "Negative"]["review"].astype(str).head(15).tolist()
    neg_text = "\n".join([f"- {s}" for s in neg_samples])
    
    if not neg_samples:
        return "ภาพรวมลูกค้ามีความพึงพอใจสูงมาก ไม่พบข้อร้องเรียนที่ต้องแก้ไขเร่งด่วน"

    prompt = f"""
    คุณเป็นนักวิเคราะห์กลยุทธ์ธุรกิจ สรุปรีวิวลูกค้าจากข้อมูลนี้:
    - จำนวนรีวิวเชิงลบที่วิเคราะห์: {len(neg_samples)} รายการ
    
    รายการรีวิวเชิงลบจริงจากลูกค้า:
    {neg_text}
    
    จงสรุปปัญหาสำคัญ 3 ประเด็น และแนวทางแก้ไข 1 ข้อ เป็นภาษาไทย (กระชับและเป็นมืออาชีพ)
    """

    try:
        # ใช้รุ่น 1.5-flash เพื่อความเสถียรและหลีกเลี่ยงปัญหา Quota (429)
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Gemini Error: {str(e)}"

def analyze():
    path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not path:
        return

    df = pd.read_csv(path)
    if "review" not in df.columns:
        messagebox.showerror("Error", "ต้องมีคอลัมน์ชื่อ review")
        return

    df["sentiment"] = df["review"].astype(str).apply(predict_sentiment)
    # ===== Summary =====
    summary_text.delete("1.0", tk.END)
    # call the actual summary function (was generate_summary -> NameError)
    summary_text.insert(tk.END, generate_executive_summary(df))

# ...existing code...

# ---------------- UI ----------------
root = tk.Tk()
root.title("Thai Sentiment Analysis")
root.geometry("1000x600")
root.configure(bg="#f4f6f9")

style = ttk.Style()
style.theme_use("default")
style.configure("Card.TFrame", background="white", relief="ridge", padding=15)
style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), background="#f4f6f9")
style.configure("Big.TLabel", font=("Segoe UI", 20, "bold"))
style.configure("Sub.TLabel", font=("Segoe UI", 11), foreground="#555")

# ================== HEADER ==================
ttk.Label(root, text="Thai Sentiment Analysis Dashboard", style="Title.TLabel").pack(pady=10)

ttk.Button(root, text="📂 เลือกไฟล์ CSV และวิเคราะห์", command=lambda: analyze()).pack(pady=5)

# ================== SUMMARY ==================
summary = ttk.Frame(root)
summary.pack(fill="x", padx=20, pady=10)

def summary_card(parent, emoji, title, color):
    frame = ttk.Frame(parent, style="Card.TFrame")
    lbl_emoji = tk.Label(frame, text=emoji, font=("Segoe UI Emoji", 28), bg="white")
    lbl_emoji.pack()
    lbl_title = tk.Label(frame, text=title, font=("Segoe UI", 12, "bold"), bg="white")
    lbl_title.pack()
    lbl_percent = tk.Label(frame, text="0%", font=("Segoe UI", 22, "bold"), fg=color, bg="white")
    lbl_percent.pack()
    return frame, lbl_percent

card_pos, lbl_pos = summary_card(summary, "😊", "Positive", "green")
card_neu, lbl_neu = summary_card(summary, "😐", "Neutral", "gray")
card_neg, lbl_neg = summary_card(summary, "😠", "Negative", "red")

card_pos.pack(side="left", expand=True, fill="x", padx=10)
card_neu.pack(side="left", expand=True, fill="x", padx=10)
card_neg.pack(side="left", expand=True, fill="x", padx=10)

# ===== Executive Summary =====
tk.Label(
    root,
    text="📌 Auto Executive Summary",
    font=("Arial", 13, "bold")
).pack(anchor="w", padx=10)

summary_text = tk.Text(
    root,
    height=7,
    wrap="word",
    bg="#f5f5f5"
)
summary_text.pack(fill="x", padx=10, pady=5)


# ================== REVIEWS ==================
reviews_frame = ttk.Frame(root)
reviews_frame.pack(fill="both", expand=True, padx=20, pady=10)

def review_column(parent, title):
    frame = ttk.Frame(parent, style="Card.TFrame")
    tk.Label(frame, text=title, font=("Segoe UI", 12, "bold"), bg="white").pack()
    scrollbar = ttk.Scrollbar(frame)
    scrollbar.pack(side="right", fill="y")
    text = tk.Text(frame, height=15, wrap="word", yscrollcommand=scrollbar.set)
    text.pack(fill="both", expand=True)
    scrollbar.config(command=text.yview)
    return frame, text

frame_p, text_p = review_column(reviews_frame, "Positive Reviews")
frame_n, text_n = review_column(reviews_frame, "Neutral Reviews")
frame_ng, text_ng = review_column(reviews_frame, "Negative Reviews")

reviews_frame.columnconfigure((0,1,2), weight=1)

frame_p.grid(row=0, column=0, sticky="nsew", padx=5)
frame_n.grid(row=0, column=1, sticky="nsew", padx=5)
frame_ng.grid(row=0, column=2, sticky="nsew", padx=5)


# ================== LOGIC ==================
def analyze():
    path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not path:
        return

    df = pd.read_csv(path)
    if "review" not in df.columns:
        messagebox.showerror("Error", "ต้องมีคอลัมน์ชื่อ review")
        return

    # 1. Run Sentiment Analysis
    df["sentiment"] = df["review"].astype(str).apply(predict_sentiment)

    # 2. Update Executive Summary
    summary_text.delete("1.0", tk.END)
    summary_text.insert(tk.END, generate_executive_summary(df))

    # 3. Categorize Reviews
    pos_list = df[df.sentiment == "Positive"]["review"].tolist()
    neu_list = df[df.sentiment == "Neutral"]["review"].tolist()
    neg_list = df[df.sentiment == "Negative"]["review"].tolist()

    # 4. Update Statistic Cards
    total = len(df)
    lbl_pos.config(text=f"{len(pos_list)/total*100:.1f}%")
    lbl_neu.config(text=f"{len(neu_list)/total*100:.1f}%")
    lbl_neg.config(text=f"{len(neg_list)/total*100:.1f}%")

    # 5. Update Review Text Boxes (using the text_p, text_n, text_ng names)
    for t in (text_p, text_n, text_ng):
        t.delete("1.0", tk.END)

    for i, r in enumerate(pos_list, 1):
        text_p.insert(tk.END, f"{i}. {r}\n\n")
    for i, r in enumerate(neu_list, 1):
        text_n.insert(tk.END, f"{i}. {r}\n\n")
    for i, r in enumerate(neg_list, 1):
        text_ng.insert(tk.END, f"{i}. {r}\n\n")

    # 6. Show Visualization
    plt.figure(figsize=(5, 5))
    df["sentiment"].value_counts().plot(kind="pie", autopct="%.1f%%", title="Sentiment Distribution")
    plt.ylabel("")
    plt.show()

root.mainloop()