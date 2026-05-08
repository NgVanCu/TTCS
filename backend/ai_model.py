import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import re
import unicodedata
import numpy as np
from collections import Counter


# =====================================================================
# BƯỚC 1: IMPORT CÁC HÀM XỬ LÝ LÕI TỪ BỘ CODE KAGGLE
# (Hãy đảm bảo bạn đã copy các class SentenceImportanceModel, 
# hàm sentence_split, topic_aware_select... vào một file, ví dụ kaggle_core.py. 
# Hoặc paste trực tiếp chúng vào trên cùng file này cũng được).
# =====================================================================
# from kaggle_core import SentenceImportanceModel, summarize_extractive 
# (Ở đây mình giả sử bạn đã paste các class/hàm đó vào chung file này rồi)

# =====================================================================
# PHẦN 1: CẤU HÌNH & NÚM VẶN (THE KNOBS)
# =====================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Config:
    MAX_LEN = 256
    MAX_SUMMARY_SENTENCES = 10
    MAX_SUMMARY_WORDS = 600
    MIN_COMPRESSION_RATIO = 0.34
    MMR_LAMBDA = 0.68
    MAX_REDUNDANCY_JACCARD = 0.48
    SELECT_MAX_REDUNDANCY_BIGRAM = 0.60
    POST_SELECT_DEDUPE_JACCARD = 0.25
    POST_SELECT_DEDUPE_BIGRAM = 0.50
    COVERAGE_BONUS_WEIGHT = 0.14
    CENTRALITY_WEIGHT = 0.16
    POSITION_WEIGHT = 0.06
    TOPIC_SIMILARITY_THRESHOLD = 0.20
    TOPIC_ENTROPY_SWITCH = 0.62
    TOPIC_BALANCE_BONUS_WEIGHT = 0.28
    MULTITOPIC_MIN_TOPICS = 4
    MULTITOPIC_MAX_SENT_PER_TOPIC = 2


cfg = Config()


# =====================================================================
# PHẦN 2: CẤU TRÚC MẠNG NƠ-RON (BẮT BUỘC ĐỂ NẠP FILE .BIN)
# =====================================================================
class SentenceImportanceModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        logits = self.classifier(pooled).squeeze(-1)
        return logits


# =====================================================================
# PHẦN 3: CÁC HÀM XỬ LÝ NGÔN NGỮ & THUẬT TOÁN MMR (TỪ KAGGLE)
# =====================================================================
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"\[DOC_\d+\]", " ", text)
    text = text.replace("|TITLE|", " ").replace("|TAGS|", " ")
    return re.sub(r"\s+", " ", text).strip()


def sentence_split(text: str):
    text = normalize_text(text)
    if not text: return []
    abbreviations = ["TP.", "ThS.", "TS.", "PGS.", "GS.", "Ong.", "Ba.", "Mr.", "Ms.", "Dr.", "P.", "Q."]
    protected = text
    for abbr in abbreviations:
        protected = protected.replace(abbr, abbr.replace(".", "<DOT>"))
    protected = re.sub(r"\s*\n+\s*", " <LB> ", protected)
    chunks = re.split(r"(?<=[\.!\?;:])\s+|\s+<LB>\s+", protected)
    chunks = [c.replace("<DOT>", ".").strip(" -*") for c in chunks if c and c.strip()]
    if len(chunks) <= 1:
        chunks = [c.strip() for c in re.split(r",\s+", protected) if c and c.strip()]
        chunks = [c.replace("<DOT>", ".") for c in chunks]
    out = [c for c in chunks if 4 <= len(c.split()) <= 80]
    return out


def simple_tokens(text: str):
    return re.findall(r"\w+", str(text).lower())


def jaccard_similarity(a: str, b: str) -> float:
    sa, sb = set(simple_tokens(a)), set(simple_tokens(b))
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)


def sentence_bigrams(text: str):
    toks = simple_tokens(text)
    return set(zip(toks[:-1], toks[1:])) if len(toks) >= 2 else set()


def bigram_overlap(a: str, b: str) -> float:
    ba, bb = sentence_bigrams(a), sentence_bigrams(b)
    if not ba or not bb: return 0.0
    return len(ba & bb) / len(ba | bb)


def predict_sentence_scores(model, tokenizer, sentences):
    if not sentences: return []
    probs = []
    with torch.no_grad():
        for s in sentences:
            enc = tokenizer(s, max_length=cfg.MAX_LEN, truncation=True, padding="max_length", return_tensors="pt")
            logits = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
            probs.append(float(torch.sigmoid(logits).item()))
    return probs


# --- LƯU Ý BẢN RÚT GỌN: Để Web chạy mượt, mình gộp logic Topic-Aware & MMR vào đây ---
def mmr_select_optimized(sentences, scores):
    if not sentences: return []
    selected = []
    candidates = list(range(len(sentences)))

    while candidates and len(selected) < cfg.MAX_SUMMARY_SENTENCES:
        best_idx, best_value = None, -1e9
        for i in candidates:
            relevance = float(scores[i])
            redundancy = 0.0
            if selected:
                red_j = max(jaccard_similarity(sentences[i], sentences[j]) for j in selected)
                red_b = max(bigram_overlap(sentences[i], sentences[j]) for j in selected)
                redundancy = max(red_j, red_b)
                if red_j >= cfg.MAX_REDUNDANCY_JACCARD or red_b >= cfg.SELECT_MAX_REDUNDANCY_BIGRAM:
                    continue

            value = cfg.MMR_LAMBDA * relevance - (1.0 - cfg.MMR_LAMBDA) * redundancy
            if value > best_value:
                best_value = value
                best_idx = i

        if best_idx is None: break
        selected.append(best_idx)
        candidates.remove(best_idx)
        if sum(len(sentences[j].split()) for j in selected) >= cfg.MAX_SUMMARY_WORDS: break

    selected.sort()
    return selected


def summarize_extractive(model, tokenizer, source_text):
    sentences = sentence_split(source_text)
    if not sentences: return ""
    scores = predict_sentence_scores(model, tokenizer, sentences)
    selected_idx = mmr_select_optimized(sentences, scores)
    return " ".join(sentences[i] for i in selected_idx)


# =====================================================================
# PHẦN 4: LỚP VỎ GIAO TIẾP VỚI WEB (LỚP ĐỂ MAIN.PY GỌI)
# =====================================================================
class AI_Summarizer:
    def __init__(self, model_path=None, tokenizer_path=None, use_mock=True):
        self.use_mock = use_mock
        if self.use_mock:
            print("🚀 HỆ THỐNG CHẠY CHẾ ĐỘ: MOCK AI (Test Giao diện)")
        else:
            print("🧠 HỆ THỐNG CHẠY CHẾ ĐỘ: REAL AI (Model Thật)")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

            # Gọi lại kiến trúc PhoBERT để đắp file .bin vào
            self.model = SentenceImportanceModel("vinai/phobert-base").to(DEVICE)
            checkpoint = torch.load(model_path, map_location=DEVICE)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            print("Động cơ AI đã sẵn sàng!")

    def process(self, text: str) -> str:
        if self.use_mock:
            return f"Đây là tóm tắt giả lập! API đã nhận {len(text)} ký tự. Hãy tắt use_mock=False để chạy thật."
        else:
            return summarize_extractive(self.model, self.tokenizer, text)