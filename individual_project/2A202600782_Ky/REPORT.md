# REPORT — Day 8 RAG Pipeline v2

## 1. Mục tiêu dự án

Dự án xây dựng pipeline RAG end-to-end cho chủ đề:

- Pháp luật Việt Nam về ma túy và các chất cấm.
- Bài báo về nghệ sĩ Việt Nam liên quan tới ma túy.

Luồng tổng thể:

```text
Collect/Crawl data
→ Convert Markdown
→ Chunking + Indexing
→ Semantic Search + Lexical Search
→ Reranking
→ PageIndex/vectorless fallback
→ Retrieval pipeline
→ Generation có citation
```

## 2. CLI chạy test nhanh

CLI được thêm tại:

```bash
python tools/quick_test.py
```

Nếu repo có `.venv/bin/python`, CLI sẽ tự chạy lại bằng interpreter trong `.venv`
để dùng đúng dependency của dự án.

Các chế độ chạy:

```bash
# Smoke test nhanh, không tải model lớn và không gọi API ngoài
python tools/quick_test.py

# Smoke test một task cụ thể trong README
python tools/quick_test.py --task 6

# Chạy bộ pytest cá nhân theo README
python tools/quick_test.py --mode pytest

# Chạy pytest cho một task cụ thể
python tools/quick_test.py --mode pytest --task 5

# Chạy toàn bộ tests/
python tools/quick_test.py --mode full
```

Lệnh chấm điểm gốc trong README vẫn dùng được:

```bash
pytest tests/ -v
pytest tests/test_individual.py::TestTask1 -v
pytest tests/test_individual.py::TestTask5 -v
```

## 3. CLI chạy query và in kết quả

CLI query được thêm tại:

```bash
python tools/run_query.py "Hình phạt tàng trữ trái phép chất ma tuý?"
```

Nếu repo có `.venv/bin/python`, CLI sẽ tự chạy lại bằng interpreter trong `.venv`.
API key OpenAI được đọc từ `.env` qua `OPENAI_API_KEY` khi chạy generation.

### Cách 1 — In retrieval results, không gọi OpenAI

Dùng khi muốn xem pipeline lấy những chunk nào, score bao nhiêu, source là
`hybrid` hay `pageindex`.

```bash
python tools/run_query.py --mode retrieval "Hình phạt tàng trữ trái phép chất ma tuý?" --top-k 3
```

Output là JSON gồm `content`, `score`, `metadata`, `source`.

### Cách 2 — In câu trả lời RAG có citation, có gọi OpenAI

Dùng khi muốn xem câu trả lời cuối cùng từ Task 10.

```bash
python tools/run_query.py "Hình phạt tàng trữ trái phép chất ma tuý?" --top-k 5
```

Lệnh tương đương bằng one-liner Python:

```bash
python -c "from src.task10_generation import generate_with_citation; r=generate_with_citation('Hình phạt tàng trữ trái phép chất ma tuý?'); print(r['answer'])"
```

Điều kiện cần:

```bash
OPENAI_API_KEY=...
```

Biến này nên nằm trong `.env`; không commit `.env` lên repository.

## 4. Công nghệ sử dụng

| Nhóm | Công nghệ | Vai trò |
|---|---|---|
| Ngôn ngữ | Python | Implement pipeline và test suite |
| Config | `python-dotenv`, `.env` | Quản lý API key/model config |
| Crawl | Crawl4AI | Crawl bài báo, lấy markdown/title/metadata |
| Convert | MarkItDown | Convert PDF/DOC/DOCX/JSON sang Markdown |
| OCR fallback | Tesseract, `pytesseract`, `pdf2image`, Pillow | Trích xuất PDF scan khi text layer kém |
| Chunking | `langchain-text-splitters` + regex structure-aware | Cắt văn bản luật theo Điều, bài báo theo recursive splitter |
| Embedding | `BAAI/bge-m3` qua `sentence-transformers` | Dense embedding multilingual, phù hợp tiếng Việt |
| Vector store | Weaviate | Lưu vector, hỗ trợ semantic/hybrid retrieval |
| Lexical search | `rank-bm25`, `pyvi` | BM25 + tách từ tiếng Việt |
| Reranking | `BAAI/bge-reranker-v2-m3`, RRF, MMR | Tăng precision, fuse dense/sparse, giảm trùng lặp |
| Vectorless fallback | PageIndex SDK hoặc local structural fallback | Fallback khi hybrid yếu |
| Generation | OpenAI SDK | Sinh câu trả lời có citation |
| Test | pytest/unittest | Automated tests cho task 1-10 |
| Group eval | DeepEval/RAGAS/TruLens | Đánh giá RAG theo golden dataset |

## 5. Quá trình theo README tasks

### Task 1 — Thu thập văn bản pháp luật

- Dữ liệu gốc đặt tại `data/landing/legal/`.
- Hiện có các văn bản pháp luật dạng `.pdf`/`.doc`, gồm luật và nghị định liên quan phòng chống ma túy.
- Test kiểm tra thư mục tồn tại, có tối thiểu 3 file và file không rỗng.

### Task 2 — Crawl bài báo

- URL nguồn được lưu trong `data/landing/news/link.txt`.
- `src/task2_crawl_news.py` đọc URL, normalize để bỏ tracking query, crawl bằng Crawl4AI, làm sạch markdown và lưu JSON.
- Output gồm `url`, `title`, `date_crawled`, `content_markdown`.

### Task 3 — Convert sang Markdown

- `src/task3_convert_markdown.py` scan `data/landing/` và xuất sang `data/standardized/`.
- Giữ cấu trúc `legal/` và `news/`.
- Có xử lý `.doc` cũ qua LibreOffice headless trước khi đưa vào MarkItDown.
- Có OCR fallback cho PDF scan.

### Task 4 — Chunking & Indexing

- `src/task4_chunking_indexing.py` dùng chunking structure-aware.
- Văn bản luật được cắt theo ranh giới Điều; Điều quá dài được chia nhỏ nhưng vẫn prepend header Điều.
- Bài báo được cắt bằng recursive splitter và prepend tiêu đề bài.
- Metadata giàu gồm `source`, `type`, `doc_title`, `chuong`, `dieu`, `dieu_title`, `chunk_id`.
- Chunks được persist ở `data/index/chunks.jsonl`.
- Embedding model: `BAAI/bge-m3`, dimension 1024.
- Vector store mục tiêu: Weaviate collection `DrugLawDocs`.

### Task 5 — Semantic Search

- `src/task5_semantic_search.py` implement `semantic_search(query, top_k)`.
- Query được embed bằng cùng model `bge-m3`.
- Weaviate `near_vector` trả kết quả theo cosine distance, sau đó đổi thành similarity score.
- Output chuẩn: `content`, `score`, `metadata`.

### Task 6 — Lexical Search

- `src/task6_lexical_search.py` implement BM25 trên `data/index/chunks.jsonl`.
- Dùng `pyvi` để tách từ tiếng Việt; fallback regex nếu thiếu package.
- Có boost cho số Điều, mã văn bản và phrase pháp lý quan trọng.
- Output được sort theo score giảm dần.

### Task 7 — Reranking

- `src/task7_reranking.py` có 3 cơ chế:
  - Cross-encoder `BAAI/bge-reranker-v2-m3` cho precision.
  - MMR để tăng diversity.
  - RRF để gộp ranked lists từ dense và sparse.
- Task 9 dùng RRF để fuse semantic + lexical, sau đó rerank bằng cross-encoder nếu bật.

### Task 8 — PageIndex Vectorless RAG

- `src/task8_pageindex_vectorless.py` cung cấp `pageindex_search(query, top_k)`.
- Có hướng SDK PageIndex khi có API key.
- Có local structural fallback dựa trên cây Điều/chunk để demo không phụ thuộc dịch vụ ngoài.
- Kết quả gắn `source: "pageindex"` để Task 9 nhận biết fallback.

### Task 9 — Retrieval Pipeline

- `src/task9_retrieval_pipeline.py` implement `retrieve(...)`.
- Luồng:
  - Chạy semantic search và lexical search.
  - Gộp bằng RRF.
  - Rerank bằng cross-encoder.
  - Chuẩn hóa score bằng sigmoid.
  - Nếu kết quả yếu hơn threshold thì fallback PageIndex/vectorless.
- Output có `content`, `score`, `metadata`, `source` là `hybrid` hoặc `pageindex`.

### Task 10 — Generation có Citation

- `src/task10_generation.py` implement:
  - `reorder_for_llm` để giảm lost-in-the-middle.
  - `format_context` để gắn nhãn citation.
  - `generate_with_citation` để retrieve context và gọi LLM.
- Prompt yêu cầu chỉ dùng context và mỗi câu factual phải có citation.
- Nếu không đủ evidence, trả về thông báo không thể xác minh.

## 6. Bài nhóm và evaluation

Thư mục `group_project/evaluation/` đã có:

- `golden_dataset.json`: bộ câu hỏi/đáp án kỳ vọng.
- `eval_pipeline.py`: khung chạy DeepEval/RAGAS/TruLens.
- `results.md`: nơi lưu kết quả đánh giá.

Theo README nhóm, pipeline nên đánh giá 4 metric:

- Faithfulness.
- Answer Relevance.
- Context Recall.
- Context Precision.

Ngoài ra cần A/B test ít nhất 2 cấu hình, ví dụ:

- Hybrid + reranking.
- Hybrid không reranking hoặc dense-only.

## 7. Ghi chú vận hành

- Smoke test nhanh không thay thế hoàn toàn bộ chấm điểm `pytest tests/ -v`; nó dùng để kiểm tra nhanh artifact, import và các đường chạy nhẹ.
- Full pytest có thể tải model embedding/reranking, cần Weaviate hoặc fallback phù hợp, và có thể cần API key cho generation.
- Nên dùng smoke test trong lúc phát triển, full pytest trước khi nộp/demo.
