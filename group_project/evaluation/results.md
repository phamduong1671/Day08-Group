# RAG Evaluation Results

## Framework sử dụng

- **Framework:** DeepEval `4.0.5`
- **Judge model:** OpenAI `gpt-4o-mini`
- **Pipeline đánh giá:** `src.task10.generate_with_citation` (retrieval Weaviate hybrid + generation `gpt-4o-mini`)
- **Golden dataset:** 20 câu chấm thành công
- **Evaluation mode:** Full dataset
- **Workers:** 3
- **Threshold pass:** 0.7 · **Thời gian chạy:** 12.2 phút


---

## Overall Scores (A/B)

| Metric | Config A (hybrid + rerank) | Config B (no rerank) | Δ (A − B) |
|--------|---------------------------|----------------------|-----------|
| Faithfulness | 0.744 | 0.748 | -0.004 |
| Answer Relevancy | 0.953 | 0.892 | 0.061 |
| Context Recall | 0.936 | 0.983 | -0.047 |
| Context Precision | 0.971 | 0.761 | 0.210 |
| **Average** | **0.901** | **0.846** | **0.055** |

---

## A/B Comparison Analysis

**Config A — Hybrid + Rerank:** semantic (bge-m3) + BM25 hợp nhất bằng RRF, sau đó cross-encoder `bge-reranker-v2-m3` chấm lại top kết quả.


**Config B — Hybrid, no Rerank:** giống A nhưng bỏ bước cross-encoder; lấy trực tiếp thứ hạng sau RRF; không dùng threshold fallback theo score vì RRF score có thang đo khác cross-encoder.


**Kết luận:** Config tốt hơn theo điểm trung bình là **A (có rerank)** (Δ average = 0.055). Reranking thường nâng Context Precision rõ nhất vì nó đẩy chunk liên quan lên đầu; nếu Δ nhỏ, retrieval gốc đã đủ tốt cho corpus pháp luật có cấu trúc rõ.


---

## Worst Performers (Bottom 3 — Config A)

| # | ID | Question | Faith | Relev | Recall | Prec | Avg |
|---|----|----------|-------|-------|--------|------|-----|
| 1 | GD-050 | Ca sĩ Miu Lê bị bắt ở đâu, vào thời điểm nào và về tội gì? | 0.333 | 0.667 | 1.000 | 1.000 | 0.750 |
| 2 | GD-052 | Diễn viên hài Trần Hữu Tín bị kết án bao nhiêu năm tù và về tội gì? | 0.000 | 1.000 | 1.000 | 1.000 | 0.750 |
| 3 | GD-053 | Rapper Mr. Nhân bị khởi tố về tội gì và trong bối cảnh vụ án nào? | 0.800 | 1.000 | 0.214 | 1.000 | 0.754 |

**Phân tích:** không có case dưới threshold trong full dataset hiện tại. Các case bottom vẫn nên được audit thủ công vì một metric riêng lẻ có thể thấp dù điểm trung bình còn trên ngưỡng.


---

## Recommendations


### Cải tiến 1: Chốt ground truth cho các câu có `note`
**Action:** Chốt ground truth cho các câu có `note`  
**Expected impact:** Đối chiếu khối lượng/khung hình phạt với văn bản gốc trong corpus, bỏ ghi chú. → Context Recall & Faithfulness tăng vì judge so với đáp án chính xác.  

### Cải tiến 2: Mở rộng Q&A cho mảng tin tức (news/)
**Action:** Mở rộng Q&A cho mảng tin tức (news/)  
**Expected impact:** Dataset hiện đã có câu news, nhưng vẫn nên tăng độ phủ theo nhiều bài và nhiều dạng câu hỏi. → Lộ rõ hơn điểm yếu retrieval trên văn bản phi cấu trúc.  

### Cải tiến 3: Tăng top_k retrieval cho câu cross-reference
**Action:** Tăng top_k retrieval cho câu cross-reference  
**Expected impact:** Các câu tổng hợp nhiều điều luật cần nhiều evidence hơn (top_k 5 → 8). → Context Recall tăng cho nhóm câu hard.  


---

## Appendix — Per-question scores (Config A)

| ID | Doc | Diff | Faith | Relev | Recall | Prec |
|----|-----|------|-------|-------|--------|------|
| GD-001 | bo-luat-hinh-su-2017 | easy | 1.000 | 1.000 | 1.000 | 1.000 |
| GD-005 | bo-luat-hinh-su-2017 | medium | 0.833 | 1.000 | 1.000 | 1.000 |
| GD-008 | bo-luat-hinh-su-2017 | hard | 1.000 | 1.000 | 1.000 | 1.000 |
| GD-017 | bo-luat-hinh-su-2017 | hard | 0.700 | 0.900 | 0.500 | 1.000 |
| GD-021 | luat-phong-chong-ma-tuy-2021 | easy | 1.000 | 1.000 | 1.000 | 1.000 |
| GD-025 | luat-phong-chong-ma-tuy-2021 | medium | 0.833 | 1.000 | 1.000 | 1.000 |
| GD-027 | luat-phong-chong-ma-tuy-2021 | medium | 0.857 | 1.000 | 1.000 | 0.917 |
| GD-036 | nghi-dinh-105-2021 | easy | 0.500 | 0.833 | 1.000 | 0.750 |
| GD-041 | thong-tu-danh-muc-ma-tuy | easy | 0.667 | 1.000 | 1.000 | 1.000 |
| GD-046 | bo-luat-hinh-su-2017 | medium | 1.000 | 1.000 | 1.000 | 1.000 |
| GD-049 | news/article_01 | easy | 0.889 | 0.889 | 1.000 | 1.000 |
| GD-050 | news/article_06 | easy | 0.333 | 0.667 | 1.000 | 1.000 |
| GD-051 | news/article_09 | easy | 0.500 | 1.000 | 1.000 | 1.000 |
| GD-052 | news/article_04 | easy | 0.000 | 1.000 | 1.000 | 1.000 |
| GD-053 | news/article_12 | easy | 0.800 | 1.000 | 0.214 | 1.000 |
| GD-054 | news/article_03 | medium | 1.000 | 1.000 | 1.000 | 0.867 |
| GD-055 | news/article_11 | easy | 0.500 | 1.000 | 1.000 | 1.000 |
| GD-056 | news/article_15 | hard | 0.857 | 0.778 | 1.000 | 1.000 |
| GD-057 | news/article_14 | medium | 0.750 | 1.000 | 1.000 | 1.000 |
| GD-058 | news/article_17 | hard | 0.857 | 1.000 | 1.000 | 0.887 |
