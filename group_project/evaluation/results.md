# RAG Chatbot Evaluation Results

- Run time: 2026-06-09T06:31:29
- Judge: `heuristic_overlap`
- Evaluated: 20 / 20 cases
- Top-k: 5
- Exact phrase mode: False
- Elapsed: 4.83s

## Average Scores

| Metric | Score |
|---|---:|
| Faithfulness | 0.919 |
| Answer Relevancy | 0.628 |
| Context Recall | 0.796 |
| Context Precision | 0.719 |
| **Overall** | **0.766** |

## Worst Cases

| ID | Difficulty | Avg | Question |
|---|---|---:|---|
| GD-058 | hard | 0.569 | Tại sao việc nghệ sĩ nổi tiếng sử dụng hoặc tổ chức sử dụng ma túy được xem là nguy hiểm hơn người b |
| GD-046 | medium | 0.604 | Một người bị bắt khi đang cất giấu một lượng nhỏ heroin để sử dụng cá nhân thì bị xử lý theo tội gì? |
| GD-005 | medium | 0.638 | Tội mua bán trái phép chất ma túy có khung hình phạt cao nhất là gì? |

## Per-case Scores

| ID | Doc | Faith | Relevancy | Recall | Precision | Avg | Sources |
|---|---|---:|---:|---:|---:|---:|---:|
| GD-001 | bo-luat-hinh-su-2017 | 0.889 | 0.559 | 0.667 | 0.737 | 0.713 | 5 |
| GD-005 | bo-luat-hinh-su-2017 | 0.905 | 0.542 | 0.542 | 0.562 | 0.638 | 5 |
| GD-008 | bo-luat-hinh-su-2017 | 0.926 | 0.742 | 0.707 | 0.846 | 0.805 | 5 |
| GD-017 | bo-luat-hinh-su-2017 | 0.897 | 0.626 | 0.667 | 0.684 | 0.719 | 5 |
| GD-021 | luat-phong-chong-ma-tuy-2021 | 0.933 | 0.649 | 1.000 | 0.786 | 0.842 | 5 |
| GD-025 | luat-phong-chong-ma-tuy-2021 | 0.946 | 0.524 | 0.810 | 0.786 | 0.766 | 5 |
| GD-027 | luat-phong-chong-ma-tuy-2021 | 0.946 | 0.518 | 0.929 | 0.786 | 0.795 | 5 |
| GD-036 | nghi-dinh-105-2021 | 0.880 | 0.864 | 1.000 | 0.818 | 0.890 | 5 |
| GD-041 | thong-tu-danh-muc-ma-tuy | 0.887 | 0.650 | 0.650 | 0.625 | 0.703 | 5 |
| GD-046 | bo-luat-hinh-su-2017 | 0.842 | 0.448 | 0.581 | 0.545 | 0.604 | 5 |
| GD-049 | news/article_01 | 0.938 | 0.562 | 0.778 | 0.737 | 0.754 | 5 |
| GD-050 | news/article_06 | 0.930 | 0.723 | 0.855 | 0.667 | 0.794 | 5 |
| GD-051 | news/article_09 | 0.925 | 0.651 | 1.000 | 0.864 | 0.860 | 5 |
| GD-052 | news/article_04 | 0.916 | 0.701 | 0.873 | 0.765 | 0.814 | 5 |
| GD-053 | news/article_12 | 0.933 | 0.699 | 0.831 | 0.750 | 0.803 | 5 |
| GD-054 | news/article_03 | 0.958 | 0.784 | 1.000 | 0.714 | 0.864 | 5 |
| GD-055 | news/article_11 | 0.949 | 0.737 | 0.866 | 0.846 | 0.849 | 5 |
| GD-056 | news/article_15 | 0.931 | 0.550 | 0.738 | 0.682 | 0.725 | 5 |
| GD-057 | news/article_14 | 0.949 | 0.702 | 0.918 | 0.667 | 0.809 | 5 |
| GD-058 | news/article_17 | 0.910 | 0.329 | 0.514 | 0.522 | 0.569 | 5 |
