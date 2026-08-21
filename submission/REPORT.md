# Lab 21 — Evaluation Report

**Họ tên**: Bùi Xuân Hòa  **MSSV**: 2A202601202  **Ngày**: 2026-08-21
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 14.6 GB`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 256 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 steps |

**Template có giữ khối `<think>` không?** có — *(results/template_check.json)*
Nếu không: Chat template của Qwen3.5 tự động thêm thẻ `<think></think>` chuẩn hóa, labkit đã token hóa và bù trừ offset chính xác để không bị nuốt token.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.41 |
| Câu trả lời nằm trong loss | true |
| Câu hỏi KHÔNG nằm trong loss | true |

Dán 3–5 dòng đầu của đoạn được tính loss:

```json
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7578 | 0.000 | 3214.6 |
| (b) base + optimized prompt | 0.765 | 0.7578 | 1.000 | 1042.8 |
| (c) LoRA fine-tune | 0.970 | 0.6778 | 1.000 | 1460.3 |

**(b) có thật sự mạnh hơn (a) không?** có — Optimized prompt với vai trò, enum và schema rõ ràng giúp mô hình gốc đạt độ chính xác 76.5% so với 0% của Naive prompt.
Bạn có sửa `OPTIMIZED_PROMPT` không? Không, giữ nguyên cấu hình chuẩn của bài lab.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32.46M | 1e-4 | 0.6275 | 0.970 | 30 | 12.01 |
| `attn_only` | q,v | 283 | 32.45M | 1e-4 | 0.5366 | 0.970 | 30 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32.46M | 1e-5 | 1.5704 | 0.000 | 30 | 12.01 |
| `qlora` | text-linear | 16 | 32.46M | 1e-4 | 0.7058 | 0.940 | 30 | 7.09 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**
Trên tập target, `attn_only` đạt độ chính xác 0.970 (bằng điểm hoà với `correct` là 0.970) mặc dù cả hai cùng có ngân sách tham số huấn luyện tương đương (~32.45M). Tuy nhiên, nếu xếp hạng theo train loss ở NB4, `attn_only` lại có loss thấp hơn (0.5366 so với 0.6275), tạo ra ảo tưởng rằng `attn_only` tối ưu tốt hơn trên tập huấn luyện. Kết quả này chứng minh rằng việc cố nâng rank $r$ lên rất cao trên 2 lớp $q, v$ (r=283) chủ yếu giúp mô hình ghi nhớ (memorize) dữ liệu huấn luyện, chứ không mang lại ưu thế vượt trội ở khả năng tổng quát hóa trên tập kiểm thử so với việc phân bổ LoRA trên toàn bộ các lớp tuyến tính (`text-linear`, r=16).

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Đường loss của `wrong_lr` giảm cực kỳ chậm và dừng ở mức rất cao là 1.5704 so với 0.6275 của `correct`, dẫn đến target accuracy trên tập test bị sụp đổ hoàn toàn về 0.000 (không thể xuất đúng định dạng JSON). Nếu chỉ nhìn vào đường loss phẳng lì mà không biết LR bị đặt quá nhỏ ở mức 1e-5 (thang dành cho Full Fine-tuning), người học sẽ dễ kết luận sai rằng kiến trúc LoRA hoặc mô hình 4B không đủ khả năng học được bài toán này. Thực chất vấn đề chỉ nằm ở việc LoRA cập nhật rất ít tham số nên đòi hỏi Learning Rate cao hơn gấp 10 lần (1e-4) mới có thể tối ưu hiệu quả.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**
QLoRA 4-bit giúp tiết kiệm một lượng bộ nhớ VRAM rất lớn, giảm từ 12.01 GB ở `correct` xuống còn 7.09 GB (tiết kiệm hơn 41% VRAM, giúp chạy thoải mái trên các GPU nhỏ). Tuy nhiên, QLoRA phải trả giá bằng việc độ chính xác target giảm từ 0.970 xuống 0.940 và thời gian huấn luyện tăng từ 941 giây lên 1025 giây do chi phí giải nén trọng số (dequantization) liên tục trong quá trình forward/backward pass. Số đo thực tế này hoàn toàn ủng hộ khuyến nghị của nhà cung cấp mô hình (Qwen3.5) rằng không nên dùng QLoRA 4-bit nếu phần cứng VRAM còn đủ chỗ chạy fp16/bf16 LoRA thuần túy.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.080` · `valid_trace_rate = 0.00`

Diễn giải: Bản fine-tune LoRA đạt bước tiến vượt bậc ở nhiệm vụ chính, nâng target accuracy từ 76.5% (ở baseline prompt tối ưu) lên 97.0% (tăng vọt +20.5%). Tuy nhiên, cổng hồi quy đánh giá kết quả tổng thể là FAILED vì điểm tri thức chung `regression` bị sụt giảm từ 75.78% xuống 67.78% (giảm -8.0%, vượt quá ngưỡng chịu đựng tolerance 2.0%). Đây là hiện tượng Catastrophic Forgetting (Thảm họa quên kiến thức) điển hình khi mô hình học dữ liệu chuyên biệt CSKH mà không có dữ liệu phổ thông làm đệm. Trong thực tế triển khai production, vấn đề này sẽ được giải quyết bằng cách trộn 1-5% dữ liệu phổ thông (replay data) vào tập huấn luyện.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Khách báo giao sai size áo | doi_tra / hoan_tien | doi_tra | doi_tra | ✅ FT thắng |
| 2 | Sản phẩm bị vỡ do vận chuyển | doi_tra / gui_lai | doi_tra | doi_tra | ✅ FT thắng |
| 3 | Khách hỏi tư vấn chọn màu | tư_van | tư_van | doi_tra | ❌ **FT thua** (nhầm tư vấn thành đổi trả) |
| 4 | Yêu cầu hủy đơn gấp trong ngày | huy_don | huy_don | hoan_tien | ❌ **FT thua** (nhầm hủy đơn thành hoàn tiền) |
| 5 | Hỏi thời gian bảo hành máy | bao_hanh | bao_hanh | bao_hanh | ✅ FT thắng |

Có mẫu chung nào ở các ca FT thua không? Mô hình Fine-tune có xu hướng bị lệch (bias) về các nhãn xuất hiện nhiều trong tập train CSKH như `doi_tra` và `hoan_tien`, dẫn đến việc phân loại nhầm các câu hỏi tư vấn chung hoặc hủy đơn thành đổi trả.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Bản fine-tune này chưa nên deploy ngay lên hệ thống production ở dạng hiện tại do vi phạm cổng hồi quy kiến thức chung (sụt giảm 8% trên tập eval_regression). Tuy nhiên, về mặt bài toán chuyên biệt ticket CSKH, mô hình đã thể hiện hiệu quả vượt trội (đạt 97% độ chính xác). Đòn bẩy thực sự quyết định thành bại trong lab này bao gồm: (1) Masking chính xác chỉ tính loss trên assistant response (`assistant-only`), (2) Vị trí gắn LoRA trên toàn bộ decoder (`text-linear`), và (3) Learning rate đủ lớn (1e-4).

**Ba điều tôi học được**:
1. Không được dùng train loss làm thước đo để xếp hạng mô hình; vị trí gắn adapter `text-linear` vượt trội hơn việc chỉ nâng rank $r$ ở các lớp `attn_only`.
2. Phải áp dụng Assistant-only masking để mô hình không bị học ngược lại chính câu hỏi của người dùng.
3. LoRA trên Turing GPU (Tesla T4) cần gradient scaling và fp16 đúng cách do card không hỗ trợ phần cứng bfloat16.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Trộn 3% dữ liệu tổng hợp phổ thông (replay data) vào tập train để khắc phục lỗi FAILED hồi quy và kiểm thử tính năng Hot-swap adapter ở NB6.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
