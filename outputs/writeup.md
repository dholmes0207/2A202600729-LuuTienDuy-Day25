# NimbusAI GPU FinOps — Bài viết phân tích

**Tên:** Lưu Tiến Duy  
**MSSV:** 2A202600729  
**Ngày:** 01/07/2026  

---

## 1. Baseline vs. Optimized

### Kết quả tổng hợp

| Chỉ số | Baseline | Optimized | Thay đổi |
|---|---|---|---|
| **Chi phí hàng tháng** | $27,133 | $14,626 | **-$12,507 (-46%)** |
| **$/1M-token (inference)** | $6.488 | $1.126 | **-$5.362 (-82.6%)** |
| **Purchasing (GPU)** | $25,667 | $15,627 | **-$10,040 (-39.1%)** |

### Phân tích theo đòn bẩy

| Đòn bẩy | Savings (USD/tháng) | % đóng góp |
|---|---|---|
| **Inference (cascade/cache/batch)** | $1,212 | 9.7% |
| **Purchasing (spot/reserved)** | $10,040 | 80.3% |
| **Right-size util-lies** | $655 | 5.2% |
| **Kill idle GPUs** | $600 | 4.8% |
| **TỔNG** | **$12,507** | **100%** |

---

## 2. Phân tích từng đòn bẩy

### Purchasing (spot/reserved) — đóng góp lớn nhất (80.3%)

Đòn bẩy mua GPU chiếm phần lớn nhất vì chi phí GPU-hour là thành phần lớn trong tổng chi phí. Các job training (`job-train-llm`, `job-train-embed`, `job-finetune`) đều có `interruptible=1` nên được chuyển sang **spot** instance, tiết kiệm ~37-40% so với on-demand. Các job inference chạy 24/7 (`job-infer-chat`, `job-infer-rag`) vượt điểm hòa vốn (≥55% duty cycle) nên được chuyển sang **reserved 3yr**, tiết kiệm ~45%.

**Insight quan trọng:** Điểm hòa vốn cho reserved instance tại discount 45% là **55% utilization = 13.2 giờ/ngày**. Bất kỳ job nào chạy >13.2h/ngày liên tục nên cam kết reserved.

### Inference (cascade/cache/batch) — tiết kiệm $/1M-token nhiều nhất

Mặc dù chỉ đóng góp 9.7% trong tổng USD, đây là đòn bẩy quan trọng nhất về **đơn giá token**:
- **Cascade:** 80% traffic được route sang model nhỏ ($0.20/1M input vs $3.00/1M) — savings 15x trên mỗi request đơn giản
- **Prompt Caching:** Giảm 90% chi phí input đã cache — team `assistant` và `rag` cache ~50% input tokens
- **Batch API:** Giảm 50% cho eval traffic (team `eval` sử dụng batch)
- **Discount stack:** batch + 100% cache → chỉ còn 5% chi phí gốc (0.5 × 0.1 = 0.05)

### Right-size util-lies

GPU `gpu-h100-4` bị phát hiện "nói dối" → hạ cấp từ H100 ($2.50/hr) xuống A100 ($1.79/hr). Tiết kiệm $0.71/hr × 24h × 30 ngày = $511/tháng.

### Kill idle GPUs

GPU `gpu-h100-5` idle 8 giờ mỗi đêm → lãng phí $2.50 × 8 = $20/ngày = $600/tháng.

---

## 3. GPU-Util Lie

### GPU nào bị "lie"?

| GPU | Type | GPU-Util% | MFU | MBU |
|---|---|---|---|---|
| **gpu-h100-4** | H100 | **98.2%** | **0.194** | 0.207 |
| **gpu-a10g-1** | A10G | **96.9%** | **0.268** | 0.302 |

### Tại sao đây là "lie"?

`nvidia-smi` báo cáo **GPU-Util = 98%**, nghĩa là clock của GPU "đang bận" 98% thời gian. Tuy nhiên, **MFU chỉ ~0.20** — tức GPU chỉ thực sự sử dụng 20% FLOPs tiềm năng.

**Nguyên nhân gốc rễ:**
- **Memory stall:** GPU đang chờ dữ liệu từ HBM. Kernel được launch nhưng phần lớn thời gian chờ memory → clock "bận" nhưng không tính toán
- **Kernel launch overhead:** Quá nhiều kernel nhỏ, mỗi kernel tốn overhead → utilization cao nhưng throughput thấp
- **I/O wait:** GPU đang chờ dữ liệu từ CPU/PCIe

### Tác động tài chính

Bạn trả tiền cho **cả giờ H100** ($2.50/hr) nhưng chỉ nhận được **1/5 FLOPs thực sự**. Effective cost = $2.50/hr ÷ 0.20 = **$12.50 per FLOPs-hour** — đắt gấp 5x so với kỳ vọng.

---

## 4. Phần mở rộng đã làm

### Extension 3: `cache_is_worth_it()` — Kinh tế học Cache

**Kết quả đo lường:**
- Model lớn ($3.00/1M): break-even chỉ **0.19 reads** → cache hầu như luôn có lợi
- Model nhỏ ($0.20/1M): break-even **2.78 reads** → cache chỉ có lợi khi prefix được đọc lại ≥3 lần
- Khi áp dụng cache gate: savings thay đổi từ 82.6% → 82.2% (giảm -0.5%)
- **Insight:** Với model nhỏ rẻ, chi phí ghi cache ($0.50/1M) quá cao so với savings mỗi lần đọc ($0.18/1M). Cần ≥3 lần đọc để hòa vốn.

### Extension 4: Ngân sách Reasoning

**Kết quả đo lường:**
- Reasoning chỉ chiếm **8.4% requests** và **16.5% tokens**
- Nhưng chiếm **94.0% năng lượng** (29,788 Wh vs 1,888 Wh)
- Carbon: reasoning tạo 11,319 gCO2e vs non-reasoning chỉ 717 gCO2e
- Toàn bộ reasoning traffic đến từ team `eval`
- **Insight:** Reasoning query tiêu thụ năng lượng ~80× query thường vì yêu cầu nhiều inference pass (chain-of-thought, verification). Cần giới hạn reasoning cho các task thực sự phức tạp (toán, code, logic proof).

### Extension 5: Carbon-aware Scheduling

**Kết quả đo lường:**
- Chuyển interruptible jobs từ us-east-1 → europe-north1: giảm **92.1% carbon** (1,606 → 127 kgCO2e/tháng)
- Tiết kiệm điện: $507 → $232/tháng (chuyển sang us-east-wa)
- **Best balanced region:** us-east-wa (score 0.048) — cân bằng carbon (90 gCO2/kWh) và giá điện ($0.055/kWh)
- **Trade-off:** europe-north1 sạch nhất nhưng latency cao cho user US. us-east-wa là lựa chọn cân bằng tốt nhất.

---

## 5. Khuyến nghị cho NimbusAI

### Top 3 hành động ưu tiên (theo ROI)

1. **Chuyển tất cả interruptible jobs sang spot + checkpoint** (ROI cao nhất)
   - Training jobs chiếm phần lớn chi phí GPU
   - Spot tiết kiệm ~40% với interrupt rate 5%
   - Yêu cầu: implement checkpoint mỗi 30 phút

2. **Triển khai cascade routing** cho inference
   - 80% traffic có thể dùng model nhỏ (savings 15×/request)
   - Yêu cầu: thêm classifier/router trước API endpoint
   - Kết hợp batch API cho eval traffic

3. **Giám sát bằng MFU thay vì GPU-Util**
   - Tắt dashboard `nvidia-smi` utilization
   - Triển khai đo MFU/MBU thực tế
   - Alert khi GPU-Util > 90% nhưng MFU < 30%
   - Right-size GPU cho workload memory-bound

### Bổ sung dài hạn

- **Reserved instance** cho inference 24/7 (cam kết 3yr)
- **Carbon-aware scheduling:** chuyển batch training sang europe-north1 hoặc us-east-wa
- **Tag mandate:** duy trì tag coverage ≥ 80% để enable chargeback
- **Reasoning budget cap:** giới hạn reasoning queries cho các use case thực sự cần thiết

---

## Sustainability

| Chỉ số | Giá trị |
|---|---|
| Energy per query | 0.24 Wh |
| Carbon per query | 0.091 gCO2e |
| Cheapest + cleanest region | europe-north1 (Norway) |
| Best balanced region | us-east-wa |
| Carbon savings (all interruptible → cleanest) | 92.1% |
