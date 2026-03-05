# Vietnamese Novel Text-to-Audio

Công cụ CLI chuyển đổi file tiểu thuyết tiếng Việt (`.txt`) thành file âm thanh MP3 theo từng chương, hỗ trợ ba engine TTS:

- **F5-TTS** — clone giọng từ file mẫu, chạy hoàn toàn **offline** trên Apple Silicon (MPS)
- **Edge-TTS** — giọng đọc Microsoft neural, chất lượng cao, **cần internet**, không cần giọng mẫu
- **Piper TTS** — offline, nhẹ, nhanh, ONNX-based, hỗ trợ đọc tên nhân vật nước ngoài qua multi-model

---

## Mục lục

1. [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
2. [Cài đặt](#cài-đặt)
3. [Cấu trúc dự án](#cấu-trúc-dự-án)
4. [Cấu hình](#cấu-hình)
5. [Chọn TTS engine](#chọn-tts-engine)
6. [Piper TTS — Tên nhân vật nước ngoài](#piper-tts--tên-nhân-vật-nước-ngoài)
7. [Chuẩn hoá text tiếng Việt](#chuẩn-hoá-text-tiếng-việt)
8. [Cấu hình giọng tham chiếu (F5-TTS)](#cấu-hình-giọng-tham-chiếu-f5-tts)
9. [Cách sử dụng](#cách-sử-dụng)
10. [Cấu trúc output](#cấu-trúc-output)
11. [Nguyên lý hoạt động](#nguyên-lý-hoạt-động)
12. [Resume (tiếp tục khi bị gián đoạn)](#resume)
13. [Kiểm tra tính toàn vẹn model](#kiểm-tra-tính-toàn-vẹn-model)
14. [Thư viện sử dụng](#thư-viện-sử-dụng)
15. [Hiệu năng ước tính](#hiệu-năng-ước-tính)
16. [Xử lý sự cố](#xử-lý-sự-cố)

---

## Yêu cầu hệ thống

| Yêu cầu | F5-TTS | Edge-TTS | Piper TTS |
|---------|--------|----------|-----------|
| **Máy** | Apple Silicon (M1–M5) | Bất kỳ | Bất kỳ |
| **RAM** | Tối thiểu 8 GB, khuyến nghị 16 GB+ | Không giới hạn | ~500 MB |
| **Python** | 3.11 | 3.11 | 3.11 |
| **ffmpeg** | Bắt buộc | Bắt buộc | Bắt buộc |
| **Internet** | Chỉ lần đầu tải model (~1.5 GB) | **Bắt buộc mỗi lần** | Chỉ lần đầu tải model (~60 MB) |
| **Giọng mẫu** | Bắt buộc | Không cần | Không cần |

---

## Cài đặt

### Bước 1 — Cài Python 3.11 và ffmpeg

```bash
brew install python@3.11 ffmpeg
```

Kiểm tra:

```bash
/opt/homebrew/bin/python3.11 --version   # Python 3.11.x
/opt/homebrew/bin/ffmpeg -version        # ffmpeg version 8.x
```

### Bước 2 — Tạo virtual environment

Chạy từ thư mục gốc của project:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
```

> **Chú ý:** Không dùng `python3` của macOS (3.8.2 — không tương thích với F5-TTS).

### Bước 3 — Cài các thư viện

```bash
.venv/bin/pip install -r requirements.txt
```

### Bước 4 — Đặt file giọng tham chiếu (chỉ cho F5-TTS)

Đặt file MP3 giọng mẫu vào thư mục `reference/` và cập nhật đường dẫn trong `config.yaml`:

```yaml
reference:
  audio_src: "reference/ten_file_giong_mau.mp3"
```

File WAV sẽ được tự động tạo tại `reference/voice.wav` khi chạy lần đầu.

> Nếu dùng **Edge-TTS** hoặc **Piper TTS**, bước này không cần thiết.

### Bước 5 — Tải model Piper TTS (chỉ cho Piper)

Tải model tiếng Việt từ [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices):

```bash
# Tạo thư mục chứa model
mkdir -p models/piper/vi/vi_VN/vais1000/medium

# Tải model tiếng Việt (~60 MB)
curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx" \
  -o models/piper/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx
curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx.json" \
  -o models/piper/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx.json

# (Tuỳ chọn) Tải model tiếng Anh để đọc tên nhân vật nước ngoài (~60 MB)
mkdir -p models/piper/en/en_US/lessac/medium
curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" \
  -o models/piper/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" \
  -o models/piper/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

---

## Cấu trúc dự án

```
text-to-audio/
├── input/                    # Thả file .txt vào đây
├── output/                   # File MP3 xuất ra (tự tạo)
│   └── ten_file/
│       ├── chuong_001.mp3
│       └── chuong_002.mp3
├── reference/
│   ├── voice_mau.mp3         # File giọng mẫu nguồn (dùng với F5-TTS)
│   └── voice.wav             # Tự tạo từ file MP3 nguồn lần đầu chạy
├── models/
│   ├── vi_f5tts/             # Vietnamese fine-tuned checkpoint (F5-TTS, tuỳ chọn)
│   │   ├── model_latest.safetensors
│   │   └── vocab.txt
│   └── piper/                # Piper TTS models (mỗi model cần file .onnx + .onnx.json)
│       ├── vi/vi_VN/vais1000/medium/   # Model tiếng Việt (bắt buộc với Piper)
│       └── en/en_US/lessac/medium/     # Model tiếng Anh (tuỳ chọn, cho tên nước ngoài)
├── cache/                    # WAV tạm thời từng chunk (tự dọn sau khi merge)
├── checkpoints/              # JSON tiến trình mỗi file (để resume)
├── src/
│   ├── config.py             # Đọc và validate config.yaml
│   ├── preprocessor.py       # Làm sạch và chuẩn hoá text tiếng Việt
│   ├── chapter_detector.py   # Phát hiện ranh giới chương bằng regex
│   ├── chunker.py            # Tách text chương → các chunk nhỏ ≤200 ký tự
│   ├── tts_engine.py         # Wrapper F5-TTS (MPS, lazy load, voice cloning)
│   ├── tts_engine_edge.py    # Wrapper Edge-TTS (Microsoft neural TTS)
│   ├── tts_engine_piper.py   # Wrapper Piper TTS (ONNX, offline, multi-model)
│   ├── audio_merger.py       # Ghép WAV chunk → MP3 qua pydub
│   └── pipeline.py           # Orchestrator: file → chapters → MP3
├── main.py                   # CLI entry point
├── verify_models.py          # Kiểm tra SHA256 model đã tải
├── config.yaml               # Cài đặt người dùng
└── requirements.txt
```

---

## Cấu hình

Chỉnh sửa `config.yaml` theo nhu cầu:

```yaml
reference:
  audio_src: "reference/voice_preview.mp3"   # File MP3 giọng mẫu (chỉ F5-TTS)
  wav_path: "reference/voice.wav"
  start_ms: 2000   # Bỏ qua N ms đầu (tránh intro/nhạc nền)

tts:
  engine: "piper"   # "f5" | "edge" | "piper"  ← chọn engine tại đây

  # --- F5-TTS settings (chỉ dùng khi engine: "f5") ---
  device: "mps"    # "mps" cho Apple Silicon GPU | "cpu" nếu gặp lỗi MPS
  nfe_step: 16     # Số bước inference (16 = nhanh, 32 = chất lượng tốt hơn)
  speed: 1.9
  ckpt_file: "models/vi_f5tts/model_latest.safetensors"
  vocab_file: "models/vi_f5tts/vocab.txt"

  # --- Edge-TTS settings (chỉ dùng khi engine: "edge") ---
  edge:
    voice: "vi-VN-HoaiMyNeural"   # Giọng nữ | "vi-VN-NamMinhNeural" = giọng nam
    rate: "+20%"                   # Tốc độ
    volume: "+0%"

  # --- Piper TTS settings (chỉ dùng khi engine: "piper") ---
  piper:
    model_path: "models/piper/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx"
    speaker_id:           # null cho single-speaker; số nguyên cho multi-speaker
    length_scale: 1.0     # Tốc độ: thấp hơn = nhanh hơn (0.5 = 2x), cao hơn = chậm hơn
    noise_scale: 0.667    # Biến thiên prosody
    noise_w_scale: 0.8    # Biến thiên thời lượng âm tiết
    foreign_phoneme: true # Xử lý tên nhân vật nước ngoài
    foreign_lang: "ja"   # Fallback IPA khi không có foreign model: "ja" romaji | "en" espeak
    foreign_model_path_en: ""   # Đường dẫn model Piper tiếng Anh (để "" để vô hiệu)
    foreign_model_path_ja: ""   # Đường dẫn model Piper tiếng Nhật (để "" để vô hiệu)
    pronunciation_dict:   # Override IPA thủ công theo từng từ
      # "Thor": "tɔː"
      # "Takagi": "takaɡi"

audio:
  bitrate: "192k"
  silence_sentence_ms: 200
  silence_paragraph_ms: 500

chunker:
  max_chars: 200
  min_chars: 50

paths:
  input_dir: "input"
  output_dir: "output"
  cache_dir: "cache"
  checkpoint_dir: "checkpoints"
```

---

## Chọn TTS engine

Đổi một dòng trong `config.yaml` để chuyển engine:

```yaml
tts:
  engine: "piper"   # ← "f5" | "edge" | "piper"
```

### So sánh ba engine

| Tiêu chí | F5-TTS | Edge-TTS | Piper TTS |
|----------|--------|----------|-----------|
| **Chất lượng tiếng Việt** | Trung bình (~10–20% sai) | Rất tốt | Tốt |
| **Giọng đọc** | Clone từ file mẫu | Microsoft neural cố định | VITS model cố định |
| **Tốc độ** | ~20–50s/chunk | ~1–2s/chunk | ~0.1–0.5s/chunk |
| **Offline** | Hoàn toàn offline | Cần internet mỗi lần | Hoàn toàn offline |
| **RAM** | ~8 GB+ | Không đáng kể | ~500 MB |
| **Tên nước ngoài** | Sai (model không nhận ra) | Không hỗ trợ | Hỗ trợ qua multi-model |
| **Giọng mẫu** | Bắt buộc | Không cần | Không cần |
| **Chi phí** | Miễn phí | Miễn phí (unofficial) | Miễn phí |

### Khi nào dùng F5-TTS

- Cần clone đúng giọng người thật
- Không có kết nối internet ổn định
- Muốn kiểm soát hoàn toàn giọng đọc

### Khi nào dùng Edge-TTS

- Ưu tiên chất lượng từ ngữ chính xác nhất
- Không cần giọng đọc cụ thể
- Cần xử lý nhanh (batch lớn)

### Khi nào dùng Piper TTS

- Offline hoàn toàn, không cần Apple Silicon
- Cần tốc độ nhanh với RAM thấp
- Truyện có nhiều tên nhân vật Nhật/Anh (hỗ trợ multi-model)

### Giọng Edge-TTS tiếng Việt

| Voice | Giới tính | Ghi chú |
|-------|-----------|---------|
| `vi-VN-HoaiMyNeural` | Nữ | Mặc định, giọng miền Bắc |
| `vi-VN-NamMinhNeural` | Nam | Giọng miền Bắc |

> **Lưu ý:** Edge-TTS dùng API không chính thức của Microsoft Edge. Hoàn toàn miễn phí nhưng Microsoft có thể thay đổi hoặc chặn bất kỳ lúc nào.

---

## Piper TTS — Tên nhân vật nước ngoài

Truyện isekai tiếng Việt thường chứa tên nhân vật Nhật hoặc Anh viết bằng chữ Latin (Takagi Hiroshi, Thor, Steve…). Model tiếng Việt sẽ đọc sai hoàn toàn vì không có các âm này trong tập huấn luyện.

Piper TTS giải quyết vấn đề này bằng cách **tách văn bản theo ngôn ngữ từng đoạn** và **dùng model riêng biệt** cho từng phần.

### Cơ chế hoạt động

**Phát hiện tên nước ngoài:** Các từ viết hoa hoàn toàn ASCII, từ 4 ký tự trở lên, được coi là tên nước ngoài (ví dụ: `Takagi`, `Hiroshi`, `Thor`, `Steve`). Các từ thuần ASCII tiếng Việt phổ biến (`Trong`, `Nguyen`, `Theo`…) được loại trừ khỏi danh sách này.

**Phân loại Nhật / Anh:** Dựa trên cụm phụ âm:
- **Tên Nhật (romaji):** Chỉ có các cụm phụ âm hợp lệ trong Hepburn (`sh`, `ch`, `ts`, `ky`…) → ví dụ: `Takagi`, `Hiroshi`, `Sano`, `Ichiro`
- **Tên Anh / Tây:** Có cụm phụ âm không tồn tại trong tiếng Nhật (`th`, `st`, `br`…) → ví dụ: `Thor`, `Steve`, `Arthur`

**Routing theo model:**

| Đoạn | Ví dụ | Model dùng |
|------|-------|-----------|
| Tiếng Việt | "đang nói chuyện" | VI model |
| Tên Anh | "Thor", "Steve" | EN model |
| Tên Nhật | "Takagi Hiroshi" | JA model (nếu có) hoặc EN model |

### Chế độ multi-model (khuyến nghị)

Kích hoạt bằng cách trỏ đến model tiếng Anh trong `config.yaml`:

```yaml
piper:
  foreign_model_path_en: "models/piper/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
  foreign_model_path_ja: ""   # Không có JA model chính thức; tên Nhật dùng EN model
```

Khi `foreign_model_path_en` được set, pipeline tự động:
1. Nhận dạng các đoạn tên nước ngoài trong từng chunk
2. Tổng hợp phần tiếng Việt bằng VI model
3. Tổng hợp tên nước ngoài bằng EN model
4. Ghép các đoạn audio lại theo thứ tự

> **Lưu ý về model tiếng Nhật:** Hiện chưa có model Piper tiếng Nhật tương thích với `piper-tts` Python library. Tên nhân vật Nhật sẽ được đọc bằng EN model, cho kết quả tốt hơn đáng kể so với VI model.

### Chế độ IPA injection (fallback)

Khi không cấu hình foreign model (`foreign_model_path_en: ""`), engine dùng phương pháp thay thế:

- Tên Nhật → chuyển đổi Hepburn romaji → IPA → inject `[[IPA]]` vào VI model
- Tên Anh → dùng espeak en-us → inject `[[IPA]]` vào VI model

Bật/tắt bằng `foreign_phoneme: true/false`.

### Override phát âm thủ công

Thêm vào `pronunciation_dict` trong `config.yaml` để ghi đè phát âm của bất kỳ từ nào (hoạt động ở mọi chế độ):

```yaml
piper:
  pronunciation_dict:
    "Thor": "tɔː"       # Override bằng IPA string
    "Takagi": "takaɡi"
```

---

## Chuẩn hoá text tiếng Việt

Trước khi gửi vào TTS, `preprocessor.py` tự động chuyển đổi:

### Ký tự đặc biệt (web novel)

| Input | Output |
|-------|--------|
| `【nhiệm vụ kích hoạt】` | `nhiệm vụ kích hoạt` |
| `thời gian ——còn lại` | `thời gian , còn lại` |
| `"Anh nói:"` | `"Anh nói:"` |

Các bracket CJK `【】〔〕「」『』《》〈〉` được xoá bỏ vì model TTS không đọc được và thường gây lỗi phát âm.

### Định dạng thời gian

| Input | Output |
|-------|--------|
| `0:29:59` | `hai mươi chín phút năm mươi chín giây` |
| `1:30:00` | `một giờ ba mươi phút` |
| `45:30` | `bốn mươi lăm phút ba mươi giây` |

### Số và đơn vị

| Input | Output |
|-------|--------|
| `1` | `một` |
| `15` | `mười lăm` |
| `21` | `hai mươi mốt` |
| `1234` | `một nghìn hai trăm ba mươi tư` |
| `1,000,000` | `một triệu` |

Quy tắc đặc biệt tiếng Việt được áp dụng: `lăm` (không phải `năm`) cho số đuôi 5 sau `mươi`, `mốt` cho số đuôi 1 sau `mươi`, `tư` cho số đuôi 4 sau `mươi`.

---

## Cấu hình giọng tham chiếu (F5-TTS)

F5-TTS cần một đoạn audio mẫu để clone giọng. Chất lượng phụ thuộc nhiều vào file này.

### Chọn đoạn audio phù hợp

```yaml
reference:
  audio_src: "reference/giong_mau.mp3"
  wav_path: "reference/voice.wav"
  start_ms: 2000   # Bỏ qua 2 giây đầu
```

- **`start_ms`**: Bỏ qua N millisecond đầu của file nguồn. Hữu ích khi file có nhạc intro, câu subscribe YouTube, hay tiếng ồn ở đầu.
- Pipeline luôn lấy **4 giây** kể từ `start_ms` làm reference (tối ưu cho F5-TTS).
- Xoá `reference/voice.wav` để pipeline tạo lại với `start_ms` mới.

### Kiểm tra ref_text

Khi F5-TTS load model, nó tự transcribe đoạn reference audio bằng Whisper nội bộ và in ra:

```
[TTS] ref_text: 'nhạc dịu êm, khẽ chạm vào lòng người, gió nhẹ len qua'
```

Nếu `ref_text` chứa nội dung không mong muốn, tăng `start_ms` để bỏ qua phần đó.

### Yêu cầu file giọng mẫu tốt

- Giọng đọc rõ ràng, không có nhạc nền
- Phát âm chuẩn tiếng Việt
- Độ dài gốc ít nhất 6–10 giây
- Format: MP3, WAV, FLAC đều được

---

## Cách sử dụng

> **Luôn dùng `.venv/bin/python`**, không dùng `python` hay `python3` của hệ thống vì macOS mặc định là Python 3.8.

### Xử lý một file

```bash
.venv/bin/python main.py process input/ten_file.txt
```

### Xử lý tất cả file trong thư mục `input/`

```bash
.venv/bin/python main.py process-all
```

### Xem tiến trình xử lý

```bash
.venv/bin/python main.py status input/ten_file.txt
```

Ví dụ output:

```
Status for: input/truyen.txt
  Overall: in_progress
  Chapters done: 3 / 10

  Chapter breakdown:
    [✓] chapter_1: done
    [✓] chapter_2: done
    [✓] chapter_3: done
    [~] chapter_4: in progress (12 chunks synthesized)
    [ ] chapter_5: ...
```

---

## Cấu trúc output

Với file đầu vào `input/truyen_part1.txt`:

```
output/
└── truyen_part1/
    ├── chuong_001.mp3    # Chương 1
    ├── chuong_002.mp3    # Chương 2
    └── chuong_003.mp3    # Chương 3
```

File `chuong_000.mp3` được tạo nếu có phần "Lời mở đầu" (văn bản trước chương đầu tiên dài hơn 100 ký tự).

Nếu file không có tiêu đề chương, toàn bộ nội dung được xuất thành `chuong_001.mp3` mà không đọc thêm từ nào thừa ở đầu.

---

## Nguyên lý hoạt động

```
File .txt
   │
   ▼
[preprocessor]     ← NFC normalize, mở rộng viết tắt, xoá ký tự CJK,
   │                 chuyển số/giờ → chữ tiếng Việt
   ▼
[chapter_detector] ← Regex tìm "Chương X", "Hồi X", "Chapter X"
   │
   ▼  (danh sách Chapter objects)
[chunker]          ← Tách theo câu, gom lại thành chunk ≤200 ký tự,
   │                 gộp chunk ngắn (<50 ký tự) vào chunk liền trước
   ▼  (danh sách Chunk objects)
[engine selector]  ← Đọc cfg["tts"]["engine"]
   │
   ├── "f5"   → [TTSEngine]      ← F5-TTS trên MPS, clone giọng mẫu
   │
   ├── "edge" → [TTSEngineEdge]  ← Edge-TTS qua Microsoft API
   │
   └── "piper"→ [TTSEnginePiper] ← Piper ONNX, phát hiện ngôn ngữ từng đoạn,
   │                               routing VI/EN/JA model theo segment
   ▼  (numpy arrays WAV)
[audio_merger]     ← Ghép chunk + thêm khoảng nghỉ → xuất MP3 192kbps
```

### Phát hiện chương

Các pattern được nhận dạng:

```
Chương 1        CHƯƠNG I        Chương XII
Hồi 5           HỒI 10
Chapter 1       CHAPTER 10
1.              2.              (tiêu đề số đơn giản)
```

Nếu không tìm thấy bất kỳ pattern nào, toàn bộ file được xử lý như một chương (không thêm từ nào vào đầu audio).

### Chunking text

- Tách text theo dấu kết câu (`. ! ? …`) tiếp theo là khoảng trắng + chữ hoa
- Gom các câu lại thành chunk không quá `max_chars` ký tự
- Không bao giờ cắt giữa chừng một câu
- Chunk ngắn hơn `min_chars` được gộp vào chunk liền trước
- Đoạn văn (cách bởi dòng trống) tạo ra chunk `paragraph_break` → thêm 500ms nghỉ trong audio

---

## Resume

Nếu quá trình bị ngắt giữa chừng (tắt máy, Ctrl+C, lỗi mạng), chạy lại cùng lệnh:

```bash
.venv/bin/python main.py process input/ten_file.txt
```

Tool sẽ tự động:
- Bỏ qua các chương đã hoàn thành
- Tiếp tục chương đang dở từ chunk cuối cùng đã synthesis (load lại từ cache)

Dữ liệu resume lưu tại `checkpoints/ten_file.json`:

```json
{
  "chapter_1": "done",
  "chapter_2": "done",
  "chapter_3": { "done_chunks": [0, 1, 2, 3] }
}
```

File checkpoint tự xoá sau khi toàn bộ file hoàn thành.

> **Lưu ý:** Nếu đổi engine, hãy xoá checkpoint và cache cũ trước để tránh dùng audio đã tổng hợp bởi engine khác:
> ```bash
> rm -rf cache/ten_file/ checkpoints/ten_file.json
> ```

---

## Kiểm tra tính toàn vẹn model

Model F5-TTS được tải về từ HuggingFace lần đầu chạy (~1.5 GB). Sau đó kiểm tra:

```bash
.venv/bin/python verify_models.py
```

Hashes SHA256 đã xác minh (nguồn: HuggingFace blob metadata, tháng 3/2025):

| File | SHA256 |
|------|--------|
| `F5TTS_v1_Base/model_1250000.safetensors` | `670900fd14e6c458...` |
| `F5TTS_Base/model_1200000.safetensors` | `4180310f91d592ce...` |
| `F5TTS_v1_Base/vocab.txt` | `2a05f992e00af9b0...` |
| `vocos-mel-24khz/pytorch_model.bin` | `97ec976ad1fd67a3...` |

> **Bảo mật model loading:** F5-TTS v1.1.16 dùng `safetensors` (an toàn tuyệt đối) cho model chính và `weights_only=True` cho vocoder — không có rủi ro arbitrary code execution từ model.

---

## Thư viện sử dụng

| Thư viện | Version | Vai trò | Ghi chú |
|----------|---------|---------|---------|
| `f5-tts` | 1.1.16 | F5-TTS engine, voice cloning | Model dùng safetensors |
| `edge-tts` | ≥6.1.9 | Edge-TTS engine (Microsoft neural) | Unofficial API, miễn phí |
| `piper-tts` | ≥1.4.0 | Piper TTS engine (ONNX offline) | Cần tải model riêng |
| `pydub` | 0.25.1 | Xử lý audio, xuất MP3 | |
| `soundfile` | 0.13.1 | Đọc/ghi WAV | |
| `tqdm` | 4.67.3 | Progress bar | |
| `pyyaml` | 6.0.3 | Đọc config.yaml | Dùng `yaml.safe_load()` |
| `numpy` | 2.4.2 | Xử lý mảng audio | |

**Hệ thống (Homebrew):**
- `python@3.11` — runtime
- `ffmpeg` — được pydub gọi để encode MP3

---

## Hiệu năng ước tính

### F5-TTS (Apple M-series, MPS)

| Metric | Ước tính |
|--------|---------|
| Tốc độ synthesis | ~20–50x real-time |
| 1 chunk (~150 ký tự) | ~20–40 giây |
| 1 chương (~3000 từ, ~5 phút audio) | ~5–10 phút |
| Tải model lần đầu | ~15–30 giây |

> Dùng `nfe_step: 16` thay vì `32` để tăng tốc gấp 2 lần với chất lượng giảm nhẹ.

### Edge-TTS (phụ thuộc kết nối internet)

| Metric | Ước tính |
|--------|---------|
| Tốc độ synthesis | ~1–3 giây/chunk |
| 1 chương (~3000 từ, ~5 phút audio) | ~30–60 giây |
| Tải model | Không cần (server-side) |

### Piper TTS (offline, CPU)

| Metric | Ước tính |
|--------|---------|
| Tốc độ synthesis (VI only) | ~0.1–0.5 giây/chunk |
| Tốc độ synthesis (multi-model) | ~0.2–1 giây/chunk |
| 1 chương (~3000 từ, ~5 phút audio) | ~1–3 phút |
| Tải model lần đầu | ~1–3 giây |
| RAM sử dụng | ~200 MB (VI) / ~400 MB (VI + EN) |

> Với multi-model (VI + EN), lần đầu gặp tên nước ngoài sẽ tốn thêm ~1–2 giây để load EN model, sau đó giữ trong bộ nhớ cho toàn bộ file.

---

## Xử lý sự cố

### Lỗi: `command not found: python`

Luôn dùng đường dẫn đầy đủ đến Python trong venv:

```bash
# Sai
python main.py process input/file.txt

# Đúng
.venv/bin/python main.py process input/file.txt
```

Hoặc activate venv trước:

```bash
source .venv/bin/activate
python main.py process input/file.txt
```

### Lỗi MPS / GPU (F5-TTS)

Đổi `device` trong `config.yaml` sang `cpu`:

```yaml
tts:
  device: "cpu"
```

### Không tìm thấy ffmpeg

```bash
brew install ffmpeg
# Kiểm tra pydub thấy ffmpeg không:
.venv/bin/python -c "from pydub.utils import which; print(which('ffmpeg'))"
```

Nếu output là `None`, thêm Homebrew vào PATH:

```bash
export PATH="/opt/homebrew/bin:$PATH"
```

### Piper: Lỗi `FileNotFoundError` khi load model

File `.onnx` hoặc `.onnx.json` chưa tải về hoặc đặt sai đường dẫn. Kiểm tra:

```bash
ls models/piper/vi/vi_VN/vais1000/medium/
# Phải thấy: vi_VN-vais1000-medium.onnx  vi_VN-vais1000-medium.onnx.json
```

Nếu thiếu, tải lại theo hướng dẫn trong [Bước 5 — Tải model Piper TTS](#bước-5--tải-model-piper-tts-chỉ-cho-piper).

### Piper: Tên nước ngoài vẫn đọc sai

1. Kiểm tra `foreign_phoneme: true` trong config
2. Nếu chưa cấu hình EN model, bật multi-model mode:
   ```yaml
   piper:
     foreign_model_path_en: "models/piper/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
   ```
3. Với tên đặc biệt vẫn sai, thêm override thủ công vào `pronunciation_dict`:
   ```yaml
   piper:
     pronunciation_dict:
       "Thor": "tɔː"
   ```

### Audio F5-TTS có tiếng lạ ở đầu

Nguyên nhân: phần đầu file giọng mẫu chứa nhạc intro hoặc câu không mong muốn. Cách fix: tăng `start_ms` trong `config.yaml`, sau đó xoá `voice.wav` để tạo lại:

```yaml
reference:
  start_ms: 3000   # Thử tăng dần: 1000, 2000, 3000...
```

```bash
rm reference/voice.wav
.venv/bin/python main.py process input/file.txt
```

### Edge-TTS lỗi kết nối

```
edge_tts.exceptions.NoAudioReceived
```

Kiểm tra internet, thử lại. Nếu lỗi liên tục, Microsoft có thể đang block tạm thời — đợi vài giờ hoặc chuyển sang Piper TTS.

### Không nhận dạng được chương

Kiểm tra format tiêu đề trong file `.txt`. Các format được hỗ trợ:

```
Chương 1: Tiêu đề
CHƯƠNG I
Hồi 5
Chapter 10
```

Nếu file không có tiêu đề chương, toàn bộ sẽ xuất thành `chuong_001.mp3`.

### Chất lượng F5-TTS kém

Thử các bước sau theo thứ tự:

1. Tăng `nfe_step` lên `64` (chậm ~2x nhưng chất lượng tốt hơn)
2. Chọn đoạn giọng mẫu sạch hơn (không nhạc nền, phát âm rõ)
3. Điều chỉnh `start_ms` để lấy đoạn đọc tự nhiên nhất
4. Dùng Vietnamese fine-tuned checkpoint `toandev/F5-TTS-Vietnamese` (xem phần `ckpt_file` trong config)
5. Chuyển sang Piper TTS hoặc Edge-TTS để so sánh

### Xoá cache và chạy lại từ đầu

```bash
rm -rf cache/ checkpoints/
.venv/bin/python main.py process input/ten_file.txt
```
