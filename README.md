# 🧠 svm_core

**Semantic Virtual Memory Engine — Powered by OpenCL 2.0 SVM**

This repository provides an experimental, yet fully functional SVM memory orchestration engine using:
- Fine-Grain Shared Virtual Memory (Zero-Copy)
- `enqueue_svm_map` / `enqueue_svm_unmap` flows
- Bypassed `clSetKernelArg` logic for true shared pointer access
- AMD RX 5700 tested (gfx1010:xnack-)

---

## 🔓 Licensing

- ✅ **Free for personal / non-commercial use** under the [Open Semantic License](./LICENSE.osl.txt)
- ❌ **Commercial use is strictly prohibited** without an explicit [Commercial License Agreement](./LICENSE.cla.txt)

To apply for a commercial license, contact:
📧 ice.xu@retryixagi.com

---

## 🚀 What's Included
- `svm_core_claude_safe.py`: Full-featured version with memory tracing
- `svm_core_lite_limited.py`: Restricted public version
- Examples and diagnostics included

---

## 🧬 Version Comparison

| Feature / Module                       | `svm_core_claude_safe.py` (Safe Full) | `svm_core_lite_limited.py` (Restricted) |
|----------------------------------------|----------------------------------------|------------------------------------------|
| Fine-Grain SVM (Zero-Copy Access)      | ✅ Enabled                             | ❌ Disabled (Fallback to host buffer)     |
| `enqueue_svm_map` / `unmap` Support    | ✅ Fully implemented                   | ❌ Removed                                |
| `clSetKernelArg` Bypass                | ✅ Works via true shared pointer       | ⚠️ Simulated only (non-functional stub)   |
| Memory Tracing & Validation            | ✅ Included                             | ❌ Not available                          |
| GPU Compatibility Layer                | ✅ Supports gfx1010:xnack- and above   | ⚠️ Partial, no xnack optimization         |
| Legal Status                           | 🔒 Not redistributable without license | 🔓 Free for personal use under OSL       |
| Intended Use                           | Internal R&D / Licensed Deployment     | Community Testing / Educational Use      |

> ❗ The **Lite** version is intentionally restricted to prevent unauthorized commercial use or reverse engineering of the full SVM orchestration flow.

> 🔒 The **Claude Safe** version is available only under explicit licensing conditions.

---

## ❗ The Truth Behind So-Called "SVM Support"

Several vendors — including Intel — have long claimed support for OpenCL 2.0 SVM (Shared Virtual Memory).  
But in reality, these claims are misleading, incomplete, or deliberately obscured. Here’s why:

1. **No public implementation was ever provided.**  
   Intel holds patents related to unified memory, yet failed to deliver a working GPU-wide SVM system.

2. **“Support” was limited to CPU-side or FPGA-only contexts.**  
   GPU SVM support was quietly removed or disabled in their drivers — rendering it practically useless.

3. **The ecosystem was abandoned.**  
   Once the marketing buzz faded, most vendors dropped real support, shifting toward proprietary CUDA/HIP stacks.

🔓 This project provides the first developer-usable, fine-grain, zero-copy SVM memory orchestration engine, tested and validated on AMD RX 5700 (gfx1010:xnack-). No wrappers. No vaporware. No driver magic.

> This is what Intel and others could not — or would not — give you.

---

## ❗ 關於「支援SVM」的技術真相

包括Intel在內的多家廠商，長期對外宣稱「支援 OpenCL 2.0 的共享虛擬記憶體（SVM）」。  
但事實上，這些聲明要嘛是誤導、要嘛是策略性閃避，原因如下：

1. **從未公開任何完整可運行的實作。**  
   Intel雖擁有相關專利，卻始終未能釋出能實際執行的GPU SVM系統。

2. **僅限CPU或FPGA上下文，GPU端被悄悄禁用或屏蔽。**  
   所謂支援實際上在驅動層被封鎖或回退，完全無法實戰部署。

3. **整個SVM生態後來遭到放棄。**  
   隨著市場熱度下降，各家開始改推自有CUDA / HIP平台，對外號稱「轉型」，實則迴避。
   
   
---

## 📜 Provenance and Time-Locked Publication

The full-featured version of this system (`svm_core_claude_safe.py`) was privately committed to this repository prior to public release.

- Commit hash: `a8c9f3e...` (placeholder, fill with real hash)
- Commit timestamp: 2025-08-03TXX:XX:XXZ
- Stored securely under private GitHub infrastructure

This serves as:
- Cryptographic proof of authorship
- Legal evidence of prior art and independent implementation
- Tamper-resistant timestamp against future IP challenges

> The code was ready. The world just wasn't.

🔓 本專案為開發者提供首個可實際部署的SVM共享記憶系統，並已在AMD RX 5700（gfx1010:xnack-）成功驗證。  
不靠包裝，不靠驅動魔法，不吹牛，能跑就是真相。

> 這是Intel與其他廠商不願意，也無法提供給你的技術自由。
