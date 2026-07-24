# LAD-Net 重現 — 只有架構,沒有資料

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white)
![Data](https://img.shields.io/badge/data-unavailable-e05d44)
![Reproduction](https://img.shields.io/badge/status-architecture--only-e05d44)
![License](https://img.shields.io/badge/license-MIT-2ea44f)

重現的論文:
> "LAD-Net: A Novel Light Weight Model for Early Apple Leaf Pests and Diseases
> Classification", IEEE/ACM TCBB, 2023. 沒有找到官方程式碼;這是純粹從論文
> 文字獨立重新實作的 clean-room 版本。

## 重現結果

沒有 accuracy/F1/latency 這些數字——論文的三個資料集都拿不到(詳見下方),
所以完全沒有訓練過。唯一能比較的是架構規模:

| 指標 | 論文 | 重現結果 |
|---|---|---|
| 參數量 | 約 0.32M | 356,400(+11%) |
| 模型大小 | 1.25 MB | 約 1.43 MB(+14%) |

`src/model.py::LADNet` 的 forward pass 已經驗證過(`python src/model.py`
可以直接跑),參數量跟論文宣稱的數字差距約 11%——比 ConViTX 重現的參數量
差距更接近,但還是沒有完全對上,因為論文有幾個架構細節沒有寫清楚(詳見
下方「已知偏差」)。

## 範圍說明:只有架構,沒有跑過任何訓練或評估

論文的三個資料集,在這個環境裡全部都拿不到:

| 資料集 | 用途 | 狀態 |
|---|---|---|
| **AppleSet6**(原始 821 張 / 增強後 11,820 張,6 類) | 主要實驗 — Table 7、8、9 | ❌ 作者自己在乾縣蘋果監測站拍的私有資料集,論文裡完全沒有提供任何公開下載連結 |
| **LateAppleSet**(5 類) | 泛化測試 — Table 10 | ❌ 名義上公開,但放在百度 AI Studio 上;`aistudio.baidu.com` 回傳 HTTP 403(需要登入),這個環境沒有帳號憑證 |
| **Tomato9**(9 類,AI Challenger 2018 子集) | 泛化測試 — Table 10 | ❌ AI Challenger 官網(`challenger.ai`)已經下線(連線被拒絕),現存的只剩百度網盤分享(提取碼 "iksk"),一樣有登入門檻 |

依照使用者的決定,沒有用替代資料集湊數(例如用別的公開蘋果葉病害資料集
取代 LateAppleSet)——這個 repo **只做架構實作跟驗證**:確認 forward pass
的張量形狀、比對參數量跟論文宣稱的規模。

## 已知偏差 / 論文本身沒講清楚的地方

完整清單見 `src/model.py` 的 module docstring(跟這裡同步更新)。摘要如下:

1. **AD Convolution 每個分支的(kernel、dilation、padding)** 只有在
   LAD-Inception 的三個多尺度分支裡有具體給出(Fig. 4:1×3/3×1 的組合,
   dilation 分別是 1、2、3)。stem 那層的 AD Convolution(Table 1 第一列)
   沿用同一種 asymmetric 分解方式。
2. **LAD-Inception 每個分支的 channel 寬度**沒有給——只知道整個 block 的
   總輸出(從 192 channel 輸入變成 272 channel 輸出)。這裡分成
   64+64+64+64+16 給 5 個分支(3 個 AD-conv 分支 + 1 個單純 1×1 + 1 個
   maxpool→1×1)——是取整數湊出來的,不是論文寫的數字。
3. **CBAM** 採用 Woo et al. 2018 的標準設計(論文直接引用 [28],沒有描述
   任何修改)。
4. **Stem stride 矛盾**:Table 1 把 stem 的 AD Convolution 標成
   「3×3/stride 2」,但它自己寫的形狀變化(224×224 → 56×56)其實是 4 倍
   的空間縮小,不管怎麼組合 padding/dilation,單一 stride-2 的 conv 都做不
   出這個結果。這裡用 **stride=4**(保留 kernel=3、dilation=3)去湊出
   Table 1 宣稱的輸出形狀——優先對齊 Table 1 的形狀,而不是那個看起來
   不一致的 stride 標示。
5. **兩層 MaxPool**(LAD-Inception 前的 28→14;之後的 14→7)標示
   kernel=3/stride=2,沒有給 padding;padding=0 的話會得到 13 和 6,不是
   Table 1 寫的 14 和 7。這裡兩處都用 **padding=1** 去對齊 Table 1 宣稱的
   形狀。

## 檔案說明

- `src/model.py` — 完整架構:`ADConv`(asymmetric+dilated conv)、
  `CBAM`/`ChannelAttention`/`SpatialAttention`、`CBLR`、`LRCBAM`、
  `LADInception`、`LADNet`。直接執行(`python src/model.py`)可以跑
  forward pass 的形狀/參數量驗證。

## 如果之後拿到 AppleSet6 或類似的資料集

`LADNet(num_classes=N)` 可以接受任意類別數。訓練腳本可以參考
`PiTLiD_repro/src/train_apple_pitlid.py` /
`ConViTX_repro/src/train_plantvillage.py` 的寫法(Adam、lr=5e-4、
batch=32、cosine-annealing LR,對應 Table 6,200 epochs),等真的拿到
資料之後這會是下一步。

## 環境需求

Python 3.10+、PyTorch(建議 CUDA 版)。沒有其他依賴——這個 repo 目前只有
架構程式碼,還沒有訓練/資料處理流程。

## 授權

本 repo 程式碼採用 MIT 授權。
