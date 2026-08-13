# LAD-Net 重現 — 架構完成,主要資料集缺失,泛化測試部分可做

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white)
![Data](https://img.shields.io/badge/AppleSet6-unavailable-e05d44)
![Data](https://img.shields.io/badge/Tomato9-partial-dbab09)
![Reproduction](https://img.shields.io/badge/status-partial-dbab09)
![License](https://img.shields.io/badge/license-MIT-2ea44f)

重現的論文:
> "LAD-Net: A Novel Light Weight Model for Early Apple Leaf Pests and Diseases
> Classification", IEEE/ACM TCBB, 2023. 沒有找到官方程式碼;這是純粹從論文
> 文字獨立重新實作的 clean-room 版本。

## Pretrained Models

Tomato9 訓練出的權重檔案不大(1.2~1.4MB),直接 commit 進 repo,不用另外
發 Release:

| 模型 | 架構版本 | Test Accuracy | 路徑 |
|---|---|---|---|
| `tomato9_seed1_adfix/best_model.pt` | 新架構(LR-CBAM 用 AD 分解,294,960 參數) | **97.73%** | [runs/tomato9_seed1_adfix/best_model.pt](runs/tomato9_seed1_adfix/best_model.pt) |
| `tomato9_seed1/best_model.pt` | 舊架構(LR-CBAM 用標準卷積,356,400 參數) | 97.73% | [runs/tomato9_seed1/best_model.pt](runs/tomato9_seed1/best_model.pt) |

載入方式:

```python
import torch
from src.model import LADNet

model = LADNet(num_classes=9)  # Tomato9
model.load_state_dict(torch.load("tomato9_seed1_adfix/best_model.pt", map_location="cpu"))
model.eval()
```

## 重現結果

**主要實驗(AppleSet6,Table 7/8/9)**:沒有數字——資料集拿不到,完全沒
訓練過。唯一能比較的是架構規模:

| 指標 | 論文 | 重現結果 |
|---|---|---|
| 參數量 | 約 0.32M | **294,960**(−8%) |
| 模型大小 | 1.25 MB | 約 1.18 MB(−6%) |

**2026-08-11 更新**:LR-CBAM 內部的卷積原本用一般標準卷積(理由:「LR-CBAM」
命名沒有「AD」兩個字,跟「AD Convolution」「LAD-Inception」分開命名,合理
推測只有明確標 AD/LAD 的部分才用非對稱分解)。後來找到論文 Conclusion
明確寫「AD 卷積用來取代所有標準卷積」,推翻了這個假設——改成 LR-CBAM 內部
也用 1×k+k×1 分解後,參數量從 356,400(+11%)降到 294,960(−8%),從
「超出論文」變成「略少於論文」,但整體誤差量級差不多,算是更貼近論文原意
的版本。詳見 `src/model.py` 的 module docstring「Known deviations #6」。

**泛化測試(Tomato9,Table 10)**:拿到真正的 AI Challenger 2018 原始
資料集,組出了 Tomato9 的 9 個可用類別(缺 Bacterial_spot,詳見下方
「Tomato9 資料來源」),用論文 Table 6 的協定訓練(Adam、lr=5e-4、
batch=32、cosine annealing、200 epochs):

| 指標 | 論文 | 舊架構(356,400 參數) | 新架構(294,960 參數,LR-CBAM 用 AD 分解) |
|---|---|---|---|
| Accuracy | 97.92% | 97.73% | **97.73%** |
| Precision(macro) | — | 96.97% | 96.58% |
| Recall(macro) | — | 94.37% | 94.49% |
| F1(macro) | — | 95.54% | 95.43% |

準確率兩個架構幾乎一模一樣,但新架構參數量少了 17%——LR-CBAM 改用 AD
分解沒有犧牲準確率,純粹是效率提升,也更貼近論文 Conclusion 明講的
「AD 卷積取代所有標準卷積」這個設計原則。

`src/model.py::LADNet` 的 forward pass 已經驗證過(`python src/model.py`
可以直接跑)。

## 主要資料集狀況(AppleSet6 / LateAppleSet)

| 資料集 | 用途 | 狀態 |
|---|---|---|
| **AppleSet6**(原始 821 張 / 增強後 11,820 張,6 類) | 主要實驗 — Table 7、8、9 | ❌ 作者自己在乾縣蘋果監測站拍的私有資料集,論文裡完全沒有提供任何公開下載連結 |
| **LateAppleSet**(5 類) | 泛化測試 — Table 10 | ❌ 名義上公開,但放在百度 AI Studio 上;`aistudio.baidu.com` 回傳 HTTP 403(需要登入),這個環境沒有帳號憑證 |

## Tomato9 資料來源與處理(2026-07-29 更新)

使用者提供了 AI Challenger 2018 農作物病害辨識資料集的原始檔案(train +
validation + testA + testB,共 4 個 zip,約 4.1GB,RAR 打包)。testA/testB
沒有附標籤(比賽保留用的),只有 train(31,718 張)和 validation(4,540
張)有完整的 `disease_class` 標註。

**類別對照表**來源:[foamliu/Crop-Disease-Detection](https://github.com/foamliu/Crop-Disease-Detection)
的 `labels.csv`(61 類的「作物-病害-嚴重程度」標準命名)。番茄對應
class ID 41(healthy)、42-43(白粉病)、44-45(細菌性斑點病)、46-47(早疫病)、
48-49(晚疫病)、50-51(葉黴病)、52-53(斑點病)、54-55(斑枯病)、
56-57(紅蜘蛛)、58-59(黃化捲葉病毒)、60(花葉病毒)。

`data_prep/extract_tomato9.py` 把一般/嚴重兩個等級合併成同一類,並且
**排除白粉病(42-43)**——PlantVillage 的番茄分類本來就沒有白粉病,排除
它之後,AI Challenger 的番茄子集(11 個基礎類別)剛好變成 Tomato9(9 種
病害 + healthy = 10 類),跟論文命名完全吻合。

**已知資料限制**:實際檢查發現 class 44/45(細菌性斑點病)在原始資料裡
幾乎是空的(train 只有 1 張、val 0 張)。查資料時看到社群文章提過競賽
主辦方後來把 44、45 這兩類直接刪除,這裡的結果驗證了這件事——不是
`extract_tomato9.py` 抓漏,是 AI Challenger 這個資料集本身的限制。目前
的處理方式是 `data_prep/make_tomato9_split.py` 把 `Tomato___Bacterial_spot`
排除在訓練切分之外,實際訓練用的是其他 9 類(8 種病害 + healthy)。

用 `data_prep/make_tomato9_split.py` 依照論文 Table 6 附近提到的
4:1:1 比例(跟 AppleSet6 主實驗一樣)做 train/val/test 切分:

| 類別 | train | val | test |
|---|---|---|---|
| Tomato___Early_blight | 528 | 132 | 132 |
| Tomato___Late_blight | 1046 | 262 | 261 |
| Tomato___Leaf_Mold | 503 | 126 | 126 |
| Tomato___Septoria_leaf_spot | 935 | 234 | 234 |
| Tomato___Spider_mites | 619 | 155 | 155 |
| Tomato___Target_Spot | 49 | 12 | 13 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 2961 | 740 | 741 |
| Tomato___Tomato_mosaic_virus | 199 | 50 | 49 |
| Tomato___healthy | 921 | 230 | 230 |

用 `src/train.py` 訓練(Adam、lr=5e-4、batch=32、cosine annealing、200
epochs,對應論文 Table 6),結果見上方「重現結果」。

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
- `src/train.py` — 訓練腳本,照論文 Table 6 的協定(Adam、lr=5e-4、
  batch=32、cosine annealing、200 epochs)。
- `data_prep/extract_tomato9.py` — 從 AI Challenger 2018 原始標註 JSON
  裡篩出 Tomato9 對應的類別,合併嚴重程度。
- `data_prep/make_tomato9_split.py` — 對 Tomato9 做 4:1:1 的
  train/val/test 切分。

## 如果之後拿到 AppleSet6 或 LateAppleSet

`LADNet(num_classes=N)` 可以接受任意類別數,`src/train.py` 可以直接套用
在任何符合 `ImageFolder` 結構(train/val/test 各自一個資料夾,底下按類別
分子資料夾)的資料集上,只需要調整 `--data_dir` 和 `--num_classes`。

## 環境需求

Python 3.10+、PyTorch(建議 CUDA 版)、scikit-learn、matplotlib。

## 授權

本 repo 程式碼採用 MIT 授權。不包含任何資料集圖片——AI Challenger 2018
資料集需要自行取得,原始授權見該資料集的使用條款。
