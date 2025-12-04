# SAM2 マウストラッキング

SAM2（Segment Anything Model 2）を使って動画からマウスをセグメンテーションし、重心位置を追跡するツールです。

## 📁 リポジトリ構成

```
sample1120_y/
├── segmentation.py          # メインのセグメンテーションスクリプト
├── plot_trajectory.py       # 軌跡可視化スクリプト
├── SAM2_mouse_tracking.ipynb # Jupyter Notebook版
├── requirements.txt         # Python依存パッケージ
├── videos/                  # 動画ファイル（※Git管理外）
├── positions/               # 出力CSV
├── outputs/                 # 可視化出力（※Git管理外）
└── sam2/                    # SAM2本体（※Git管理外、別途取得）
```

## 🚀 クイックスタート

### 1. リポジトリをクローン

```bash
git clone git@github.com:yuki-yamagata/sample1120_y.git
cd sample1120_y
```

### 2. Python仮想環境を作成

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. SAM2 をインストール（重要）

SAM2 はサイズが大きいため、このリポジトリには含まれていません。以下のコマンドで取得・インストールしてください：

```bash
# SAM2 をクローン
git clone https://github.com/facebookresearch/sam2.git

# SAM2 をインストール
pip install ./sam2

# チェックポイント（学習済みモデル）をダウンロード
cd sam2/checkpoints
./download_ckpts.sh
cd ../..
```

### 4. 動画ファイルを配置

`videos/` フォルダに解析したい動画ファイル（MP4）を配置してください。

### 5. セグメンテーションを実行

```bash
# 仮想環境を有効化
source .venv/bin/activate

# セグメンテーション実行（対話的にマウスをクリックで指定）
python segmentation.py \
  --video-mp4 videos/your_video.mp4 \
  --frames-dir videos/your_video_frames \
  --output-csv positions/mouse_tracks.csv \
  --num-objects 2
```

### 6. 軌跡を可視化

```bash
# PNG画像とMP4アニメーションを出力
python plot_trajectory.py \
  --csv positions/mouse_tracks.csv \
  --fps 25 \
  --save-png \
  --save-anim \
  --frames-dir videos/your_video_frames
```

出力は `outputs/` フォルダに保存されます。

---

## 📋 動作要件

- **Python**: 3.10〜3.12
- **OS**: macOS (Apple Silicon推奨) または Linux (CUDA推奨)
- **ffmpeg**: 動画からフレーム抽出に使用

### macOS の場合

PyTorch の MPS（Metal）バックエンドが自動で使用されます。

```bash
# MPS が使えるか確認
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

### Linux (CUDA) の場合

CUDA対応の PyTorch をインストールすると高速です：

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

---

## 🔧 トラブルシューティング

| 問題 | 対処法 |
|------|--------|
| `ModuleNotFoundError: No module named 'sam2'` | `pip install ./sam2` を実行 |
| MPS が `False` (macOS) | macOS 12.3以降 + Python 3.10以降が必要 |
| チェックポイントが見つからない | `sam2/checkpoints/download_ckpts.sh` を実行 |
| ffmpeg が見つからない | `brew install ffmpeg` (macOS) または `apt install ffmpeg` (Linux) |

---

## 📊 出力ファイル

### CSV形式（`positions/mouse_tracks_*.csv`）

```csv
object_id,frame_idx,cx,cy
1,0,443.04,911.11
1,1,442.43,910.82
2,0,1313.04,227.93
...
```

- `object_id`: マウスのID（クリック順）
- `frame_idx`: フレーム番号
- `cx`, `cy`: 重心座標（ピクセル）

### 可視化出力（`outputs/`）

- `*_time_series.png`: x座標・y座標の時系列グラフ
- `*_2d.png`: 2D軌跡プロット
- `*_traj.mp4`: アニメーション動画

---

## 📚 詳細ドキュメント

より詳しい手順は [`README_sam2_refined.md`](./README_sam2_refined.md) を参照してください。

## 🔗 参考リンク

- [SAM2 公式リポジトリ](https://github.com/facebookresearch/sam2)
- [SAM2 Video Predictor Demo](https://github.com/facebookresearch/sam2/blob/main/notebooks/video_predictor_example.ipynb)

---

## 📝 開発メモ（自分用）

### 変更を GitHub に反映する手順

```bash
# 1. 変更を確認
git status

# 2. 変更をステージング
git add .

# 3. コミット
git commit -m "変更内容を書く"

# 4. push
git push origin main
```
