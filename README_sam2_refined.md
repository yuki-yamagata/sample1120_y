# SAM2 セグメンテーション環境セットアップ（洗練版）

この文書は テクニカルスタッフから受領した `sample_README.md` をもとに、macOSでのvenv利用、LinuxでのCUDA利用を含む、より再現性の高い手順に整備したものです。SAM2（Segment Anything Model 2）を使ってビデオからマスク（領域分割）を得て、結果画像と重心位置CSVを出力します。

- 元の手順: `sample_README.md`
- 本書の対象: 初学者〜中級者、macOS (Apple Silicon/MPS)・Linux (CUDA) を想定

## 動作要件

- Python 3.10〜3.12
- macOS: Apple Silicon (M1/M2/M3) 推奨。GPUはCUDA非対応なので、PyTorchのMPS（Metal）バックエンドを使用します。
- Linux: NVIDIA GPU + CUDAがあると高速。CPUのみでも動作は可能ですが遅いです。

## 1. 仮想環境（venv）の作成と有効化

`/Users/yamagatayuki/Documents/fromInabasan/251120/sample1120_y` を作業ディレクトリにします。

```bash
cd /Users/yamagatayuki/Documents/fromInabasan/251120/sample1120_y
python3 -m venv .venv
source .venv/bin/activate
python -V  # Pythonバージョン確認（3.10〜3.12推奨）
```

## 2. 依存パッケージのインストール

まず基本ライブラリを入れます。

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` には OpenCV / Matplotlib / NumPy / Pillow / tqdm などの基本ライブラリを記載しています。

## 3. SAM2の取得とインストール

このフォルダには `sam2/` ディレクトリを同梱しています（`scripts/setup_sam2.sh` 実行済みであれば再取得不要）。最新版を取得したい場合やフォルダが無い場合は下記を実行してください。

```bash
# 作業ディレクトリは sample1120_y のまま
git clone https://github.com/facebookresearch/sam2.git
pip install ./sam2
```

### チェックポイント（事前学習モデル）のダウンロード

SAM2同梱のスクリプトでモデルをダウンロードします。

```bash
pushd sam2/checkpoints
./download_ckpts.sh
popd
```

- 失敗する場合は `curl`/`wget` のインストールやネットワークを確認してください。

## 4. macOS（MPS）/ Linux（CUDA）向けPyTorch

- macOS: 近年のPyTorchはMPSが同梱されるため、追加インストール不要のことが多いです。MPSが使えるかの簡易確認:

```bash
python - <<'PY'
import torch
print('Torch:', torch.__version__)
print('MPS available:', torch.backends.mps.is_available())
PY
```

- Linux(CUDA): CUDA対応のPyTorchをインストールすると高速です。環境に合わせて以下を参考にしてください（例: CUDA 12.1）。

```bash
# 参考例: CUDA 12.1 のPyTorch（環境により異なります）
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

CPUのみの場合は通常のPyTorchで構いません。

## 5. データ配置と前処理（オプション）

- ビデオは `videos/D1AT_20251029_Trim_night_30sec.mp4` に配置済み想定。
- `segmentation.py` は `videos/night_30sec/` にJPEGが無い場合、自動的にMP4から切り出します。手動で切り出したい場合は以下を利用してください。

```bash
ffmpeg -i videos/D1AT_20251029_Trim_night_30sec.mp4 -q:v 2 -start_number 0 videos/night_30sec/'%05d.jpg'
```

## 6. 実行方法

対象オブジェクト（マウス）を0フレーム目で2個体クリック指定し、Enterで確定します。すると全フレームにわたりセグメンテーション・位置算出が行われます。

対象オブジェクト（マウス）を0フレーム目で指定した個体数だけクリックし、Enterで確定します。すると全フレームにわたりセグメンテーション・位置算出が行われます。

```bash
# venvが有効な状態で（例: 夜間ビデオを使用）
python segmentation.py

# 別動画を使う例
python segmentation.py \
  --video-mp4 videos/D1AT_20251029_Trim_day_30sec.mp4 \
  --frames-dir videos/day_30sec \
  --output-csv positions/mouse_tracks_day.csv \
  --num-objects 2
```

- 出力: 画像は `videos/*_segmented/`、重心座標CSVは `positions/` に保存されます。
- `--num-objects` でクリックする個体数を指定できます（デフォルト2）。
- 各ウィンドウを閉じる前に必ずクリックしてください。クリックせず閉じるとエラーになります。
- フレーム抽出が走る場合、完了まで数十秒かかることがあります。

- 出力: 画像は `videos/*_segmented/`、重心座標CSVは `positions/` に保存されます。

## 7. よくあるトラブルと対処

- OpenCVやMatplotlibが見つからない: `pip install -r requirements.txt` を再実行。
- SAM2のimport失敗: `pip install ./sam2` の再実行と、仮想環境の有効化 `source .venv/bin/activate` を確認。
- MPSがfalse（macOS）: macOSのバージョンとPython/PyTorchの互換性を確認。MPS未対応でもCPUで実行は可能ですが遅くなります。
- CUDA未認識（Linux）: 適切なCUDA版PyTorchをインストールし、`nvidia-smi` でGPU認識を確認。
- フレーム抽出でエラー: 抽出先ディレクトリに書き込み権限があるか、`--frames-dir` が正しいか確認してください。
- 「You're likely running Python from the parent directory...」と表示されたら、`pip uninstall SAM-2` のあと `pip install ./sam2` を実行し、editableインストールを避けてください。

## 8. 解析結果の可視化（任意）

重心CSVを簡単に可視化するには `plot_trajectory.py` が利用できます。

```bash
# venvが有効な状態で
python plot_trajectory.py --csv positions/mouse_tracks_night.csv --fps 25
```

- 引数を省略すると夜間CSV・FPS=25を用います。
- 可視化専用のためGPUは不要です。

## 9. 自動セットアップスクリプト（任意）

手順を自動化した `scripts/setup_sam2.sh` を用意しました。実行でvenv作成、依存インストール、SAM2取得＋チェックポイントDLまでまとめて行います。

```bash
# sample1120_y ディレクトリで
bash scripts/setup_sam2.sh
```

## 10. Jupyter Notebookでの実行

対話的に解析を進めたい場合は、用意したJupyter Notebookを使用できます。

```bash
# sample1120_y ディレクトリで
source .venv/bin/activate
jupyter notebook SAM2_mouse_tracking.ipynb
```

ブラウザでノートブックが開き、セルを順に実行することでセグメンテーションから可視化まで実行できます。

## 11. 参考

- SAM2 公式: <https://github.com/facebookresearch/sam2>
- Video Predictor Demo: <https://github.com/facebookresearch/sam2/blob/main/notebooks/video_predictor_example.ipynb>
