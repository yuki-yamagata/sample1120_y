# はじめに

取得したマウスデータと、SAM2の事前学習モデルを用いてビデオのセグメンテーションを簡易試行する手順です。
QuPathと接続できれば楽なのかもですが、一旦ソースコードで簡易試行した内容です。

【SAM2】
https://github.com/facebookresearch/sam2

【SAM2 Vide Prediction DEMO】
https://github.com/facebookresearch/sam2/blob/main/notebooks/video_predictor_example.ipynb

# 環境構築

## SAM2

上述のリンクのインストール手順に従えば問題ないですが、参考までに、以下に実行例を示します。

1. pythonの任意の仮想環境をactivateします。

    condaを使った場合の例。
    ```bash
    conda create -n sam2 python=3.12
    conda activate sam2
    ```

1. 任意のディレクトリで下記を実行し、sam2のソース一式のダウンロードとライブラリのインストールを行います。

    ```bash
    git clone https://github.com/facebookresearch/sam2.git && cd sam2
    pip install -e .    
    ```
## 簡易試行用ライブラリのインストール

    ```bash
    pip install opencv-python matplotlib Pillow pathlib
    ```
    過不足あったらすみません。。

## ファイルの準備

1. このREADMEと同じディレクトリに置いてある```sample```フォルダを、先ほど配置したsam2/直下に配置します。

1. ```sample```ディレクトリに移動します。

    ```bash
    cd sample
    ```

1. videoのフレーム分割（オプション）

    videoをフレームごとの画像ファイルに分割します。
    ※すでに分割したファイルを配置済みのため、サンプルで動かす分には不要です。

    ```bash
    ffmpeg -i videos/D1AT_20251029_Trim_night_30sec.mp4 -q:v 2 -start_number 0 videos/night_30sec/'%05d.jpg'
    ```


# セグメンテーション

以下の実行される中身は、ほぼSAM2の公式DEMOと同じ内容（というか少し端折っていて情報量が少ない）なので、./notebooks/video_predictor_example.ipynbのコードを順に動かしたほうが、理解自体は深まると思います。

## セグメンテーションの実行

現時点では簡易試行のため、対象のオブジェクトを示す座標は事前に取得済みの前提で、いったん進めます。

今回は、```videos/D1AT_20251029_Trim_night_30sec.mp4```を用います。


1. 以下のコードをターミナルで実行します。
    ```
    python segmentation.py
    ```

1. 0フレーム目の画像がポップアップしてくるので、１個体目のマウスの座標をクリックして、Enterキーを押します。

1. 0フレームのみでの１個体目のセグメンテーション結果が表示されるので、確認して閉じます。

1. また0フレーム目の画像がポップアップしてくるので、２個体目のマウスの座標をクリックして、Enterキーを押します。

1. 0フレームのみでの２個体目のセグメンテーション結果が表示されるので、確認して閉じます。

1. 全フレームの認識が始まります。セグメンテーションされた画像群が```./videos/*_segmented```に、位置情報（重心）が```./position```に、出力されます。（重心の計算・プロット自体はSAM2の機能を呼び出しいているわけではございません）
