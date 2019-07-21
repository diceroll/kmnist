# ProbSpace 「くずし字」識別チャレンジ

<https://prob.space/competitions/kuzushiji-mnist>

## 使い方

### 必要なライブラリ

* numpy
* pandas
* Pillow
* tqdm
* chainer
* cupy
* chainercv
* albumentations

### 学習

はじめに、**output**という名前のディレクトリを作成してください

* 基本的な学習コマンド  
`python train.py -g GPU番号 -dn 出力ディレクトリ名`  
※その他オプションは`python train.py -h`  

* 5-fold CVの例  
`./run.sh`

### テスト

* 学習済みモデルを使用する場合
    1. [こちら](https://drive.google.com/open?id=1LZFqHxDHabMuxysiNOnETX5xz0AoVvzb)からzipファイル(1.8GB)をダウンロード
    1. zipファイルを展開
    1. `python test.py -g GPU番号`

* 自分で学習したモデルを使用する場合
    1. test.py内の以下の部分を変更してください

    ```
    models = [
            ('SEResNeXt101', None,
            ['output/20190611_2314/best_model.npz',
            'output/20190612_0039/best_model.npz',
            'output/20190612_1230/best_model.npz',
            'output/20190612_1537/best_model.npz',
            'output/20190612_2333/best_model.npz']),
            ('SEResNeXt101', resize,
            ['output/20190613_1544/best_model.npz',
            'output/20190613_1609/best_model.npz',
            'output/20190614_0537/best_model.npz',
            'output/20190614_0852/best_model.npz',
            'output/20190614_1808/best_model.npz']),
        ]
    ```

    ※`models = [(モデル名, 前処理, モデルのパスのリスト)]`  

    2. `python test.py -g GPU番号`
