# Overview
ToFNetをPytorchで実装した

# Requirement
- Pytorch

# test_3dim.py
画像の出力がない最小限のToFNetテスト
使用する場合は次の3つの変数を変更する

```
test_dataset_folder: testデータセットのpath
nameprefix:          テストにつける名前
trained_model_path:  ToFNet学習後のパラメータのpath
```

# test_infer_time.py
ToFNetの計算速度を算出できる
  
# test_simple_compare.py
test_3dim.pyの画像結果を保存する機能がついている。
  
# train_val_3dim.py
ToFNetの学習に使用する。  
学習パラメータを変えるときはoption/train-settings.jsonを変更する。


# data/~
データセットの読み込み、前処理のスクリプトがある。

# datasets/~
hdr_toolsのassign_datanumber.pyで作成したデータセットを使用する。  
作成したデータセットはこのフォルダに移すか、保存先を変える。
```
ディレクトリ例

datasets/  
　├ livingroom-train/
　│　└ 00000/
　│　└ 00001/
　│　└ 00002/
　│　└ 00003/
　├ livingroom-val/
　│　└ 00000/
　│　└ 00001/
　├ livingroom-test/
　│　└ 00000/
　│　└ 00001/
　├ bathroom-train/
　│　└ 00000/
　│　└ 00001/
　│　└ 00002/
　└ etc
```

# models/~
ToFNetやTotalVariationのスクリプトが入っている。

# option/~
ToFNetの学習で使用するパラメータの管理を行っている。  
train-settings.jsonを変更して、train_val_3dim.pyを実行すると設定ファイルを反映して学習を開始する。

```
train-settings.jsonの設定例

{
    "workers": 2,
    バッチを取り出すためのプロセス数で2以上を指定しても高速化しない(未確認)

    "batch_size": 20,
    ミニバッチのサイズを大きく方が学習を安定にさせやすい。8, 16, 32あたりで使用する

    "image_size": 256,
    Cropをする前の入力画像の大きさ。256×256

    "num_epochs": 2000,
    学習を行う回数。初めは適当な値で学習し, Lossの値を見て変更する。

    "lr": 0.0002,
    学習率で元論文ではこの値を使用している。

    "lr_decay_step": 100,
    学習率の減衰を100 epochごとに行う。

    "lr_gammma": 0.95,
    減衰の大きさで学習率がlr_gammma倍される。

    "tv_weight": 8e-07,
    Total Variation Lossの大きさ。

    "adv_weight": 0.15,
    Adversarial Lossの大きさ。

    "beta1": 0.5,
    Adamのパラメータ

    "ngpu": 1,
    gpuの数

    "IMG_WIDTH": 128,
    ToFNet処理時(Crop後)の画像サイズ

    "IMG_HEIGHT": 128,
    ToFNet処理時(Crop後)の画像サイズ

    "INPUT_CHANNELS": 3,
    iToFのチャンネル数。

    "OUTPUT_CHANNELS": 1,
    ToFNetはDepthMapを出力する。

    "train_idname": "appendix3-flip-corner-v3-3rd-adv015-tv8e-7-beta05",
    学習ごとの区別がつくように名前をつける。例えば"データセット名+特徴+ハイパーパラメータ"

    "training_datasets_folder_name": "datasets/v3-4cont-appendix3",
    使用する学習データセットを指定する。

    "valid_datasets_folder_name": "datasets/v3-data-cont-val-3rd",
    検証データセットを指定する。

    "training_results_folder_name": "results",
    特に変更する必要はない。resultsフォルダに学習過程を保存する。
    resultsフォルダにはG-Loss-log.csvには学習時のLossの値を保存している。
    .json設定ファイルも保存されている。

    "training_param_folder_name": "checkpoints",
    学習後のToFNetのパラメータが保存されている。

    "option_purpose": "156 images beta1"
    学習の目的の説明でToFNet中では使用しない。メモとして使用する変数。
}
```
---
  
# Note
ソースコード中に具体的な注意点を書いているので
コメント, READMEを読んでから変更するとエラーが減ります。
