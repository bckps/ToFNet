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


# models/~
ToFNetやTotalVariationのスクリプトが入っている。

# option/~
ToFNetの学習で使用するパラメータの管理を行っている。  
train-settings.jsonを変更して、train_val_3dim.pyを実行すると設定ファイルを反映して学習を開始する。


---
  
# Note
ソースコード中に具体的な注意点を書いているので
コメントを読んでから変更するとエラーが減ります。
