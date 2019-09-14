[![Build Status](https://travis-ci.org/takahi-i/lecture-tmu-2019.svg?branch=master)](https://travis-ci.org/takahi-i/lecture-tmu-2019)

# lecture-tmu-2019

首都大学東京で実施した講義の資料です。

## 準備

- [Docker version 17 or later](https://docs.docker.com/install/#support)

## 開発環境のセットアップ

以下のコマンドを実行して、開発環境を構築します。以下のコマンドは開発環境用のDockerイメージを構築します。

- `make init`

Dockerイメージを構築後、以下のコマンドを実行して開発環境用のDockerコンテナを生成、ログインします。

- `make create-container`

上記のコマンドを実行すると生成されたDockerコンテナにログインされます。

## 分類機を動す

前節で生成したDockerコンテナ上で以下のコマンドを実行すうるとモデルがせいせされます。

- `make train` モデルを生成
- `make inference` モデルを生成し、サンプルデータを利用して推論を実行する

### テスト実施

Dockerコンテナ上で`make test`を実行する。

### Jupyter Notebookを実行

Dockerコンテナ上で`make jupyter`を実行すると、Dockerコンテナ上にJupyter Notebookサーバが立ち上がる。
立ち上がったJupyter Notebookサーバにはホスト環境でWebブラウザを使って　http://localhost:8888 を開くことでアクセスできる。 
