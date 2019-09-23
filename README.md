[![Build Status](https://travis-ci.org/takahi-i/lecture-tmu-2019.svg?branch=master)](https://travis-ci.org/takahi-i/lecture-tmu-2019)

# lecture-tmu-2019

首都大学東京で実施した講義の資料です。

## 準備

- [Docker v17 以降](https://docs.docker.com/install/#support)

## 開発環境のセットアップ

以下のコマンドを実行して、開発環境を構築します。以下のコマンドは開発環境用のDockerイメージを構築します。

- `make init`

Dockerイメージを構築後、以下のコマンドを実行して開発環境用のDockerコンテナを生成、ログインします。

- `make create-container`

上記のコマンドを実行すると生成されたDockerコンテナにログインされます。

## 分類機を動す

前節で生成したDockerコンテナ上で以下のコマンドを実行するとモデルが生成されます。

- `make train` モデルを生成
- `make inference` モデルを生成し、サンプルデータを利用して推論を実行する

## テストを実行

Dockerコンテナ上で`make test`を実行する。

## Jupyter Notebookを実行

Dockerコンテナ上で`make jupyter`を実行すると、Dockerコンテナ上にJupyter Notebookサーバが立ち上がる。
立ち上がったJupyter Notebookサーバにはホスト環境でWebブラウザを使って http://localhost:8888 を開くことでアクセスできる。

## Credits

本レポジトリはの Jupyter Notebookは [フリーライブラリで学ぶ機械学習入門 サンプルコード](https://github.com/yosukekatada/mlbook) のものです。
著者の一人である菊田氏に使用を相談したところ、快諾をいただきました。
