import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import io
import numpy as np
import seaborn as sns
import chardet
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression


# データフレームの作成（ここではdfという名前のデータフレームとしま

# 各列の情報をまとめる表を作成

# 日本語フォントの設定

def summary(df):

    summary_table = pd.DataFrame(columns=['Column', 'Data Type', 'Count', 'Min', 'Max', 'Missing'])

    for col in df.columns:
        data_type = df[col].dtype
        count = df[col].count()
        min_val = df[col].min()
        max_val = df[col].max()
        missing = df[col].isnull().sum()

        summary_table = summary_table.append({
            'Column': col,
            'Data Type': data_type,
            'Count': count,
            'Min': min_val,
            'Max': max_val,
            'Missing': missing
        }, ignore_index=True)

    # 表を描画
    plt.figure(figsize=(20, 18))
    plt.axis('off')
    plt.table(cellText=summary_table.values,
              colLabels=summary_table.columns,
              loc='center')
    plt.title('Summary Table')
    plt.show()



@st.cache
def load_data(file):
    data = file.getvalue().decode('shift-jis')
    return pd.read_csv(io.StringIO(data))

def make_unique(column_names):
    seen = set()
    new_columns = []
    for col in column_names:
        if col in seen:
            counter = 1
            new_col = f"{col}.{counter}"
            while new_col in seen:
                counter += 1
                new_col = f"{col}.{counter}"
            seen.add(new_col)
            new_columns.append(new_col)
        else:
            seen.add(col)
            new_columns.append(col)
    return new_columns

def remove_unnecessary_rows(df):
    df = df.dropna(how='all')  # すべての要素がNaNの行を削除
    df = df[~df.apply(lambda row: row.astype(str).str.contains('タイトル|不要な行').any(), axis=1)]  # 特定のキーワードを含む行を削除
    return df

def combine_datetime(df, datetime_columns, date_format_info):
    df['datetime'] = pd.NaT
    for index, row in df.iterrows():
        datetime_str = ''
        for col, format_info in zip(datetime_columns, date_format_info):
            col_data = str(row[col])
            if format_info['start'] is not None and format_info['end'] is not None:
                col_data = col_data[format_info['start']:format_info['end']]
            datetime_str += col_data
        try:
            df.at[index, 'datetime'] = pd.to_datetime(datetime_str, format=''.join([f['format'] for f in date_format_info]))
        except Exception as e:
            st.error(f"行 {index} の日時変換に失敗しました: {e}")
    return df

def upload_and_process_file(key, file, encoding):
    # エンコーディングに基づいてファイルの内容をデコード
    data = file.getvalue().decode(encoding)
    header_row = st.selectbox(f"{file.name}のヘッダー行の位置を選択してください", options=list(range(10)), index=1, key=f"header_row_{key}")
    df = pd.read_csv(io.StringIO(data), header=header_row)
    df = remove_unnecessary_rows(df)
    st.subheader(f"アップロードされたファイル: {file.name}")
    st.write(df)

    drop_rows = st.multiselect(f"{file.name} の削除したい行を選択してください", df.index.tolist(), key=f"drop_rows_{key}")
    cleaned_df = df.drop(drop_rows) if drop_rows else df.copy()
    st.write(f"行を削除した後のファイル: {file.name}")
    st.write(cleaned_df)

    cleaned_df.columns = make_unique(cleaned_df.columns)
    st.write(f"列名を設定した後のファイル: {file.name}")
    st.write(cleaned_df)

    columns_to_keep = st.multiselect(f"{file.name} の保持したい列を選択してください", cleaned_df.columns.tolist(), default=cleaned_df.columns.tolist(), key=f"columns_to_keep_{key}")
    df_final = cleaned_df[columns_to_keep] if columns_to_keep else cleaned_df.copy()
    st.write(f"列を選択した後のファイル: {file.name}")
    st.write(df_final)

    return df_final, key
# エンコードを指定してファイルを読み込む関数
def read_csv_with_encoding(uploaded_file, encoding):
    try:
        return pd.read_csv(io.StringIO(uploaded_file.getvalue().decode(encoding)))
    except (UnicodeDecodeError, pd.errors.EmptyDataError) as e:
        st.write(f"{encoding}での読み込み中にエラーが発生しました: {e}")
        return None



# タイトル
st.title("製造現場での不良や変動の要因解析のためのデータ分析アプリ")

# サイドバーに機能選択ボックスを表示
page = st.sidebar.selectbox("ページを選択してください", ["機能1: CSVファイル統合", "機能2: データの基本情報、基本統計量", "機能3: 時系列グラフと相関関係","機能4: カテゴリ変数ごとの箱ひげ図","機能5: 移動平均と移動分散の可視化","機能6: PLS分析と説明変数の影響度可視化","機能7: コード表示"])



# ファイルをアップロードされたファイルを格納するリスト
dfs = []
cleaned_dfs = []
header_set_dfs = []
selected_columns_dfs = []




# メイン処理部分
if page == "機能1: CSVファイル統合":
    st.header("機能1: CSVファイル統合")

    # 複数ファイルのアップロード
    uploaded_files = st.file_uploader("ファイルをアップロードしてください", type=['csv'], accept_multiple_files=True)

    if uploaded_files:
        processed_files = []
        for i, uploaded_file in enumerate(uploaded_files):
            file_key = f"file_{i+1}"
            st.subheader(f"ファイル {i+1}: {uploaded_file.name}")

            # ファイルのバイナリデータを読み込む
            raw_data = uploaded_file.getvalue()

            # エンコーディングの検出
            result = chardet.detect(raw_data)
            detected_encoding = result['encoding']
            st.write(f"検出されたエンコーディング: {detected_encoding}")

            # ユーザーにエンコーディングを選択させる
            encodings_to_try = [detected_encoding, 'utf-8-sig', 'shift-jis', 'Windows-1252', 'MacRoman']
            manual_encoding = st.selectbox("手動でエンコーディングを選択してください", encodings_to_try, key=file_key)

            # ファイル処理
            df_final, key = upload_and_process_file(file_key, uploaded_file, manual_encoding)

            if df_final is not None:
                st.write(f"ファイル {i+1} の処理結果:")
                st.write(df_final)
                processed_files.append((df_final, file_key))

        if len(processed_files) > 1:
            st.subheader("統合設定")
            merge_column = st.selectbox("結合に使用する列を選択してください", processed_files[0][0].columns.tolist())

            if st.button("ファイルを結合"):
                st.write("ファイルを結合しています...")
                merged_df = processed_files[0][0]  # 最初のデータフレームを取得
                for i in range(1, len(processed_files)):
                    merged_df = pd.merge(merged_df, processed_files[i][0], on=merge_column)

                merged_df = merged_df.sort_values(by=merge_column).reset_index(drop=True)
                st.subheader("結合後のデータ")
                st.write(merged_df)

                # CSVとして保存
                csv = merged_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(label="結合データをCSVとしてダウンロード", data=csv, file_name='merged_data.csv', mime='text/csv')
        elif len(processed_files) == 1:
            st.write("1つのファイルのみがアップロードされました。処理を続行してください。")










    elif page == "機能2":
          st.header("機能2: データの可視化")
          # 機能2のコードをここに記述
          st.write("ここにデータの可視化機能のコードを記述します。")

    elif page == "機能3":
          st.header("機能3: モデルのトレーニング")
          # 機能3のコードをここに記述
          st.write("ここにモデルのトレーニング機能のコードを記述します。")
    elif page == "コード表示":
          st.header("コード表示")

          # 現在のスクリプトファイルを読み込んで表示
          with open(__file__, 'r', encoding='utf-8') as f:
              code = f.read()

          st.code(code, language='python')





# 機能2: データの可視化




if page == "機能2: データの基本情報、基本統計量":
    st.header("機能2: データの基本情報、基本統計量")

    # ファイルアップローダー
    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=['csv'])

    if uploaded_file:
        # ファイルのバイナリデータを読み込む
        raw_data = uploaded_file.getvalue()

        # エンコーディングの検出
        result = chardet.detect(raw_data)
        detected_encoding = result['encoding']
        st.write(f"検出されたエンコーディング: {detected_encoding}")

        # ユーザーにエンコーディングを選択させる
        encodings_to_try = [detected_encoding, 'utf-8-sig', 'shift-jis', 'Windows-1252', 'MacRoman']
        manual_encoding = st.selectbox("手動でエンコーディングを選択してください", encodings_to_try)

        # 選択されたエンコーディングでデータを読み込む
        df = read_csv_with_encoding(uploaded_file, manual_encoding)
        if df is not None:
            st.write(f"選択された{manual_encoding}エンコーディングで読み込み成功")
        else:
            st.error(f"{manual_encoding}エンコーディングでも読み込みに失敗しました。")

        if df is not None:
            # データの基本情報の表示
            st.subheader("データの基本情報")
            st.write("データの型:")
            st.write(df.dtypes)
            st.write("欠損値の数:")
            st.write(df.isnull().sum())
            st.write("データの基本統計量:")
            st.write(df.describe())

            # 各列のヒストグラムの表示
            st.subheader("各列のヒストグラム")
            for col in df.select_dtypes(include=[np.number]).columns:
                fig, ax = plt.subplots()
                sns.histplot(df[col], kde=True, ax=ax)
                ax.set_title(f"{col} のヒストグラム")
                st.pyplot(fig)

            # 欠損値の処理
            st.subheader("欠損値の処理")
            missing_option = st.selectbox("欠損値をどのように処理しますか？", ["平均値で埋める", "欠損値がある行を削除する"])

            if missing_option == "平均値で埋める":
                df = df.fillna(df.mean())
            elif missing_option == "欠損値がある行を削除する":
                df = df.dropna()

            st.write("欠損値処理後のデータ")
            st.write(df)

            # 加工されたファイルをCSVとして保存する機能
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(label="加工されたデータをCSVとしてダウンロード", data=csv, file_name='processed_data.csv', mime='text/csv')


if page == "機能3: 時系列グラフと相関関係":
    st.header("機能3: 時系列グラフと相関関係")

    # ファイルアップローダー
    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=['csv'])

    if uploaded_file:
        # UTF-8-SIGエンコーディングでCSVを読み込む
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig')

        # 各列に欠損値があるかどうかを確認
        st.subheader("各列の欠損値の数")
        st.write(df.isnull().sum())

        # 時間を表す列をインデックスにセットする機能
        time_column = st.selectbox('時間を表す列を選択してください', df.columns)
        
        # エラーハンドリング：選択された列がdatetime型に変換できない場合
        try:
            df[time_column] = pd.to_datetime(df[time_column], errors='raise')
            df.set_index(time_column, inplace=True)
        except Exception:
            st.error("選択された列は時間を表す列ではありません。もう一度「時間を表す列を選択してください」。")
            st.stop()  # 以降の処理を停止

        # 列の中から目的変数を選ぶ
        target_var = st.selectbox('目的変数を選択してください', df.columns)

        # エラーハンドリング：目的変数として時間列が選ばれた場合
        if target_var == time_column:
            st.error("時間を表す列は目的変数として選択できません。適切な列を選択してください。")
            st.stop()  # 以降の処理を停止

        # 単位を選択
        time_unit = st.selectbox('遅れ時間の単位を選択してください', ['秒', '分', '時間', '日', '月', '年'])

        # スライドバーで時間をずらす量を設定
        time_lag = st.slider(f'目的変数の時間をずらす量（単位: {time_unit}）', min_value=-100, max_value=100, value=0, step=1)

        # 遅れ時間を選択された単位に基づいてシフト
        if time_unit == '秒':
            shifted_target = df[target_var].shift(periods=time_lag, freq='S')
        elif time_unit == '分':
            shifted_target = df[target_var].shift(periods=time_lag, freq='T')
        elif time_unit == '時間':
            shifted_target = df[target_var].shift(periods=time_lag, freq='H')
        elif time_unit == '日':
            shifted_target = df[target_var].shift(periods=time_lag, freq='D')
        elif time_unit == '月':
            shifted_target = df[target_var].shift(periods=time_lag, freq='M')
        elif time_unit == '年':
            shifted_target = df[target_var].shift(periods=time_lag, freq='Y')

        # 各説明変数の決定係数を計算して棒グラフで表示
        st.subheader(f"遅れ時間 {time_lag} {time_unit}における各説明変数の決定係数")
        r_squared_values = {}
        for col in df.columns:
            if col != target_var:
                correlation = df[col].corr(shifted_target)
                r_squared_values[col] = correlation ** 2

        sorted_r_squared = sorted(r_squared_values.items(), key=lambda item: item[1], reverse=True)
        variables, r_squared_scores = zip(*sorted_r_squared)

        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.bar(variables[:10], r_squared_scores[:10], color='skyblue')  # 上位10の説明変数を表示
        ax1.set_xlabel('説明変数')
        ax1.set_ylabel('決定係数 (R²)')
        ax1.set_title(f'遅れ時間 {time_lag} {time_unit}における決定係数の上位10変数')
        ax1.tick_params(axis='x', rotation=45)
        st.pyplot(fig1)

        # 列の中から表示させる説明変数を選ぶ
        explanatory_var = st.selectbox('表示させる説明変数を選択してください', variables)

        # 時系列グラフと決定係数のグラフ
        st.subheader(f"{explanatory_var} の時系列グラフと決定係数（遅れ時間: {time_lag} {time_unit}）")

        fig2, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))

        # 目的変数の時系列グラフ
        axes[0].scatter(df.index, df[target_var], label=target_var, color='orange')
        axes[0].set_title(f'{target_var} の時系列データ')
        axes[0].set_ylabel(target_var)
        axes[0].legend()

        # 説明変数の時系列グラフ
        axes[1].scatter(df.index, df[explanatory_var], label=explanatory_var, color='blue')
        axes[1].set_title(f'{explanatory_var} の時系列データ')
        axes[1].set_ylabel(explanatory_var)
        axes[1].legend()

        # 決定係数のグラフ
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        valid_data = pd.concat([df[explanatory_var], shifted_target], axis=1).dropna()
        correlation = valid_data[target_var].corr(valid_data[explanatory_var])
        r_squared = correlation ** 2
        ax3.scatter(valid_data[explanatory_var], valid_data[target_var], alpha=0.5)
        ax3.set_title(f'{explanatory_var} と {target_var} の決定係数 (R²={r_squared:.2f})')
        ax3.set_xlabel(explanatory_var)
        ax3.set_ylabel(target_var)

        st.pyplot(fig2)
        st.pyplot(fig3)








if page == "機能4: カテゴリ変数ごとの箱ひげ図":
    st.header("機能4: カテゴリ変数ごとの箱ひげ図")

    # ファイルアップローダー
    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=['csv'])

    if uploaded_file:
        # UTF-8-SIGエンコーディングでCSVを読み込む
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig')

        # 各列に欠損値があるかどうかを確認
        st.subheader("各列の欠損値の数")
        st.write(df.isnull().sum())

        # カテゴリ変数と目的変数を選ぶ
        category_var = st.selectbox('カテゴリ変数を選択してください', df.columns)
        target_var = st.selectbox('目的変数を選択してください', df.columns)

        if category_var and target_var:
            st.subheader(f"{category_var} ごとの {target_var} の箱ひげ図")

            # 箱ひげ図の描画
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x=df[category_var], y=df[target_var], ax=ax)
            ax.set_title(f'{category_var} ごとの {target_var} の箱ひげ図')
            ax.set_xlabel(category_var)
            ax.set_ylabel(target_var)
            st.pyplot(fig)

            # カテゴリごとの統計量を計算
            stats_df = df.groupby(category_var)[target_var].agg([
                ('平均値', 'mean'),
                ('中央値', 'median'),
                ('標準偏差', 'std'),
                ('最小値', 'min'),
                ('最大値', 'max'),
                ('サンプルサイズ', 'count'),
                ('四分位範囲 (IQR)', lambda x: x.quantile(0.75) - x.quantile(0.25))
            ]).reset_index()

            st.subheader(f"{category_var} ごとの {target_var} の統計量")
            st.write(stats_df)



# 機能5: 移動平均と移動分散の可視化
if page == "機能5: 移動平均と移動分散の可視化":
    st.header("機能5: 移動平均と移動分散の可視化")

    # ファイルアップローダー
    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=['csv'])

    if uploaded_file:
        # UTF-8-SIGエンコーディングでCSVを読み込む
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig')

        # 時間を表す列をインデックスにセットする機能
        time_column = st.selectbox('時間を表す列を選択してください', df.columns)
        df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
        df.set_index(time_column, inplace=True)

        # データの時刻の最小値と最大値を表示
        min_time = df.index.min()
        max_time = df.index.max()
        st.write(f"データの時刻の範囲: {min_time} から {max_time}")

        # Timestamp型をdatetime型に変換
        min_time = min_time.to_pydatetime()
        max_time = max_time.to_pydatetime()

        # 分析する期間を指定
        start_time = st.slider("開始時刻を選択してください", min_value=min_time, max_value=max_time, value=min_time, format="YYYY-MM-DD HH:mm")
        end_time = st.slider("終了時刻を選択してください", min_value=min_time, max_value=max_time, value=max_time, format="YYYY-MM-DD HH:mm")

        # 指定した期間のデータをフィルタリング
        df = df.loc[start_time:end_time]

        # 指定した期間のデータの確認
        st.write(f"指定した期間のデータ数: {len(df)}")
        st.write(df.head())

        # 移動平均と移動分散を計算する列を選択
        target_var = st.selectbox('移動平均と移動分散を計算する列を選択してください', df.columns)

        # ウィンドウサイズを指定
        window_size = st.number_input('ウィンドウサイズを指定してください', min_value=1, value=5, step=1)

        # 移動平均、移動分散、変動係数を計算
        df['Moving Average'] = df[target_var].rolling(window=window_size).mean()
        df['Moving Variance'] = df[target_var].rolling(window=window_size).var()
        df['Coefficient of Variation'] = df['Moving Variance']**0.5 / df['Moving Average']

        # 移動平均の時系列グラフ
        st.subheader(f"{target_var} の移動平均")
        fig1, ax1 = plt.subplots(figsize=(15, 5))

        ax1.plot(df.index, df[target_var], label=target_var, color='blue', alpha=0.5)
        ax1.plot(df.index, df['Moving Average'], label='Moving Average', color='green')
        ax1.set_xlabel('Date')
        ax1.set_ylabel(target_var, color='blue')
        ax1.legend(loc='upper left')

        st.pyplot(fig1)

        # 移動分散の時系列グラフ
        st.subheader(f"{target_var} の移動分散")
        fig2, ax2 = plt.subplots(figsize=(15, 5))

        ax2.plot(df.index, df[target_var], label=target_var, color='blue', alpha=0.5)
        ax2.set_xlabel('Date')
        ax2.set_ylabel(target_var, color='blue')
        ax2.legend(loc='upper left')

        ax3 = ax2.twinx()
        ax3.plot(df.index, df['Moving Variance'], label='Moving Variance', color='red')
        ax3.set_ylabel('Moving Variance', color='red')
        ax3.legend(loc='upper right')

        st.pyplot(fig2)

        # 変動係数の時系列グラフ
        st.subheader(f"{target_var} の変動係数")
        fig3, ax4 = plt.subplots(figsize=(15, 5))

        ax4.plot(df.index, df['Coefficient of Variation'], label='Coefficient of Variation', color='purple')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Coefficient of Variation', color='purple')
        ax4.legend(loc='upper right')

        st.pyplot(fig3)

        # 移動平均の最大値、最小値、中央値を計算
        max_avg = df['Moving Average'].max()
        min_avg = df['Moving Average'].min()
        median_avg = df['Moving Average'].median()

        # 結果を表で表示
        st.subheader("移動平均の統計量")
        summary_table = pd.DataFrame({
            'Statistic': ['最大値', '最小値', '中央値'],
            'Value': [max_avg, min_avg, median_avg]
        })
        st.write(summary_table)





# 機能6: PLS分析と説明変数の影響度可視化
if page == "機能6: PLS分析と説明変数の影響度可視化":
    st.header("機能6: PLS分析と説明変数の影響度可視化")

    # ファイルアップローダー
    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=['csv'])

    if uploaded_file:
        # UTF-8-SIGエンコーディングでCSVを読み込む
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig')

        # 時間を表す列をインデックスにセットする機能
        time_column = st.selectbox('時間を表す列を選択してください', df.columns)
        df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
        df.set_index(time_column, inplace=True)

        # データの確認
        st.write(f"データの時刻の範囲: {df.index.min()} から {df.index.max()}")
        st.write(df.head())

        # 目的変数を選択
        target_var = st.selectbox('目的変数を選択してください', df.columns)

        # カテゴリ変数を手動で選択して除外
        category_vars = st.multiselect('カテゴリ変数を選択してください', df.columns.drop(target_var))

        # 説明変数（目的変数、カテゴリ変数、インデックスを除外）
        explanatory_vars = df.drop(columns=[target_var] + category_vars)

        # 説明変数が選択されているか確認
        if explanatory_vars.empty:
            st.error("有効な説明変数がありません。")
        else:
            # コンポーネント数を選択
            n_components = st.selectbox('PLSのコンポーネント数を選択してください', options=[1, 2, 3], index=1)

            # 標準化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(explanatory_vars)
            y_scaled = scaler.fit_transform(df[[target_var]])

            # PLS分析の実行
            from sklearn.cross_decomposition import PLSRegression
            pls = PLSRegression(n_components=n_components)
            pls.fit(X_scaled, y_scaled)

            # 各説明変数の影響度（回帰係数の絶対値の合計）
            influence = np.sum(np.abs(pls.coef_), axis=1)
            influence_df = pd.DataFrame({'Variable': explanatory_vars.columns, 'Influence': influence})
            top5_influence = influence_df.nlargest(5, 'Influence')

            # 決定係数を計算
            r_squared = pls.score(X_scaled, y_scaled)

            # 影響度の高い説明変数の上位5位の棒グラフ
            st.subheader("影響度の高い説明変数 上位5位")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(top5_influence['Variable'], top5_influence['Influence'], color='skyblue')
            ax.set_xlabel('説明変数')
            ax.set_ylabel('影響度')
            ax.set_title(f'PLS分析による影響度の高い説明変数 上位5位 (n_components={n_components})')

            # グラフ内に決定係数 (R²) を表示
            ax.text(0.95, 0.95, f'R² = {r_squared:.2f}', transform=ax.transAxes, 
                    fontsize=12, verticalalignment='top', horizontalalignment='right', 
                    bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
            
            st.pyplot(fig)

if page == "機能7: コード表示":
    st.header("機能7: コード表示")
    # 現在のスクリプトファイルを読み込んで表示
    with open(__file__, 'r', encoding='utf-8') as f:
        code = f.read()

    st.code(code, language='python')






