import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime, timedelta
import os
import json

# ページ設定
st.set_page_config(
    page_title="観光農園予約システム",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# サイドバーナビゲーション
st.sidebar.title("観光農園予約システム")
page = st.sidebar.radio(
    "ページ選択",
    ["ホーム", "農園一覧", "予約管理", "顧客管理", "来客予測", "システム情報"]
)

# 来客予測モデルのロード（存在する場合）
@st.cache_resource
def load_prediction_model():
    model_path = "visitor_prediction_model.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        # モデルがない場合はダミーモデルを作成
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor()
        return {"model": model, "preprocessor": None, "feature_names": None}

# モックデータの生成
@st.cache_data
def generate_mock_data():
    # 農園データ
    farms = pd.DataFrame({
        "id": range(1, 6),
        "name": ["いちご農園", "りんご農園", "ぶどう農園", "みかん農園", "さくらんぼ農園"],
        "location": ["東京都", "青森県", "山梨県", "愛媛県", "山形県"],
        "description": [
            "東京近郊で楽しめるいちご狩り農園です。30分食べ放題のコースが人気です。",
            "青森県産の美味しいりんごが収穫できる農園です。秋には様々な品種のりんご狩りが楽しめます。",
            "山梨県の自然豊かな環境で育ったぶどうの収穫体験ができます。ワイン用品種も栽培しています。",
            "愛媛県特産のみかん狩りが楽しめる農園です。冬季には温州みかんの収穫体験ができます。",
            "初夏に旬を迎えるさくらんぼの収穫体験ができます。山形県の特産品を直接味わえます。"
        ],
        "main_crop": ["いちご", "りんご", "ぶどう", "みかん", "さくらんぼ"],
        "harvest_season_start": ["1月", "9月", "8月", "11月", "6月"],
        "harvest_season_end": ["5月", "11月", "10月", "1月", "7月"],
        "rating": [4.5, 4.2, 4.7, 4.0, 4.8]
    })
    
    # 予約データ
    np.random.seed(42)
    reservations = []
    for i in range(100):
        farm_id = np.random.randint(1, 6)
        date = datetime.now() + timedelta(days=np.random.randint(-30, 30))
        reservations.append({
            "id": i + 1,
            "farm_id": farm_id,
            "customer_id": np.random.randint(1, 51),
            "date": date.strftime("%Y-%m-%d"),
            "time_slot": f"{np.random.randint(9, 16)}:00",
            "adults": np.random.randint(1, 5),
            "children": np.random.randint(0, 4),
            "seniors": np.random.randint(0, 3),
            "status": np.random.choice(["確定", "キャンセル", "利用済み"], p=[0.7, 0.1, 0.2]),
            "created_at": (date - timedelta(days=np.random.randint(1, 14))).strftime("%Y-%m-%d")
        })
    reservations_df = pd.DataFrame(reservations)
    
    # 顧客データ
    customers = []
    for i in range(50):
        age_group = np.random.choice(["20代", "30代", "40代", "50代", "60代以上"], p=[0.1, 0.3, 0.3, 0.2, 0.1])
        customers.append({
            "id": i + 1,
            "name": f"顧客{i+1}",
            "email": f"customer{i+1}@example.com",
            "phone": f"090-{np.random.randint(1000, 9999)}-{np.random.randint(1000, 9999)}",
            "age_group": age_group,
            "prefecture": np.random.choice(["東京都", "神奈川県", "埼玉県", "千葉県", "その他"], p=[0.3, 0.2, 0.2, 0.2, 0.1]),
            "first_visit": (datetime.now() - timedelta(days=np.random.randint(0, 365))).strftime("%Y-%m-%d"),
            "visit_count": np.random.randint(1, 10),
            "preferences": np.random.choice(["いちご", "りんご", "ぶどう", "みかん", "さくらんぼ"], size=np.random.randint(1, 3)).tolist()
        })
    customers_df = pd.DataFrame(customers)
    
    # 来客データ
    visitor_data = []
    start_date = datetime.now() - timedelta(days=365)
    for i in range(365):
        date = start_date + timedelta(days=i)
        day_of_week = date.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        is_holiday = np.random.choice([0, 1], p=[0.9, 0.1])
        
        # 基本来客数
        base = 30
        
        # 曜日の影響（週末は多い）
        if is_weekend:
            base += 40
        
        # 祝日の影響
        if is_holiday:
            base += 30
        
        # 季節の影響
        month = date.month
        season_factor = np.sin(month / 12 * 2 * np.pi) * 20 + 20
        base += season_factor
        
        # ランダム変動
        noise = np.random.normal(0, 10)
        
        # 最終来客数（負にならないように）
        visitors = max(0, int(base + noise))
        
        visitor_data.append({
            "date": date.strftime("%Y-%m-%d"),
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "month": month,
            "visitors": visitors
        })
    visitor_df = pd.DataFrame(visitor_data)
    
    return farms, reservations_df, customers_df, visitor_df

# データの読み込み
farms, reservations, customers, visitor_data = generate_mock_data()

# ホームページ
def home_page():
    st.title("観光農園予約システム")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## ようこそ！観光農園予約システムへ
        
        このシステムでは、全国の観光農園の予約管理と顧客情報管理を一元化し、
        来客予測を行うための総合的なプラットフォームを提供しています。
        
        ### 主な機能
        
        - **予約管理**: オンライン予約受付、カレンダー形式での予約状況表示
        - **顧客管理**: 顧客情報の一元管理、訪問履歴の記録と分析
        - **農園情報**: 作物情報、収穫時期、イベント情報の管理
        - **来客予測**: 過去のデータに基づく将来の来客数予測
        
        サイドバーから各機能にアクセスできます。
        """)
    
    with col2:
        st.image("https://images.unsplash.com/photo-1523741543316-beb7fc7023d8?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1374&q=80", caption="観光農園の風景")
    
    st.markdown("---")
    
    # 最新情報
    st.subheader("最新情報")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 人気の農園")
        popular_farms = farms.sort_values("rating", ascending=False).head(3)
        for i, farm in popular_farms.iterrows():
            st.markdown(f"**{farm['name']}** - {farm['location']} (評価: {farm['rating']})")
            st.markdown(f"*{farm['description'][:100]}...*")
    
    with col2:
        st.markdown("### 今月の収穫カレンダー")
        current_month = datetime.now().strftime("%m月")
        st.markdown(f"**{current_month}の収穫可能な作物**")
        
        for i, farm in farms.iterrows():
            start_month = farm["harvest_season_start"]
            end_month = farm["harvest_season_end"]
            if _is_current_month_in_season(start_month, end_month):
                st.markdown(f"- {farm['name']}: **{farm['main_crop']}**")

# 農園一覧ページ
def farm_list_page():
    st.title("農園一覧")
    
    # 検索・フィルタリング
    col1, col2, col3 = st.columns(3)
    with col1:
        search_term = st.text_input("キーワード検索", "")
    with col2:
        location_filter = st.selectbox("地域で絞り込み", ["すべて"] + list(farms["location"].unique()))
    with col3:
        crop_filter = st.selectbox("作物で絞り込み", ["すべて"] + list(farms["main_crop"].unique()))
    
    # フィルタリング適用
    filtered_farms = farms.copy()
    if search_term:
        filtered_farms = filtered_farms[
            filtered_farms["name"].str.contains(search_term) | 
            filtered_farms["description"].str.contains(search_term)
        ]
    if location_filter != "すべて":
        filtered_farms = filtered_farms[filtered_farms["location"] == location_filter]
    if crop_filter != "すべて":
        filtered_farms = filtered_farms[filtered_farms["main_crop"] == crop_filter]
    
    # 農園一覧表示
    if len(filtered_farms) == 0:
        st.warning("条件に一致する農園がありません。検索条件を変更してください。")
    else:
        st.write(f"{len(filtered_farms)}件の農園が見つかりました")
        
        for i, farm in filtered_farms.iterrows():
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # 作物に応じた画像を表示
                crop_images = {
                    "いちご": "https://images.unsplash.com/photo-1518635017498-87f514b751ba?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1471&q=80",
                    "りんご": "https://images.unsplash.com/photo-1570913149827-d2ac84ab3f9a?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1470&q=80",
                    "ぶどう": "https://images.unsplash.com/photo-1596363505729-4190a9506133?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1471&q=80",
                    "みかん": "https://images.unsplash.com/photo-1611080626919-7cf5a9dbab12?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1470&q=80",
                    "さくらんぼ": "https://images.unsplash.com/photo-1528821128474-27f963b062bf?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1470&q=80"
                }
                image_url = crop_images.get(farm["main_crop"], "https://images.unsplash.com/photo-1523741543316-beb7fc7023d8?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1374&q=80")
                st.image(image_url, width=200)
            
            with col2:
                st.subheader(farm["name"])
                st.write(f"**場所**: {farm['location']}")
                st.write(f"**主な作物**: {farm['main_crop']}")
                st.write(f"**収穫時期**: {farm['harvest_season_start']}〜{farm['harvest_season_end']}")
                st.write(f"**評価**: {'⭐' * int(farm['rating'])}")
                st.write(farm["description"])
                
                # 予約ボタン
                if st.button(f"{farm['name']}を予約する", key=f"reserve_{farm['id']}"):
                    st.session_state["selected_farm"] = farm["id"]
                    st.session_state["page"] = "予約管理"
                    st.experimental_rerun()
            
            st.markdown("---")

# 予約管理ページ
def reservation_page():
    st.title("予約管理")
    
    tabs = st.tabs(["予約一覧", "新規予約", "予約分析"])
    
    # 予約一覧タブ
    with tabs[0]:
        st.subheader("予約一覧")
        
        # フィルタリング
        col1, col2, col3 = st.columns(3)
        with col1:
            status_filter = st.selectbox("予約状況", ["すべて", "確定", "キャンセル", "利用済み"])
        with col2:
            farm_filter = st.selectbox("農園", ["すべて"] + list(farms["name"]))
        with col3:
            date_range = st.date_input("期間", [datetime.now() - timedelta(days=30), datetime.now() + timedelta(days=30)])
        
        # フィルタリング適用
        filtered_reservations = reservations.copy()
        if status_filter != "すべて":
            filtered_reservations = filtered_reservations[filtered_reservations["status"] == status_filter]
        if farm_filter != "すべて":
            farm_id = farms[farms["name"] == farm_filter]["id"].values[0]
            filtered_reservations = filtered_reservations[filtered_reservations["farm_id"] == farm_id]
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_reservations = filtered_reservations[
                (pd.to_datetime(filtered_reservations["date"]) >= pd.to_datetime(start_date)) &
                (pd.to_datetime(filtered_reservations["date"]) <= pd.to_datetime(end_date))
            ]
        
        # 予約データと農園名、顧客名を結合
        merged_reservations = filtered_reservations.merge(
            farms[["id", "name"]], 
            left_on="farm_id", 
            right_on="id", 
            suffixes=("", "_farm")
        ).merge(
            customers[["id", "name"]], 
            left_on="customer_id", 
            right_on="id", 
            suffixes=("", "_customer")
        )
        
        # 表示用データフレーム
        display_df = merged_reservations[[
            "id", "name_farm", "name_customer", "date", "time_slot", 
            "adults", "children", "seniors", "status"
        ]].rename(columns={
            "id": "予約ID",
            "name_farm": "農園名",
            "name_customer": "顧客名",
            "date": "日付",
            "time_slot": "時間帯",
            "adults": "大人",
            "children": "子供",
            "seniors": "シニア",
            "status": "状態"
        })
        
        st.dataframe(display_df, use_container_width=True)
    
    # 新規予約タブ
    with tabs[1]:
        st.subheader("新規予約")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 農園選択
            selected_farm_id = st.selectbox(
                "農園を選択", 
                options=farms["id"].tolist(),
                format_func=lambda x: farms[farms["id"] == x]["name"].values[0],
                index=0 if "selected_farm" not in st.session_state else farms["id"].tolist().index(st.session_state["selected_farm"])
            )
            
            # 選択された農園の情報表示
            selected_farm = farms[farms["id"] == selected_farm_id].iloc[0]
            st.write(f"**収穫作物**: {selected_farm['main_crop']}")
            st.write(f"**収穫時期**: {selected_farm['harvest_season_start']}〜{selected_farm['harvest_season_end']}")
            
            # 顧客選択
            selected_customer_id = st.selectbox(
                "顧客を選択", 
                options=customers["id"].tolist(),
                format_func=lambda x: customers[customers["id"] == x]["name"].values[0]
            )
        
        with col2:
            # 日時選択
            selected_date = st.date_input("日付を選択", datetime.now() + timedelta(days=1))
            selected_time = st.selectbox("時間帯を選択", [f"{h}:00" for h in range(9, 17)])
            
            # 人数選択
            adults = st.number_input("大人", min_value=1, max_value=10, value=2)
            children = st.number_input("子供", min_value=0, max_value=10, value=0)
            seniors = st.number_input("シニア", min_value=0, max_value=10, value=0)
        
        # 備考
        notes = st.text_area("備考", "")
        
        # 予約ボタン
        if st.button("予約を確定する"):
            st.success("予約が完了しました！")
            st.balloons()
    
    # 予約分析タブ
    with tabs[2]:
        st.subheader("予約分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 農園別予約数
            st.markdown("### 農園別予約数")
            farm_counts = reservations["farm_id"].value_counts().reset_index()
            farm_counts.columns = ["farm_id", "count"]
            farm_counts = farm_counts.merge(farms[["id", "name"]], left_on="farm_id", right_on="id")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x="name", y="count", data=farm_counts, ax=ax)
            ax.set_xlabel("農園名")
            ax.set_ylabel("予約数")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            st.pyplot(fig)
        
        with col2:
            # 月別予約数
            st.markdown("### 月別予約数")
            reservations["month"] = pd.to_datetime(reservations["date"]).dt.month
            month_counts = reservations["month"].value_counts().sort_index().reset_index()
            month_counts.columns = ["month", "count"]
            month_counts["month_name"] = month_counts["month"].apply(lambda x: f"{x}月")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x="month_name", y="count", data=month_counts, ax=ax)
            ax.set_xlabel("月")
            ax.set_ylabel("予約数")
            st.pyplot(fig)
        
        # 予約状況の円グラフ
        st.markdown("### 予約状況")
        status_counts = reservations["status"].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

# 顧客管理ページ
def customer_page():
    st.title("顧客管理")
    
    tabs = st.tabs(["顧客一覧", "顧客分析", "セグメント分析"])
    
    # 顧客一覧タブ
    with tabs[0]:
        st.subheader("顧客一覧")
        
        # 検索・フィルタリング
        col1, col2 = st.columns(2)
        with col1:
            search_term = st.text_input("顧客検索", "")
        with col2:
            age_filter = st.selectbox("年齢層", ["すべて", "20代", "30代", "40代", "50代", "60代以上"])
        
        # フィルタリング適用
        filtered_customers = customers.copy()
        if search_term:
            filtered_customers = filtered_customers[
                filtered_customers["name"].str.contains(search_term) | 
                filtered_customers["email"].str.contains(search_term)
            ]
        if age_filter != "すべて":
            filtered_customers = filtered_customers[filtered_customers["age_group"] == age_filter]
        
        # 顧客データ表示
        st.dataframe(
            filtered_customers[[
                "id", "name", "email", "phone", "age_group", 
                "prefecture", "first_visit", "visit_count"
            ]].rename(columns={
                "id": "顧客ID",
                "name": "氏名",
                "email": "メールアドレス",
                "phone": "電話番号",
                "age_group": "年齢層",
                "prefecture": "都道府県",
                "first_visit": "初回訪問日",
                "visit_count": "訪問回数"
            }),
            use_container_width=True
        )
        
        # 顧客詳細表示（クリックで展開）
        customer_id = st.number_input("顧客IDを入力して詳細を表示", min_value=1, max_value=len(customers), step=1)
        if st.button("詳細を表示"):
            selected_customer = customers[customers["id"] == customer_id].iloc[0]
            
            st.markdown("### 顧客詳細情報")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**氏名**: {selected_customer['name']}")
                st.write(f"**メールアドレス**: {selected_customer['email']}")
                st.write(f"**電話番号**: {selected_customer['phone']}")
                st.write(f"**年齢層**: {selected_customer['age_group']}")
            
            with col2:
                st.write(f"**都道府県**: {selected_customer['prefecture']}")
                st.write(f"**初回訪問日**: {selected_customer['first_visit']}")
                st.write(f"**訪問回数**: {selected_customer['visit_count']}")
                st.write(f"**好みの作物**: {', '.join(selected_customer['preferences'])}")
            
            # 予約履歴
            customer_reservations = reservations[reservations["customer_id"] == customer_id].merge(
                farms[["id", "name"]], 
                left_on="farm_id", 
                right_on="id", 
                suffixes=("", "_farm")
            )
            
            st.markdown("### 予約履歴")
            if len(customer_reservations) > 0:
                st.dataframe(
                    customer_reservations[[
                        "date", "name_farm", "time_slot", "adults", "children", "seniors", "status"
                    ]].rename(columns={
                        "date": "日付",
                        "name_farm": "農園名",
                        "time_slot": "時間帯",
                        "adults": "大人",
                        "children": "子供",
                        "seniors": "シニア",
                        "status": "状態"
                    }),
                    use_container_width=True
                )
            else:
                st.info("予約履歴がありません")
    
    # 顧客分析タブ
    with tabs[1]:
        st.subheader("顧客分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 年齢層分布
            st.markdown("### 年齢層分布")
            age_counts = customers["age_group"].value_counts().reset_index()
            age_counts.columns = ["age_group", "count"]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x="age_group", y="count", data=age_counts, ax=ax)
            ax.set_xlabel("年齢層")
            ax.set_ylabel("顧客数")
            st.pyplot(fig)
        
        with col2:
            # 地域分布
            st.markdown("### 地域分布")
            prefecture_counts = customers["prefecture"].value_counts().reset_index()
            prefecture_counts.columns = ["prefecture", "count"]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x="prefecture", y="count", data=prefecture_counts, ax=ax)
            ax.set_xlabel("都道府県")
            ax.set_ylabel("顧客数")
            st.pyplot(fig)
        
        # 訪問回数分布
        st.markdown("### 訪問回数分布")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(customers["visit_count"], bins=10, ax=ax)
        ax.set_xlabel("訪問回数")
        ax.set_ylabel("顧客数")
        st.pyplot(fig)
        
        # 作物の好み分布
        st.markdown("### 作物の好み分布")
        
        # 好みの作物をカウント
        crop_preferences = []
        for prefs in customers["preferences"]:
            crop_preferences.extend(prefs)
        
        crop_counts = pd.Series(crop_preferences).value_counts().reset_index()
        crop_counts.columns = ["crop", "count"]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="crop", y="count", data=crop_counts, ax=ax)
        ax.set_xlabel("作物")
        ax.set_ylabel("好む顧客数")
        st.pyplot(fig)
    
    # セグメント分析タブ
    with tabs[2]:
        st.subheader("顧客セグメント分析")
        
        # RFM分析（簡易版）
        st.markdown("### RFM分析")
        st.markdown("""
        RFM分析は以下の3つの指標に基づいて顧客をセグメント化する手法です：
        - **Recency（最新性）**: 最後の訪問からの経過時間
        - **Frequency（頻度）**: 訪問回数
        - **Monetary（金額）**: 支出金額（このデモでは訪問回数で代用）
        """)
        
        # 訪問回数でセグメント化
        customers["segment"] = pd.cut(
            customers["visit_count"], 
            bins=[0, 2, 5, 10], 
            labels=["低頻度顧客", "中頻度顧客", "高頻度顧客"]
        )
        
        segment_counts = customers["segment"].value_counts().reset_index()
        segment_counts.columns = ["segment", "count"]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="segment", y="count", data=segment_counts, ax=ax)
        ax.set_xlabel("顧客セグメント")
        ax.set_ylabel("顧客数")
        st.pyplot(fig)
        
        # セグメント別の特性
        st.markdown("### セグメント別特性")
        
        segment_age = customers.groupby("segment")["age_group"].value_counts().unstack().fillna(0)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        segment_age.plot(kind="bar", stacked=True, ax=ax)
        ax.set_xlabel("顧客セグメント")
        ax.set_ylabel("顧客数")
        ax.legend(title="年齢層")
        st.pyplot(fig)

# 来客予測ページ
def prediction_page():
    st.title("来客予測")
    
    tabs = st.tabs(["過去の来客データ", "来客予測", "予測モデル分析"])
    
    # 過去の来客データタブ
    with tabs[0]:
        st.subheader("過去の来客データ")
        
        # 日別来客数の時系列グラフ
        st.markdown("### 日別来客数")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(x=pd.to_datetime(visitor_data["date"]), y="visitors", data=visitor_data, ax=ax)
        ax.set_xlabel("日付")
        ax.set_ylabel("来客数")
        st.pyplot(fig)
        
        # 曜日別平均来客数
        st.markdown("### 曜日別平均来客数")
        
        # 曜日名のマッピング
        day_names = {
            0: "月曜日", 1: "火曜日", 2: "水曜日", 3: "木曜日", 
            4: "金曜日", 5: "土曜日", 6: "日曜日"
        }
        visitor_data["day_name"] = visitor_data["day_of_week"].map(day_names)
        
        day_avg = visitor_data.groupby("day_name")["visitors"].mean().reindex([
            "月曜日", "火曜日", "水曜日", "木曜日", "金曜日", "土曜日", "日曜日"
        ]).reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="day_name", y="visitors", data=day_avg, ax=ax)
        ax.set_xlabel("曜日")
        ax.set_ylabel("平均来客数")
        st.pyplot(fig)
        
        # 月別平均来客数
        st.markdown("### 月別平均来客数")
        
        month_names = {
            1: "1月", 2: "2月", 3: "3月", 4: "4月", 5: "5月", 6: "6月",
            7: "7月", 8: "8月", 9: "9月", 10: "10月", 11: "11月", 12: "12月"
        }
        visitor_data["month_name"] = visitor_data["month"].map(month_names)
        
        month_avg = visitor_data.groupby("month_name")["visitors"].mean().reindex([
            "1月", "2月", "3月", "4月", "5月", "6月",
            "7月", "8月", "9月", "10月", "11月", "12月"
        ]).reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x="month_name", y="visitors", data=month_avg, ax=ax)
        ax.set_xlabel("月")
        ax.set_ylabel("平均来客数")
        st.pyplot(fig)
        
        # 平日・休日の比較
        st.markdown("### 平日・休日の比較")
        
        weekend_avg = visitor_data.groupby("is_weekend")["visitors"].mean().reset_index()
        weekend_avg["day_type"] = weekend_avg["is_weekend"].map({0: "平日", 1: "休日"})
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x="day_type", y="visitors", data=weekend_avg, ax=ax)
        ax.set_xlabel("日種別")
        ax.set_ylabel("平均来客数")
        st.pyplot(fig)
    
    # 来客予測タブ
    with tabs[1]:
        st.subheader("来客予測")
        
        # 予測期間の選択
        prediction_days = st.slider("予測日数", min_value=7, max_value=90, value=30, step=7)
        
        # 予測の実行
        if st.button("予測を実行"):
            with st.spinner("予測を計算中..."):
                # 予測データの生成（モックデータ）
                future_dates = [datetime.now() + timedelta(days=i) for i in range(1, prediction_days+1)]
                predictions = []
                
                for date in future_dates:
                    day_of_week = date.weekday()
                    is_weekend = 1 if day_of_week >= 5 else 0
                    
                    # 基本来客数
                    base = 30
                    
                    # 曜日の影響（週末は多い）
                    if is_weekend:
                        base += 40
                    
                    # 季節の影響
                    month = date.month
                    season_factor = np.sin(month / 12 * 2 * np.pi) * 20 + 20
                    base += season_factor
                    
                    # ランダム変動
                    noise = np.random.normal(0, 5)
                    
                    # 最終来客数
                    visitors = max(0, int(base + noise))
                    
                    predictions.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "day_of_week": day_of_week,
                        "is_weekend": is_weekend,
                        "month": month,
                        "predicted_visitors": visitors
                    })
                
                predictions_df = pd.DataFrame(predictions)
                predictions_df["date_obj"] = pd.to_datetime(predictions_df["date"])
                predictions_df["day_name"] = predictions_df["day_of_week"].map(day_names)
            
            # 予測結果の表示
            st.markdown("### 来客予測結果")
            
            # 日別予測グラフ
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(x="date_obj", y="predicted_visitors", data=predictions_df, ax=ax)
            ax.set_xlabel("日付")
            ax.set_ylabel("予測来客数")
            st.pyplot(fig)
            
            # 曜日別予測グラフ
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(
                x="date", 
                y="predicted_visitors", 
                hue="is_weekend",
                palette=["lightblue", "salmon"],
                data=predictions_df, 
                ax=ax
            )
            ax.set_xlabel("日付")
            ax.set_ylabel("予測来客数")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.legend(["平日", "休日"])
            st.pyplot(fig)
            
            # 予測データテーブル
            st.markdown("### 予測データ")
            st.dataframe(
                predictions_df[["date", "day_name", "predicted_visitors"]].rename(columns={
                    "date": "日付",
                    "day_name": "曜日",
                    "predicted_visitors": "予測来客数"
                }),
                use_container_width=True
            )
            
            # 運営提案
            st.markdown("### 運営提案")
            
            # 最も来客が多い日を特定
            max_visitors_day = predictions_df.loc[predictions_df["predicted_visitors"].idxmax()]
            
            # 平均来客数
            avg_visitors = predictions_df["predicted_visitors"].mean()
            
            # 週末の平均来客数
            weekend_avg = predictions_df[predictions_df["is_weekend"] == 1]["predicted_visitors"].mean()
            
            # 平日の平均来客数
            weekday_avg = predictions_df[predictions_df["is_weekend"] == 0]["predicted_visitors"].mean()
            
            st.markdown(f"""
            #### 来客予測に基づく運営提案
            
            1. **最も来客が多い日**: {max_visitors_day['date']} ({max_visitors_day['day_name']}) - 予測来客数: {max_visitors_day['predicted_visitors']}人
               - この日はスタッフを増員し、収穫量を増やすことをお勧めします。
            
            2. **平均来客数**: {avg_visitors:.1f}人/日
               - 平日平均: {weekday_avg:.1f}人
               - 休日平均: {weekend_avg:.1f}人
               - 休日は平日の約 {(weekend_avg/weekday_avg):.1f}倍の来客があります。
            
            3. **スタッフ配置の提案**:
               - 平日: 基本スタッフ配置
               - 休日: スタッフを {int((weekend_avg/weekday_avg) * 100 - 100)}% 増員
            
            4. **収穫量の調整**:
               - 休日前には収穫量を増やし、平日は通常量に調整することで、
                 鮮度の良い状態で提供できる量を最適化できます。
            """)
    
    # 予測モデル分析タブ
    with tabs[2]:
        st.subheader("予測モデル分析")
        
        # 特徴量の重要度
        st.markdown("### 特徴量の重要度")
        
        # モックデータでの特徴量重要度
        feature_importance = pd.DataFrame({
            "feature": ["day_of_week", "month", "has_event", "is_harvest_season", "is_holiday", 
                       "weather_rainy", "weather_sunny", "recent_visitors", "precipitation_prob", "temperature"],
            "importance": [0.345071, 0.158490, 0.157509, 0.130171, 0.079866, 
                          0.039845, 0.021621, 0.014993, 0.014450, 0.013893]
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="importance", y="feature", data=feature_importance, ax=ax)
        ax.set_xlabel("重要度")
        ax.set_ylabel("特徴量")
        st.pyplot(fig)
        
        # 特徴量の説明
        st.markdown("""
        ### 特徴量の解説
        
        1. **day_of_week（曜日）**: 最も重要な特徴量です。週末（土日）は平日に比べて来客数が大幅に増加します。
        
        2. **month（月）**: 季節による来客数の変動を表します。春や秋のシーズンは来客数が増加する傾向があります。
        
        3. **has_event（イベントの有無）**: 収穫祭などのイベント開催日は通常よりも来客数が増加します。
        
        4. **is_harvest_season（収穫時期）**: 作物の収穫最盛期には来客数が増加します。
        
        5. **is_holiday（祝日）**: 祝日は平日であっても来客数が増加します。
        
        6. **weather_rainy（雨天）**: 雨天時は来客数が減少する傾向があります。
        
        7. **weather_sunny（晴天）**: 晴天時は来客数がやや増加します。
        
        8. **recent_visitors（最近の来客数）**: 直近の来客傾向も将来の来客数に影響します。
        
        9. **precipitation_prob（降水確率）**: 降水確率が高いと来客数が減少する傾向があります。
        
        10. **temperature（気温）**: 気温も来客数に影響しますが、その影響は比較的小さいです。
        """)
        
        # 予測精度の評価
        st.markdown("### 予測モデルの精度")
        
        # モックデータでの評価指標
        evaluation = {
            "mae": 10.40,
            "rmse": 13.18,
            "r2": 0.83
        }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("MAE", f"{evaluation['mae']:.2f}")
            st.markdown("平均絶対誤差（Mean Absolute Error）")
        
        with col2:
            st.metric("RMSE", f"{evaluation['rmse']:.2f}")
            st.markdown("平方根平均二乗誤差（Root Mean Squared Error）")
        
        with col3:
            st.metric("R²", f"{evaluation['r2']:.2f}")
            st.markdown("決定係数（Coefficient of Determination）")
        
        st.markdown("""
        ### 精度評価の解説
        
        - **MAE（平均絶対誤差）**: 予測値と実際の値の差の絶対値の平均です。この値が小さいほど予測精度が高いことを示します。
        
        - **RMSE（平方根平均二乗誤差）**: 予測値と実際の値の差の二乗の平均の平方根です。外れ値に敏感な指標で、この値が小さいほど予測精度が高いことを示します。
        
        - **R²（決定係数）**: モデルがデータの変動をどれだけ説明できるかを示す指標です。1に近いほど予測精度が高いことを示します。0.83という値は、モデルがデータの変動の83%を説明できることを意味し、高い精度と言えます。
        """)

# システム情報ページ
def system_info_page():
    st.title("システム情報")
    
    st.markdown("""
    ## 観光農園予約システムについて
    
    このシステムは、観光農園の予約管理と顧客情報管理を効率化し、来客予測を行うための総合的なプラットフォームです。
    
    ### システム構成
    
    - **フロントエンド**: Streamlit（このウェブアプリケーション）
    - **バックエンド**: Python
    - **データベース**: PostgreSQL（デモ版ではモックデータを使用）
    - **分析エンジン**: scikit-learn（機械学習ライブラリ）
    
    ### 主要機能
    
    1. **予約管理システム**
       - オンライン予約受付
       - カレンダー形式での予約状況表示
       - 時間帯ごとの受付人数自動調整
       - 残り受付人数のリアルタイム表示
    
    2. **顧客管理システム**
       - 顧客基本情報の一元管理
       - 訪問履歴の記録と分析
       - リピーター分析
       - 顧客セグメント分析
    
    3. **農園情報管理**
       - 作物情報と収穫時期の管理
       - イベント情報の管理
    
    4. **来客予測モデル**
       - 機械学習による来客数予測
       - 特徴量重要度分析
       - 運営最適化提案
    
    ### 開発情報
    
    - **開発者**: Manus AI
    - **バージョン**: 1.0.0
    - **最終更新日**: 2025年4月10日
    """)
    
    # システム状態
    st.subheader("システム状態")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("登録農園数", len(farms))
    
    with col2:
        st.metric("登録顧客数", len(customers))
    
    with col3:
        st.metric("予約総数", len(reservations))
    
    # 利用方法
    st.subheader("利用方法")
    
    st.markdown("""
    1. **サイドバーのナビゲーション**から各機能にアクセスできます。
    
    2. **農園一覧**では、登録されている農園の情報を閲覧し、予約することができます。
    
    3. **予約管理**では、予約の一覧表示、新規予約の作成、予約データの分析ができます。
    
    4. **顧客管理**では、顧客情報の管理、顧客分析、セグメント分析ができます。
    
    5. **来客予測**では、過去の来客データの分析、将来の来客予測、予測モデルの分析ができます。
    """)
    
    # お問い合わせ
    st.subheader("お問い合わせ")
    
    st.markdown("""
    システムに関するお問い合わせは、以下の連絡先までお願いします。
    
    - **メール**: support@farm-reservation-system.example.com
    - **電話**: 03-XXXX-XXXX（平日 9:00-17:00）
    """)

# ヘルパー関数
def _is_current_month_in_season(start_month, end_month):
    current_month = datetime.now().month
    start = int(start_month.replace("月", ""))
    end = int(end_month.replace("月", ""))
    
    if start <= end:
        return start <= current_month <= end
    else:  # 年をまたぐ場合（例：11月〜2月）
        return current_month >= start or current_month <= end

# メイン処理
if page == "ホーム":
    home_page()
elif page == "農園一覧":
    farm_list_page()
elif page == "予約管理":
    reservation_page()
elif page == "顧客管理":
    customer_page()
elif page == "来客予測":
    prediction_page()
elif page == "システム情報":
    system_info_page()
