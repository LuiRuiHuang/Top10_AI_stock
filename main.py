import os
import json
import requests
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. 載入環境變數
load_dotenv()
FINMIND_TOKEN = os.getenv("FINMIND_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 初始化 Gemini 監控客戶端
client = genai.Client(api_key=GEMINI_API_KEY)

# 定義台股核心 AI 概念股清單
AI_STOCKS = {
    "2330": "台積電", "2317": "鴻海", "2454": "聯發科",
    "2382": "廣達", "3231": "緯創", "2376": "技嘉",
    "2356": "英業達", "2301": "光寶科", "6669": "緯穎", "2377": "微星"
}


def fetch_stock_data(stock_id):
    """透過 FinMind API 獲取最新的技術面與基本面資料"""
    # 獲取最新股價 (TaiwanStockPrice)
    price_url = "https://api.finmindtrade.com/api/v4/data"
    price_params = {
        "dataset": "TaiwanStockPrice",
        "data_id": stock_id,
        "start_date": "2026-05-01",  # 抓取近期數據
        "token": FINMIND_TOKEN
    }

    try:
        price_res = requests.get(price_url, params=price_params).json()
        if "data" not in price_res or not price_res["data"]:
            print(f"❌ 股票 {stock_id} 找不到價格數據。API 回傳訊息: {price_res.get('msg', '無訊息')}")
            return None

        price_df = pd.DataFrame(price_res["data"])
        latest_info = price_df.iloc[-1]

        # 獲取基本面 EPS (TaiwanStockFinancialStatements)
        fin_params = {
            "dataset": "TaiwanStockFinancialStatements",
            "data_id": stock_id,
            "start_date": "2025-01-01",
            "token": FINMIND_TOKEN
        }
        fin_res = requests.get(price_url, params=fin_params).json()
        if "data" not in fin_res or not fin_res["data"]:
            print(f"❌ 股票 {stock_id} 找不到財報數據。")
            return None

        fin_df = pd.DataFrame(fin_res["data"])
        eps_df = fin_df[fin_df["type"] == "EPS"].tail(4)
        if eps_df.empty:
            print(f"⚠️ 股票 {stock_id} 無法計算近四季 EPS。")
            return None

        # 篩選近四季 EPS 並加總
        total_eps = eps_df["value"].astype(float).sum()

        return {
            "stock_id": stock_id,
            "name": AI_STOCKS[stock_id],
            "current_price": float(latest_info["close"]),
            "four_quarters_eps": round(total_eps, 2)
        }
    except Exception as e:
        print(f"無法取得股票 {stock_id} 的數據: {e}")
        return None


def analyze_with_gemini(stock_info):
    """利用 Gemini SDK 進行結構化價格估值分析"""

    # 透過 Pydantic 定義我們期望 AI 回傳的精準 JSON 格式 (面試大加分！)
    from pydantic import BaseModel
    class StockAnalysis(BaseModel):
        valuation: str  # 便宜價、合理價、昂貴價
        reasoning: str  # 分析解釋原因
        fair_price_estimate: float  # AI 預估的合理價

    prompt = f"""
    你是一位專業的台股價值投資分析師。請根據以下個股的基本面數據，計算並評估目前價格。

    股票名稱: {stock_info['name']} ({stock_info['stock_id']})
    目前股價: {stock_info['current_price']} 元
    近四季累積 EPS: {stock_info['four_quarters_eps']} 元

    請依據普遍的本益比(PE)觀念（例如：AI核心權值股合理本益比約 15-20 倍，高成長可能更高），
    評估該股目前屬於「便宜價」、「合理價」還是「昂貴價」，並給出詳細的分析原因與預估的合理價。
    """

    # 呼叫最新 Gemini 2.5 flash 模型並要求結構化輸出
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=StockAnalysis,
            temperature=0.2  # 降低隨機性，讓估值更穩定
        ),
    )

    return json.loads(response.text)


def send_notification(stock_name, stock_id, price, analysis):
    """便宜價觸發通知 (此處以 Console 輸出與 Line Notify 架構示範)"""
    message = f"""
    🚨 【台股 AI 便宜價警報】 🚨
    標的：{stock_name} ({stock_id})
    當前價格：{price} 元 (AI 預估合理價：{analysis['fair_price_estimate']} 元)
    評級結果：【{analysis['valuation']}】
    分析原因：{analysis['reasoning']}
    """
    print(message)
    # 💡 提示：若想更完整，可在此處串接 Line Notify 或 Telegram Bot API 發送簡訊。


if __name__ == "__main__":
    print("🚀 開始執行台股 AI 概念股價值分析專案...")

    for stock_id in AI_STOCKS.keys():
        data = fetch_stock_data(stock_id)
        if data:
            print(f"正在分析 {data['name']}...")
            analysis_result = analyze_with_gemini(data)

            # 判斷是否為便宜價，若是則觸發通知
            if "便宜" in analysis_result["valuation"]:
                send_notification(data['name'], stock_id, data['current_price'], analysis_result)
            else:
                print(
                    f"-> {data['name']} 目前處於 {analysis_result['valuation']} (價格: {data['current_price']})，未觸發通知。")
            time.sleep(5)