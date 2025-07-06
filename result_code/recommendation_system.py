import httpx
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
import asyncio

class RecommendSystem:
    _instance = None  # Singleton 패턴

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RecommendSystem, cls).__new__(cls)
        return cls._instance

    def __init__(self, n_clusters=5, threshold=10):
        self.n_clusters = n_clusters
        self.threshold = threshold
        self.user_df = None
        self.game_columns = None
        self.feature_cols = ["age", "tall", "weight", "gender"]
        self.kmeans = None
        self.cluster_means = None
        self.all_recommendations = []  # 모든 유저 추천 결과 캐시

    # 비동기 데이터 조회
    async def fetch_data(self, url):
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            df_raw = pd.DataFrame(response.json()["resultList"])
            return df_raw

    # 데이터 전처리
    def preprocess(self, df_raw):
        # 피벗 테이블 생성
        score_matrix = (
            df_raw
            .pivot_table(index="userId", columns="gameId", values="score", aggfunc="mean")
            .reset_index()
        )
        self.game_columns = [c for c in score_matrix.columns if c != "userId"]
        # 유저 정보 추출
        profile_cols = ["userId", "age", "tall", "weight", "gender"]
        user_profiles = df_raw[profile_cols].drop_duplicates("userId")
        # 유저 정보와 게임 점수 결합
        self.user_df = user_profiles.merge(score_matrix, on="userId", how="left").fillna(0)

    # 클러스터링
    def cluster_users(self):
        scaler = StandardScaler()
        X = scaler.fit_transform(self.user_df[self.feature_cols])
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.user_df["cluster"] = self.kmeans.fit_predict(X)
        self.cluster_means = (
            self.user_df.groupby("cluster")[self.game_columns]
            .mean()
            .rename_axis("cluster")
        )

    # 약점 게임 추천
    def recommend_weak_games(self, userId, threshold=None, top_n=2):
        if threshold is None:
            threshold = self.threshold
        row = self.user_df[self.user_df["userId"] == userId].iloc[0]
        cid = row["cluster"]
        personal = row[self.game_columns]
        cluster_avg = self.cluster_means.loc[cid]
        gap = cluster_avg - personal
        weak_games = gap[gap > threshold].sort_values(ascending=False)
        if len(weak_games) == 0:
            weak_games = gap.sort_values(ascending=False).head(top_n)
        else:
            weak_games = weak_games.head(top_n)
        return list(weak_games.index)

    def cache_all_recommendations(self, threshold=10, top_n=2):
        results = []
        for userId in self.user_df["userId"]:
            try:
                rcmdGameList = self.recommend_weak_games(
                    userId, threshold, top_n
                )
                user_row = self.user_df[self.user_df["userId"] == userId].iloc[0]
                cluster_id = user_row["cluster"]
                results.append({
                    "userId": userId,
                    "rcmdGameList": rcmdGameList,
                    "cluster_id": int(cluster_id),
                })
            except Exception:
                continue
        self.all_recommendations = results

    async def send_recommendations_to_api(self, api_url, threshold=10, top_n=2):
        """
        모든 유저의 추천 결과를 외부 API로 전송
        """
        async with httpx.AsyncClient() as client:
            payload = []
            for userId in self.user_df["userId"]:
                try:
                    rcmdGameList = self.recommend_weak_games(userId, threshold, top_n)
                    user_data = {
                        "userId": userId,
                        "rcmdGameList": rcmdGameList
                    }
                    payload.append(user_data)
                except Exception as e:
                    print(f"유저 {userId} 추천 계산 중 오류: {e}")
                    continue
            
            try:
                response = await client.post(api_url, json=payload)
                return {
                    "status": response.status_code,
                    "message": f"추천 결과 전송 완료. 총 {len(payload)}개 유저 데이터 전송",
                    "response": response.text
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"API 전송 중 오류: {str(e)}"
                }

# Pydantic 모델들
class SendRecommendationsRequest(BaseModel):
    api_url: str
    threshold: Optional[int] = 10
    top_n: Optional[int] = 2

# FastAPI 앱 생성
app = FastAPI(title="Recommendation System API", version="1.0.0")

# 전역 변수로 recommender 인스턴스 관리
recommender = None

@app.on_event("startup")
async def startup_event():
    global recommender
    # 서버 시작 시 데이터 로드 및 클러스터링
    url = "http://localhost:8080/user/a6c92e61-2d4e-4d5f-8b11-77e6c4a9be89/result/all"
    recommender = RecommendSystem(n_clusters=5, threshold=10)
    df_raw = await recommender.fetch_data(url)
    recommender.preprocess(df_raw)
    recommender.cluster_users()
    recommender.cache_all_recommendations()  # 추천 결과 미리 계산
    print("추천 시스템이 초기화되었습니다.")

@app.post("/recommend/send")
async def send_recommendations(request: SendRecommendationsRequest):
    """
    추천 결과를 외부 API로 전송
    """
    global recommender
    
    if recommender is None:
        raise HTTPException(status_code=500, detail="추천 시스템이 초기화되지 않았습니다.")
    
    try:
        result = await recommender.send_recommendations_to_api(
            request.api_url,
            request.threshold,
            request.top_n
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추천 결과 전송 중 오류: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)