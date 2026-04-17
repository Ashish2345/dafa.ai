"""
LLM usage repository — tracks every LLM call with tokens, cost, and metadata.

Schema for `llm_usage` docs:
    {
      user_id: str,
      timestamp: datetime,
      model: str,                  # e.g. "gemini-2.5-flash"
      purpose: str,                # e.g. "answer_synthesis", "follow_up_generation"
      input_tokens: int,
      output_tokens: int,
      thinking_tokens: int,
      cost_usd: float,
      processing_time_s: float,
      collection_name: str | null, # which document workspace
      query_preview: str,          # first 200 chars of the query
    }
"""

from datetime import datetime, timezone
from typing import Optional

from loguru import logger


class LlmUsageRepository:
    """CRUD for the `llm_usage` MongoDB collection."""

    def __init__(self, database):
        self.collection = database.llm_usage

    async def record(
        self,
        user_id: str,
        model: str,
        purpose: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        thinking_tokens: int = 0,
        cost_usd: float = 0.0,
        processing_time_s: float = 0.0,
        collection_name: Optional[str] = None,
        query_preview: str = "",
    ) -> None:
        """Record a single LLM call."""
        doc = {
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc),
            "model": model,
            "purpose": purpose,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
            "cost_usd": round(cost_usd, 8),
            "processing_time_s": round(processing_time_s, 3),
            "collection_name": collection_name,
            "query_preview": query_preview[:200],
        }
        await self.collection.insert_one(doc)
        logger.debug(
            f"LLM usage recorded: {user_id} {model} {purpose} "
            f"in={input_tokens} out={output_tokens} think={thinking_tokens} "
            f"${cost_usd:.6f}"
        )

    async def get_user_totals(self, user_id: str) -> dict:
        """Aggregate totals for a user (all time)."""
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {
                "_id": None,
                "total_calls": {"$sum": 1},
                "total_input_tokens": {"$sum": "$input_tokens"},
                "total_output_tokens": {"$sum": "$output_tokens"},
                "total_thinking_tokens": {"$sum": "$thinking_tokens"},
                "total_cost_usd": {"$sum": "$cost_usd"},
            }},
        ]
        result = await self.collection.aggregate(pipeline).to_list(1)
        if not result:
            return {"total_calls": 0, "total_input_tokens": 0, "total_output_tokens": 0, "total_thinking_tokens": 0, "total_cost_usd": 0.0}
        r = result[0]
        r.pop("_id", None)
        return r

    async def get_daily_totals(self, days: int = 30) -> list[dict]:
        """Aggregate daily cost across all users for the last N days."""
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        pipeline = [
            {"$match": {"timestamp": {"$gte": cutoff}}},
            {"$group": {
                "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
                "calls": {"$sum": 1},
                "input_tokens": {"$sum": "$input_tokens"},
                "output_tokens": {"$sum": "$output_tokens"},
                "cost_usd": {"$sum": "$cost_usd"},
            }},
            {"$sort": {"_id": 1}},
        ]
        results = await self.collection.aggregate(pipeline).to_list(days + 1)
        return [{"date": r["_id"], "calls": r["calls"], "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"], "cost_usd": round(r["cost_usd"], 6)} for r in results]

    async def get_model_breakdown(self, days: int = 30) -> list[dict]:
        """Cost breakdown by model for the last N days."""
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        pipeline = [
            {"$match": {"timestamp": {"$gte": cutoff}}},
            {"$group": {
                "_id": "$model",
                "calls": {"$sum": 1},
                "cost_usd": {"$sum": "$cost_usd"},
                "total_tokens": {"$sum": {"$add": ["$input_tokens", "$output_tokens", "$thinking_tokens"]}},
            }},
            {"$sort": {"cost_usd": -1}},
        ]
        results = await self.collection.aggregate(pipeline).to_list(20)
        return [{"model": r["_id"], "calls": r["calls"], "cost_usd": round(r["cost_usd"], 6), "total_tokens": r["total_tokens"]} for r in results]
