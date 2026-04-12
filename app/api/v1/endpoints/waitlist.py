from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, EmailStr

router = APIRouter(prefix="/waitlist", tags=["waitlist"])


class WaitlistRequest(BaseModel):
    email: EmailStr
    name: Optional[str] = None
    tier: Optional[str] = None
    source: Optional[str] = None


@router.post("")
async def join_waitlist(data: WaitlistRequest):
    from app.db.mongodb import get_database

    db = await get_database()
    await db.waitlist.update_one(
        {"email": data.email},
        {
            "$set": {
                "email": data.email,
                "name": data.name,
                "tier": data.tier,
                "source": data.source,
                "updated_at": datetime.now(timezone.utc),
            },
            "$setOnInsert": {
                "created_at": datetime.now(timezone.utc),
            },
        },
        upsert=True,
    )
    return {"status": "ok", "message": "You're on the list!"}
