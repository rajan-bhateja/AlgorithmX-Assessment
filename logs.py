import os
import json
import re
from sqlalchemy import create_engine, Column, String, Text, Integer, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base
from dotenv import load_dotenv


load_dotenv()
# ==============================
# DATABASE CONFIGURATION
# ==============================
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise ValueError("DATABASE_URL not set — check:\n- docker-compose.yaml (running Docker)\n- .env (running locally)")
engine = create_engine(DB_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ==============================
# LOG MODEL
# ==============================
class Log(Base):
    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String, nullable=False)
    current_login = Column(String, nullable=False)
    user_name = Column(String, nullable=False)
    user_prompt = Column(Text, nullable=False)
    user_uploads = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    created_at = Column(DateTime)


# Create table if it doesn't exist
Base.metadata.create_all(bind=engine)

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)


# ==============================
# AUTOMATED LOG FUNCTION
# ==============================

def log_response(source: str, current_login: str, user_name: str, user_prompt: str, user_uploads: str, answer: str):
    """
    Automatically logs an answer:
      1. Saves JSON file in logs/
      2. Stores entry in PostgreSQL
    """
    # --- Save as JSON file ---
    safe_filename = re.sub(r'[<>:"/\\|?*]', "_", current_login)
    log_path = os.path.join("logs", f"{safe_filename}.json")
    log_data = {
        "source": source,
        "current_login": current_login,
        "user_name": user_name,
        "user_prompt": user_prompt,
        "user_uploads": user_uploads,
        "answer": answer
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f)

    # --- Insert into PostgreSQL ---
    db = SessionLocal()
    try:
        log_entry = Log(
            source=source,
            current_login=current_login,
            user_name=user_name,
            user_prompt=user_prompt,
            user_uploads=user_uploads,
            answer=answer
        )
        db.add(log_entry)
        db.commit()
    finally:
        db.close()

    return log_path
