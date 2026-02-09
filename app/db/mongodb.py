"""
MongoDB database connection and initialization.

Handles database connection, disconnection, and provides access to the database instance.
"""

import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from app.settings import Settings
from app.utils.exceptions import DatabaseError

logger = logging.getLogger(__name__)


class MongoDB:
    """MongoDB connection manager."""

    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None

    async def connect(self, settings: Settings) -> None:
        """
        Connect to MongoDB database.

        Args:
            settings: Application settings containing MongoDB configuration

        Raises:
            DatabaseError: If connection fails
        """
        try:
            # Build MongoDB connection URL with authentication if provided
            from urllib.parse import quote_plus
            
            mongodb_url = settings.mongodb_url
            if settings.mongodb_username and settings.mongodb_password:
                # URL encode username and password to handle special characters
                encoded_username = quote_plus(settings.mongodb_username)
                encoded_password = quote_plus(settings.mongodb_password)
                
                # Parse the URL and add authentication
                if mongodb_url.startswith("mongodb://"):
                    # Extract host and port
                    url_parts = mongodb_url.replace("mongodb://", "").split("/")
                    host_port = url_parts[0]
                    database_part = url_parts[1] if len(url_parts) > 1 else ""
                    
                    # Build authenticated URL with authSource
                    mongodb_url = f"mongodb://{encoded_username}:{encoded_password}@{host_port}"
                    
                    # Parse database and query parameters
                    if database_part:
                        if "?" in database_part:
                            db_name, query_params = database_part.split("?", 1)
                            mongodb_url += f"/{db_name}?{query_params}&authSource={settings.mongodb_db_name}"
                        else:
                            mongodb_url += f"/{database_part}?authSource={settings.mongodb_db_name}"
                    else:
                        mongodb_url += f"/{settings.mongodb_db_name}?authSource={settings.mongodb_db_name}"
                elif mongodb_url.startswith("mongodb+srv://"):
                    # MongoDB Atlas connection string
                    url_parts = mongodb_url.replace("mongodb+srv://", "").split("/")
                    host = url_parts[0]
                    database_part = url_parts[1] if len(url_parts) > 1 else ""
                    
                    # Build authenticated URL with authSource
                    mongodb_url = f"mongodb+srv://{encoded_username}:{encoded_password}@{host}"
                    
                    # Parse database and query parameters
                    if database_part:
                        if "?" in database_part:
                            db_name, query_params = database_part.split("?", 1)
                            mongodb_url += f"/{db_name}?{query_params}&authSource={settings.mongodb_db_name}"
                        else:
                            mongodb_url += f"/{database_part}?authSource={settings.mongodb_db_name}"
                    else:
                        mongodb_url += f"/{settings.mongodb_db_name}?authSource={settings.mongodb_db_name}"

            # Log connection without exposing credentials
            log_url = mongodb_url.split('@')[-1] if '@' in mongodb_url else mongodb_url
            logger.info(f"Connecting to MongoDB at {log_url}")
            
            self.client = AsyncIOMotorClient(
                mongodb_url,
                maxPoolSize=settings.mongodb_max_pool_size,
                minPoolSize=settings.mongodb_min_pool_size,
                serverSelectionTimeoutMS=5000,
            )

            # Test connection
            await self.client.admin.command("ping")

            self.database = self.client[settings.mongodb_db_name]

            # Create indexes
            await self._create_indexes()

            logger.info(f"Successfully connected to MongoDB database: {settings.mongodb_db_name}")

        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise DatabaseError(f"Failed to connect to MongoDB: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            raise DatabaseError(f"Unexpected error connecting to MongoDB: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from MongoDB database."""
        if self.client:
            logger.info("Disconnecting from MongoDB")
            self.client.close()
            self.client = None
            self.database = None
            logger.info("Successfully disconnected from MongoDB")

    async def _create_indexes(self) -> None:
        """Create database indexes for optimized queries."""
        if self.database is None:
            return

        try:
            logger.info("Creating MongoDB indexes")

            # Documents collection indexes
            documents_collection = self.database.documents

            # Unique index on document_id
            await documents_collection.create_index("document_id", unique=True)

            # Index on status for filtering
            await documents_collection.create_index("status")

            # Index on uploaded_at for sorting
            await documents_collection.create_index("uploaded_at", background=True)

            # Index on file_type for filtering
            await documents_collection.create_index("file_type")

            # Compound index for common queries
            await documents_collection.create_index([("status", 1), ("uploaded_at", -1)], background=True)

            logger.info("Successfully created MongoDB indexes")

        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")

    def get_database(self) -> AsyncIOMotorDatabase:
        """
        Get the database instance.

        Returns:
            AsyncIOMotorDatabase instance

        Raises:
            DatabaseError: If not connected
        """
        if self.database is None:
            raise DatabaseError("Not connected to MongoDB. Call connect() first.")
        return self.database

    async def health_check(self) -> bool:
        """
        Check if database connection is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            if not self.client:
                return False
            await self.client.admin.command("ping")
            return True
        except Exception as e:
            logger.error(f"MongoDB health check failed: {e}")
            return False


# Global MongoDB instance
mongodb = MongoDB()


async def get_database() -> AsyncIOMotorDatabase:
    """
    Dependency to get database instance.

    Returns:
        AsyncIOMotorDatabase instance
    """
    return mongodb.get_database()
