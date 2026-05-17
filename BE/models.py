from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from BE.database import Base

class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    version_name = Column(String, unique=True, index=True) # e.g., v1, v2
    model_path = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, default=False)

    training_runs = relationship("TrainingRun", back_populates="model_version")
    predictions = relationship("Prediction", back_populates="model_version")

class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"))
    epochs = Column(Integer)
    image_size = Column(Integer)
    dataset_count = Column(Integer)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    map50 = Column(Float, nullable=True)
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)
    status = Column(String) # running, completed, failed

    model_version = relationship("ModelVersion", back_populates="training_runs")

class ImageAsset(Base):
    __tablename__ = "image_assets"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    filepath = Column(String)
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    review_items = relationship("ReviewQueueItem", back_populates="image_asset")
    predictions = relationship("Prediction", back_populates="image_asset")

class ReviewQueueItem(Base):
    __tablename__ = "review_queue_items"

    id = Column(Integer, primary_key=True, index=True)
    image_asset_id = Column(Integer, ForeignKey("image_assets.id"))
    status = Column(String) # pending, accepted, rejected, skipped
    manual_correction = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    reviewed_at = Column(DateTime, nullable=True)

    image_asset = relationship("ImageAsset", back_populates="review_items")

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    image_asset_id = Column(Integer, ForeignKey("image_assets.id"))
    model_version_id = Column(Integer, ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True)
    class_name = Column(String)
    confidence = Column(Float)
    bbox_x = Column(Float)
    bbox_y = Column(Float)
    bbox_w = Column(Float)
    bbox_h = Column(Float)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    image_asset = relationship("ImageAsset", back_populates="predictions")
    model_version = relationship("ModelVersion", back_populates="predictions")
