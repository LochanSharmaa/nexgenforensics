from __future__ import annotations

from datetime import datetime

try:
    from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

    SQLALCHEMY_AVAILABLE = True
except Exception:  # pragma: no cover - optional production dependency.
    SQLALCHEMY_AVAILABLE = False
    DateTime = Float = ForeignKey = Integer = String = Text = func = None
    DeclarativeBase = object
    Mapped = mapped_column = None


if SQLALCHEMY_AVAILABLE:

    class Base(DeclarativeBase):
        __abstract__ = True

    class TimestampMixin:
        created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    class Tenant(Base, TimestampMixin):
        __tablename__ = "tenants"

        tenant_id: Mapped[str] = mapped_column(String(128), primary_key=True)
        name: Mapped[str] = mapped_column(String(255), nullable=False)

    class User(Base, TimestampMixin):
        __tablename__ = "users"

        user_id: Mapped[str] = mapped_column(String(128), primary_key=True)
        tenant_id: Mapped[str] = mapped_column(String(128), ForeignKey("tenants.tenant_id"), nullable=False)
        email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
        role: Mapped[str] = mapped_column(String(64), nullable=False)
        password_hash: Mapped[str] = mapped_column(Text, nullable=False)

    class EnrolledIdentity(Base, TimestampMixin):
        __tablename__ = "enrolled_identities"

        identity_id: Mapped[str] = mapped_column(String(128), primary_key=True)
        tenant_id: Mapped[str] = mapped_column(String(128), ForeignKey("tenants.tenant_id"), primary_key=True)
        metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    class BiometricTemplate(Base, TimestampMixin):
        __tablename__ = "biometric_templates"

        template_id: Mapped[str] = mapped_column(String(128), primary_key=True)
        identity_id: Mapped[str] = mapped_column(String(128), nullable=False)
        tenant_id: Mapped[str] = mapped_column(String(128), nullable=False)
        dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
        encrypted_payload: Mapped[str] = mapped_column(Text, nullable=False)

    class AuditLog(Base, TimestampMixin):
        __tablename__ = "audit_logs"

        audit_id: Mapped[str] = mapped_column(String(128), primary_key=True)
        tenant_id: Mapped[str] = mapped_column(String(128), nullable=False)
        actor_id: Mapped[str] = mapped_column(String(128), nullable=False)
        action: Mapped[str] = mapped_column(String(128), nullable=False)
        resource: Mapped[str] = mapped_column(String(255), nullable=False)
        metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    class Job(Base, TimestampMixin):
        __tablename__ = "jobs"

        job_id: Mapped[str] = mapped_column(String(128), primary_key=True)
        tenant_id: Mapped[str] = mapped_column(String(128), nullable=False)
        job_type: Mapped[str] = mapped_column(String(128), nullable=False)
        status: Mapped[str] = mapped_column(String(64), nullable=False)
        payload_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
        error: Mapped[str] = mapped_column(Text, nullable=False, default="")
        updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    class ModelVersion(Base, TimestampMixin):
        __tablename__ = "model_versions"

        version_id: Mapped[str] = mapped_column(String(128), primary_key=True)
        name: Mapped[str] = mapped_column(String(255), nullable=False)
        status: Mapped[str] = mapped_column(String(64), nullable=False)
        metrics_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    class IndexVersion(Base, TimestampMixin):
        __tablename__ = "index_versions"

        version_id: Mapped[str] = mapped_column(String(128), primary_key=True)
        tenant_id: Mapped[str] = mapped_column(String(128), nullable=False)
        status: Mapped[str] = mapped_column(String(64), nullable=False)
        metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    class AgencyPartition(Base, TimestampMixin):
        __tablename__ = "agency_partitions"

        agency_id: Mapped[str] = mapped_column(String(128), primary_key=True)
        tenant_id: Mapped[str] = mapped_column(String(128), ForeignKey("tenants.tenant_id"), nullable=False)
        name: Mapped[str] = mapped_column(String(255), nullable=False)
        isolation_policy: Mapped[str] = mapped_column(String(64), nullable=False, default="tenant_scoped")
        metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    class ModelAccuracyHistory(Base, TimestampMixin):
        __tablename__ = "model_accuracy_history"

        record_id: Mapped[str] = mapped_column(String(128), primary_key=True)
        version_id: Mapped[str] = mapped_column(String(128), nullable=False)
        benchmark_name: Mapped[str] = mapped_column(String(255), nullable=False)
        accuracy: Mapped[float] = mapped_column(Float, nullable=False)
        false_accept_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
        false_reject_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
        metrics_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    class AutoImproveJob(Base, TimestampMixin):
        __tablename__ = "auto_improve_jobs"

        job_id: Mapped[str] = mapped_column(String(128), primary_key=True)
        tenant_id: Mapped[str] = mapped_column(String(128), ForeignKey("tenants.tenant_id"), nullable=False)
        dataset_upload_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
        status: Mapped[str] = mapped_column(String(64), nullable=False)
        progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
        current_stage: Mapped[str] = mapped_column(String(128), nullable=False, default="queued")
        result_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
        error: Mapped[str] = mapped_column(Text, nullable=False, default="")
        updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    class DatasetUpload(Base, TimestampMixin):
        __tablename__ = "dataset_uploads"

        upload_id: Mapped[str] = mapped_column(String(128), primary_key=True)
        tenant_id: Mapped[str] = mapped_column(String(128), ForeignKey("tenants.tenant_id"), nullable=False)
        filename: Mapped[str] = mapped_column(String(512), nullable=False)
        storage_path: Mapped[str] = mapped_column(Text, nullable=False)
        size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
        checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
        status: Mapped[str] = mapped_column(String(64), nullable=False, default="uploaded")
        metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

else:
    Base = object
