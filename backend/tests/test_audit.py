from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from imatch_api.db.models import AuditRecord, Role
from imatch_api.db.session import get_engine
from imatch_api.services.audit_service import ACTION_LOGIN, AuditService

from .conftest import TEST_PASSWORD


@pytest.fixture
def audit(tmp_path):
    return AuditService(tmp_path / "chain.jsonl")


class TestChainIntegrity:
    def test_empty_chain_is_valid(self, audit, tenant_factory):
        tenant = tenant_factory()
        with Session(get_engine()) as session:
            assert audit.verify_chain(session, tenant.id).valid is True

    def test_chain_links_records_in_order(self, audit, tenant_factory):
        tenant = tenant_factory()
        with Session(get_engine()) as session:
            first = audit.record(session, tenant_id=tenant.id, action="a.one")
            second = audit.record(session, tenant_id=tenant.id, action="a.two")
            session.commit()

            assert first.previous_hash == ""
            assert second.previous_hash == first.entry_hash
            assert audit.verify_chain(session, tenant.id).valid is True

    def test_editing_a_record_breaks_verification(self, audit, tenant_factory):
        """This is the property the whole audit design exists for."""
        tenant = tenant_factory()
        with Session(get_engine()) as session:
            audit.record(session, tenant_id=tenant.id, action="a.one")
            target = audit.record(session, tenant_id=tenant.id, action="a.two")
            audit.record(session, tenant_id=tenant.id, action="a.three")
            session.commit()

            stored = session.get(AuditRecord, target.id)
            stored.outcome = "quietly changed"
            session.add(stored)
            session.commit()

            verification = audit.verify_chain(session, tenant.id)
            assert verification.valid is False
            assert verification.broken_at == target.id

    def test_deleting_a_record_breaks_verification(self, audit, tenant_factory):
        tenant = tenant_factory()
        with Session(get_engine()) as session:
            audit.record(session, tenant_id=tenant.id, action="a.one")
            middle = audit.record(session, tenant_id=tenant.id, action="a.two")
            audit.record(session, tenant_id=tenant.id, action="a.three")
            session.commit()

            session.delete(session.get(AuditRecord, middle.id))
            session.commit()

            assert audit.verify_chain(session, tenant.id).valid is False

    def test_chains_are_independent_per_tenant(self, audit, tenant_factory):
        first = tenant_factory("tenant-one")
        second = tenant_factory("tenant-two")
        with Session(get_engine()) as session:
            audit.record(session, tenant_id=first.id, action="a.one")
            entry = audit.record(session, tenant_id=second.id, action="a.one")
            session.commit()

            # A second tenant's first record starts its own chain rather than
            # continuing another tenant's.
            assert entry.previous_hash == ""
            assert audit.verify_chain(session, first.id).valid is True
            assert audit.verify_chain(session, second.id).valid is True

    def test_records_are_mirrored_to_disk(self, audit, tenant_factory, tmp_path):
        tenant = tenant_factory()
        with Session(get_engine()) as session:
            record = audit.record(session, tenant_id=tenant.id, action="a.one")
            session.commit()
            # Read the hash before the session closes: commit expires attributes,
            # and a detached instance cannot refresh them.
            entry_hash = record.entry_hash

        lines = (tmp_path / "chain.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["entry_hash"] == entry_hash


class TestAuditThroughTheApi:
    def test_login_is_recorded(self, client: TestClient, tenant_factory, user_factory, auth_headers):
        tenant = tenant_factory()
        user_factory(tenant, email="admin@example.com", role=Role.ADMIN)
        headers = auth_headers("admin@example.com")

        records = client.get("/api/audit", headers=headers).json()
        assert any(record["action"] == ACTION_LOGIN for record in records)

    def test_failed_login_is_recorded(self, client: TestClient, tenant_factory, user_factory, auth_headers):
        tenant = tenant_factory()
        user_factory(tenant, email="admin@example.com", role=Role.ADMIN)
        client.post("/api/auth/login", json={"email": "admin@example.com", "password": "wrong-one-1!"})

        headers = auth_headers("admin@example.com")
        records = client.get("/api/audit", headers=headers).json()
        assert any(record["action"] == "auth.login_failed" for record in records)

    def test_audit_is_scoped_to_the_tenant(
        self, client: TestClient, tenant_factory, user_factory, auth_headers
    ):
        first = tenant_factory("tenant-one")
        user_factory(first, email="one@example.com", role=Role.ADMIN)
        second = tenant_factory("tenant-two")
        user_factory(second, email="two@example.com", role=Role.ADMIN)

        client.get("/api/audit", headers=auth_headers("one@example.com"))
        records = client.get("/api/audit", headers=auth_headers("two@example.com")).json()

        assert all(record["actor_label"] != "one@example.com" for record in records)

    def test_only_admins_can_verify_the_chain(
        self, client: TestClient, tenant_factory, user_factory, auth_headers
    ):
        tenant = tenant_factory()
        user_factory(tenant)
        assert client.get("/api/audit/verify", headers=auth_headers()).status_code == 403

    def test_admin_verification_passes_on_a_clean_chain(
        self, client: TestClient, tenant_factory, user_factory, auth_headers
    ):
        tenant = tenant_factory()
        user_factory(tenant, email="admin@example.com", role=Role.ADMIN)
        response = client.get("/api/audit/verify", headers=auth_headers("admin@example.com"))
        assert response.status_code == 200
        assert response.json()["valid"] is True

    def test_case_creation_is_audited(
        self, client: TestClient, tenant_factory, user_factory, auth_headers
    ):
        tenant = tenant_factory()
        user_factory(tenant, email="admin@example.com", role=Role.ADMIN)
        headers = auth_headers("admin@example.com")

        created = client.post(
            "/api/cases",
            headers=headers,
            json={"reference": "OP-AUDIT", "title": "Audited", "lawful_basis": "Warrant 1"},
        ).json()

        records = client.get("/api/audit", headers=headers, params={"resource_id": created["id"]}).json()
        assert any(record["action"] == "case.create" for record in records)
        assert any(record["lawful_basis"] == "Warrant 1" for record in records)
