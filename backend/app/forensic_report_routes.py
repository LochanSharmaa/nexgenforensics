from datetime import datetime, timezone
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from app.schemas import ForensicFindings, MeasurementFinding, ModelSimilarityScore, SourceImageFindings
from app.services.report_pdf import render_report_pdf, tenant_report_path
from nexgen_engine.auth import AuthService, require_role
from nexgen_engine.security.audit_logger import AuditLogger
class ReportGenerateRequest(BaseModel):
    case_id:str; source_images:list[SourceImageFindings]; model_scores:list[ModelSimilarityScore]; fused_score:float=Field(ge=0,le=1); calibrated_match_probability:float=Field(ge=0,le=100); threshold_value:float=Field(ge=0,le=1); false_match_rate:float=Field(ge=0,le=1); measurements:list[MeasurementFinding]=[]; config_version:str='nexgen-default-v1'
def build_forensic_report_router(auth:AuthService,audit:AuditLogger,root):
    router=APIRouter(prefix='/api/v1/forensic-report',tags=['forensic-report'])
    @router.post('/generate')
    def generate(p:ReportGenerateRequest,authorization:str|None=Header(default=None)):
        if not authorization or not authorization.lower().startswith('bearer '): raise HTTPException(401,'Bearer token required.')
        try: principal=auth.verify_token(authorization.split(' ',1)[1])
        except Exception as exc: raise HTTPException(401,'Invalid token.') from exc
        require_role(principal.role,'operator'); decision='match' if p.fused_score>=p.threshold_value else 'no_match'
        entry=audit.append(principal.user_id,'forensic_report_generate',decision,p.fused_score,{'case_id':p.case_id,'tenant_id':principal.tenant_id})
        f=ForensicFindings(case_id=p.case_id,examiner_id=principal.user_id,timestamp=datetime.now(timezone.utc),tenant_id=principal.tenant_id,source_images=p.source_images,model_scores=p.model_scores,fused_score=p.fused_score,calibrated_match_probability=p.calibrated_match_probability,decision=decision,threshold_value=p.threshold_value,false_match_rate=p.false_match_rate,measurements=p.measurements,audit_hash=entry.entry_hash,config_version=p.config_version)
        path=tenant_report_path(root,principal.tenant_id,p.case_id,entry.entry_hash)
        try: render_report_pdf(f,path)
        except RuntimeError as exc: raise HTTPException(503,str(exc)) from exc
        return {'report_path':str(path),'audit_hash':entry.entry_hash,'findings':f.model_dump(mode='json')}
    return router

def generate_automated_report(*, audit: AuditLogger, root, case_id: str, tenant_id: str, examiner_id: str, source_images: list[SourceImageFindings], model_scores: list[ModelSimilarityScore], fused_score: float, threshold_value: float = 0.72, false_match_rate: float = 0.001, config_version: str = 'nexgen-default-v1') -> dict:
    """Shared auto-trigger path; audit logging is mandatory for every report."""
    decision = 'match' if fused_score >= threshold_value else 'no_match'
    entry = audit.append(examiner_id, 'forensic_report_generate', decision, fused_score, {'case_id': case_id, 'tenant_id': tenant_id, 'trigger': 'automatic'})
    findings = ForensicFindings(case_id=case_id, examiner_id=examiner_id, timestamp=datetime.now(timezone.utc), tenant_id=tenant_id, source_images=source_images, model_scores=model_scores, fused_score=fused_score, calibrated_match_probability=round(fused_score * 100, 2), decision=decision, threshold_value=threshold_value, false_match_rate=false_match_rate, measurements=[], audit_hash=entry.entry_hash, config_version=config_version)
    path = tenant_report_path(root, tenant_id, case_id, entry.entry_hash)
    render_report_pdf(findings, path)
    return {'report_path': str(path), 'audit_hash': entry.entry_hash, 'findings': findings.model_dump(mode='json')}
