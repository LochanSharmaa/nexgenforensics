import base64
from io import BytesIO
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

def test_imatch_generate_report_returns_audited_scoped_report():
    image=Image.new('RGB',(120,120),(100,120,140)); buffer=BytesIO(); image.save(buffer,format='JPEG')
    response=TestClient(app).post('/api/imatch/search',json={'image_base64':base64.b64encode(buffer.getvalue()).decode(),'purpose':'test','lawful_use_reason':'test','generate_report':True,'case_id':'HOOK-1'})
    assert response.status_code==200
    report=response.json()['report']; assert report['report_url']
    assert Path(report['report_url']).exists() and report['audit_hash']==report['findings']['audit_hash']
