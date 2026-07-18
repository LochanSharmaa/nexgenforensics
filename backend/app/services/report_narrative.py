from app.schemas import ForensicFindings
STANDARD_SCOPE_TEXT='This report reflects automated facial similarity analysis performed by the NexGen system at [config_version]. Automated comparison results constitute an investigative lead and are distinct from human forensic morphological examination. Findings are subject to review by a qualified examiner prior to evidentiary or operational use.'
def build_narrative(f: ForensicFindings):
    low=[]
    for image in f.source_images:
        if image.quality_score<.6: low.append('image quality below the accepted threshold')
        if abs(image.pose_yaw)>15 or abs(image.pose_pitch)>15: low.append('pose angle exceeds 15 degrees')
        if image.liveness_score<.75: low.append('liveness score below the accepted threshold')
    quality='No material quality or pose limitations were identified.' if not low else 'Limiting factors requiring examiner review: '+ '; '.join(sorted(set(low)))+'.'
    return {'methodology_summary':f'{len(f.model_scores)} comparison model(s) were used under configuration {f.config_version}; decision threshold: {f.threshold_value:.3f}.','quality_notes':quality,'scope_statement':STANDARD_SCOPE_TEXT.replace('[config_version]',f.config_version)}
