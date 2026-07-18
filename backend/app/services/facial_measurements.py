from __future__ import annotations
from dataclasses import dataclass
from math import hypot
from typing import Sequence
@dataclass(frozen=True)
class Measurement:
    name: str; value: float; confidence: float; unit: str = 'ratio'
L_OUT,L_IN,R_IN,R_OUT=33,133,362,263
N_L,N_R,N_B,N_T=129,358,168,1
M_L,M_R,J_L,J_R,TOP,CHIN=61,291,234,454,10,152
def extract_face_mesh_landmarks(image_rgb):
    try: import mediapipe as mp
    except ImportError as exc: raise RuntimeError('mediapipe is required for image landmark extraction') from exc
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True) as mesh: result=mesh.process(image_rgb)
    if not result.multi_face_landmarks: raise ValueError('No face mesh detected in image')
    return [(p.x,p.y,p.z) for p in result.multi_face_landmarks[0].landmark]
def pose_confidence(yaw,pitch): return round(max(.25,1-max(0.,max(abs(yaw),abs(pitch))-15)*.75/30),4)
def compute_measurements(points: Sequence[Sequence[float]], *, pose_yaw=0., pose_pitch=0., calibration_reference_present=False):
    if len(points)<468: raise ValueError('Face Mesh output must include at least 468 landmarks')
    d=lambda a,b:hypot(float(points[a][0])-float(points[b][0]),float(points[a][1])-float(points[b][1]))
    ipd=d(L_IN,R_IN); fw=d(J_L,J_R); fh=d(TOP,CHIN)
    if min(ipd,fw,fh)<=1e-12: raise ValueError('Degenerate landmark geometry')
    c=pose_confidence(pose_yaw,pose_pitch); unit='ratio' if not calibration_reference_present else 'ratio (calibration present)'
    mid=(points[L_IN][0]+points[R_IN][0])/2; pairs=((L_OUT,R_OUT),(M_L,M_R),(N_L,N_R),(J_L,J_R)); symmetry=max(0,min(1,1-sum(abs(abs(points[a][0]-mid)-abs(points[b][0]-mid))/ipd for a,b in pairs)/4))
    return [Measurement('interpupillary_distance_ratio',ipd/fw,c,unit),Measurement('eye_width_ratio',(d(L_OUT,L_IN)+d(R_IN,R_OUT))/2/ipd,c,unit),Measurement('nose_width_to_length_ratio',d(N_L,N_R)/d(N_B,N_T),c,unit),Measurement('mouth_width_ratio',d(M_L,M_R)/fw,c,unit),Measurement('jaw_width_ratio',fw/fh,c,unit),Measurement('facial_width_to_height_ratio',fw/fh,c,unit),Measurement('symmetry_score',round(symmetry,6),c,'score')]
