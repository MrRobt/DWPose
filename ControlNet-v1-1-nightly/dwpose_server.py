import base64
import os
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from annotator.dwpose import DWposeDetector

app = FastAPI(title='DWPose Service', version='1.0.0')
_detector: Optional[DWposeDetector] = None


@app.on_event('startup')
def startup_event() -> None:
    global _detector
    _detector = DWposeDetector()


@app.get('/healthz')
def healthz() -> dict:
    return {'status': 'ok'}


@app.post('/v1/dwpose/render')
async def render_pose(file: UploadFile = File(...)) -> Response:
    if _detector is None:
        raise HTTPException(status_code=503, detail='Detector not initialized')

    content = await file.read()
    buffer = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail='Invalid image file')

    pose_image = _detector(image)
    ok, encoded = cv2.imencode('.png', pose_image)
    if not ok:
        raise HTTPException(status_code=500, detail='Failed to encode image')

    return Response(content=encoded.tobytes(), media_type='image/png')


@app.post('/v1/dwpose/render_base64')
async def render_pose_base64(file: UploadFile = File(...)) -> JSONResponse:
    if _detector is None:
        raise HTTPException(status_code=503, detail='Detector not initialized')

    content = await file.read()
    buffer = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail='Invalid image file')

    pose_image = _detector(image)
    ok, encoded = cv2.imencode('.png', pose_image)
    if not ok:
        raise HTTPException(status_code=500, detail='Failed to encode image')

    return JSONResponse({
        'width': int(pose_image.shape[1]),
        'height': int(pose_image.shape[0]),
        'bytes': int(encoded.size),
        'image_base64': base64.b64encode(encoded.tobytes()).decode('utf-8'),
    })


if __name__ == '__main__':
    import uvicorn

    host = os.getenv('DWPOSE_HOST', '0.0.0.0')
    port = int(os.getenv('DWPOSE_PORT', '8000'))
    uvicorn.run('dwpose_server:app', host=host, port=port, reload=False)
