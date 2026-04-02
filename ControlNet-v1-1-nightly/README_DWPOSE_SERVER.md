# DWPose 常驻推理服务（FastAPI）

本目录新增 `dwpose_server.py`，用于把 DWPose 作为常驻服务运行，便于 Java 项目（如 dyq）并发调用。

## 1. 安装依赖

```bash
pip install fastapi uvicorn python-multipart onnxruntime
# 如果使用 GPU
pip install onnxruntime-gpu
```

## 2. 准备模型

把以下模型放到 `annotator/ckpts/`：

- `yolox_l.onnx`
- `dw-ll_ucoco_384.onnx`

也可通过环境变量自定义路径：

- `DWPOSE_DET_PATH`
- `DWPOSE_POSE_PATH`

## 3. 启动服务

```bash
cd ControlNet-v1-1-nightly
python dwpose_server.py
```

可选环境变量：

- `DWPOSE_HOST`（默认 `0.0.0.0`）
- `DWPOSE_PORT`（默认 `8000`）
- `DWPOSE_DEVICE`（`cuda` 或 `cpu`，默认 `cuda`）

## 4. 接口说明

### 健康检查

```bash
curl http://127.0.0.1:8000/healthz
```

### 上传图片，返回姿态渲染 PNG

```bash
curl -X POST \
  -F "file=@test_imgs/pose1.png" \
  http://127.0.0.1:8000/v1/dwpose/render \
  --output pose.png
```

### 上传图片，返回 base64 JSON

```bash
curl -X POST \
  -F "file=@test_imgs/pose1.png" \
  http://127.0.0.1:8000/v1/dwpose/render_base64
```


## 5. 导出到 dyq 项目（关键 Python 文件）

如果你希望只把关键运行文件复制到 `dyq`，可以执行：

```bash
python tools/export_to_dyq.py /path/to/dyq
```

会复制以下关键文件到 dyq 对应目录：

- `ControlNet-v1-1-nightly/dwpose_server.py`
- `ControlNet-v1-1-nightly/annotator/dwpose/__init__.py`
- `ControlNet-v1-1-nightly/annotator/dwpose/wholebody.py`
- `ControlNet-v1-1-nightly/annotator/dwpose/onnxdet.py`
- `ControlNet-v1-1-nightly/annotator/dwpose/onnxpose.py`
- `ControlNet-v1-1-nightly/annotator/dwpose/util.py`

然后在 dyq 项目中准备模型文件：

- `ControlNet-v1-1-nightly/annotator/ckpts/yolox_l.onnx`
- `ControlNet-v1-1-nightly/annotator/ckpts/dw-ll_ucoco_384.onnx`

