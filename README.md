# CameraFaceDetection
 A real-time face dectection tool for camera

## Dependencies
<pre><code>
pip install insightface==0.2.0
pip install onnx
pip install onnxruntime-gpu
</code></pre>

## Usage

Download face detection models from [Google Driver](https://drive.google.com/file/d/1amwJw2Oiq2OIocHjjKBnByLy7dqkCFAN/view?usp=sharing).

Unzip ```models.zip``` in ```./insightface_func/```.

Directory Structure:
```
-./
    |---insightface_func/
            |---models/
                    |---antelope/
                            |--- glintr100.onnx
                            |--- scrfd_10g_bnkps.onnx
                    |---new/
                            |--- det_10g.onnx
```