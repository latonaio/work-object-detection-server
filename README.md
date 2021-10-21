# work-object-detection-server  
work-object-detection-serverは、主にエッジコンピューティング環境において、ストリーミングされた動画を処理するコアのマイクロサービスです。  

## 動作環境  
work-object-detection-server は、Kubernetes および AION 上での動作を前提としています。   
以下の環境が必要となります。  
・OS: Linux OS  
・CPU: ARM/AMD/Intel  
・Kubernetes  
・AION  

## Generate code
```
python3 -m grpc_tools.protoc \
  -I ./work_object_detection \
  --python_out=./work_object_detection \
  --grpc_python_out=./work_object_detection \
  ./work_object_detection/work_object_detection.proto
```
