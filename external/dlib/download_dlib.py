"""
external/dlib/download_dlib.py
下載 Dlib 預訓練模型
"""

import os
import urllib.request
import bz2
from pathlib import Path

def download_dlib_models():
    """下載 Dlib 所需的模型檔案"""
    
    # 建立目錄
    models_dir = Path("external/dlib")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 模型 URLs
    models = {
        "shape_predictor_68_face_landmarks.dat.bz2": 
            "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
        "dlib_face_recognition_resnet_model_v1.dat.bz2":
            "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
    }
    
    for filename, url in models.items():
        output_path = models_dir / filename
        extracted_path = models_dir / filename.replace('.bz2', '')
        
        # 檢查是否已存在
        if extracted_path.exists():
            print(f"✓ {extracted_path.name} 已存在")
            continue
        
        # 下載
        print(f"下載 {filename}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"  已下載: {output_path}")
        
        # 解壓縮
        print(f"解壓縮 {filename}...")
        with bz2.BZ2File(output_path) as fr:
            with open(extracted_path, 'wb') as fw:
                fw.write(fr.read())
        
        print(f"  已解壓: {extracted_path}")
        
        # 刪除壓縮檔
        os.remove(output_path)
        print(f"  已清理: {output_path}")
    
    print("\n✓ Dlib 模型下載完成！")
    print(f"模型位置: {models_dir.absolute()}")
    
    # 檔案大小
    for file in models_dir.glob("*.dat"):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name}: {size_mb:.2f} MB")

if __name__ == "__main__":
    download_dlib_models()