# 啟動順序
(1) 將三個訓練好的model搬入專案資料夾  
(2) 準備好外部引用的模型dlib與topoFR  
(3) topoFR的backbones需替換  
(4) 建立docker  
docker-compose build  
docker-compose up  
(5) https://localhost:8000/docs測試API

### 常用指令 - 建置映像檔
docker-compose build  
docker-compose -f docker-compose.gpu.yml build

### 常用指令 - 清除舊映像和快取
docker builder prune  
docker image prune