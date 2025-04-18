# Tunnel-LIVO
## run
`roslaunch fast_livo mapping_mid360_basler.launch`
## 核心代码
### lidar
* 主要在 laserMapping.cpp 包含LiDAR ESIKF点到面ICP，退化方向检测
### camera
* 主要在lidar_selection.cpp，detect函数的核心，包了了四个函数
    * addFromSparseMap 从视觉全局地图中选取视觉子图
    * addSparseMap 从当前图像加入点到视觉全局地图
    * ComputeJ ESIKF视觉部分
    * addObservation 更新地图点的当前观测