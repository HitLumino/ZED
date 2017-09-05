# ZED
**Camera.hpp**
2
```
3
class SLSTEREO_EXPORT_DLL Pose:
4
{
5
sl::Translation getTranslation();//或得平移向量（x,y,z）
6
sl::Orientation getOrientation();//获取方向向量orientation vector
7
sl::Rotation getRotation();//获取旋转矩阵（3x3)
8
sl::Vector3<float> getRotationVector();//获取旋转向量（3x1）
9
bool valid;
10
unsigned long long timestamp;//时间戳
11
sl::Transform pose_data;//变换矩阵（T）4X4
12
int pose_confidence;//位姿估计置信度
13
};
14
​
15
```
