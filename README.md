# ZED

#Camera.hpp
```
class SLSTEREO_EXPORT_DLL Pose:
{
sl::Translation getTranslation();//或得平移向量（x,y,z）
sl::Orientation getOrientation();//获取方向向量orientation vector
sl::Rotation getRotation();//获取旋转矩阵（3x3)
sl::Vector3<float> getRotationVector();//获取旋转向量（3x1）
bool valid;
unsigned long long timestamp;//时间戳
sl::Transform pose_data;//变换矩阵（T）4X4
int pose_confidence;//位姿估计置信度
};
```
