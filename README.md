# Namespace sl
1. Core.hpp
1. Mesh.hpp
1. define.hpp
1. Camera.hpp
1. type.hpp
----------------------
## Camera.hpp
1. InitParameters
class SL_SDK_EXPORT InitParameters {
*  RESOLUTION camera_resolution;


1. Pose
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
2. class CameraMemberHandler;
```
class SLSTEREO_EXPORT_DLL Camera {
ERROR_CODE open(InitParameters init_parameters = InitParameters());//init_parameters : a structure containing all the individual parameters.一个结构体包含所有初始化参数。

//Opens the ZED camera in the desired mode (live/SVO), sets all the defined parameters, checks hardware requirements and launch internal self calibration.检查硬件要求，启动内部自我标定
//return An error code given informations about the internal process, if sl::ERROR_CODE::SUCCESS is returned, the camera is ready to us.Every other code indicates an error and the program should be stoppe。必须返回success。

inline bool isOpened() {return opene;}
void close();//关闭相机并且释放内存！

ERROR_CODE grab(RuntimeParameters rt_parameters = RuntimeParameters());
//抓图->矫正->计算深度，retrieve函数应当在grab之后使用。

RuntimeParameters(SENSING_MODE sensing_mode_ = SENSING_MODE::SENSING_MODE_STANDARD,
                          bool enable_depth_ = true, bool enable_point_cloud_ = true, REFERENCE_FRAME measure3D_reference_frame_ = REFERENCE_FRAME_CAMERA)
            : sensing_mode(sensing_mode_)
            , enable_depth(enable_depth_)
            , enable_point_cloud(enable_point_cloud_)
            , measure3D_reference_frame(measure3D_reference_frame_) {}
```
