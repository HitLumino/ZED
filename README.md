# Namespace sl
1. Core.hpp
1. Mesh.hpp
1. define.hpp
1. Camera.hpp
1. type.hpp
----------------------
## Camera.hpp
### 1. InitParameters
 _class SL_SDK_EXPORT InitParameters:_
*  RESOLUTION camera_resolution;//默认 RESOLUTION_HD720。
*  int camera_fps;//set 0 表示按照默认。
*  int camera_image_flip;//是否水平翻转，默认false。
*  bool camera_disable_self_calib;//默认false,默认自我标定且能够优化。
*  bool enable_right_side_measure;//默认flase,Defines if right MEASURE should be computed (needed for MEASURE_<XXX>_RIGHT)。
*  int camera_buffer_count_linux;//只有linux有.减少这个值会减少延迟，但是会增加坏帧。默认为4。
*  int camera_linux_id;//这个用于多个ZED，只有linux。
*  sl::String svo_input_filename;//路径去记录SVO文件，默认空。
*  bool svo_real_time_mode;//This mode simulates the live camera and consequently skipped frames if the computation framerate is too slow. default : false
*  DEPTH_MODE depth_mode;//影响深度图的质量。default : sl::DEPTH_MODE::DEPTH_MODE_PERFORMANCE
*  bool depth_stabilization;//深度度是否需要稳定化，需要positional tracking 数据，他会自动使能如果需要的话。
*  float depth_minimum_distance;//根据电脑性能，适当提高。当前默认70cm(-1).
*  UNIT coordinate_units;//定义度量单位（深度/点云/跟踪/网格），默认米  sl::UNIT::UNIT_MILLIMETER。
*  COORDINATE_SYSTEM coordinate_system;//定义坐标系，顺序，轴的方向default : sl::COORDINATE_SYSTEM::COORDINATE_SYSTEM_IMAGE。
*  int sdk_gpu_id;//默认使用gpu(-1)
*  bool sdk_verbose;//提供文字反馈在终端，默认false.
*  sl::String sdk_verbose_log_file;//log
*  bool save(sl::String filename);//保存当前配置文件到文档中，路径名
*  bool load(sl::String filename);//加载配置文件
------------------------------------------------
InitParameters默认构造函数如下
```
InitParameters(RESOLUTION camera_resolution_ = RESOLUTION_HD720,
                       int camera_fps_ = 0,
                       int camera_linux_id_ = 0,
                       sl::String svo_input_filename_ = sl::String(),
                       bool svo_real_time_mode_ = false,
                       DEPTH_MODE depth_mode_ = DEPTH_MODE_PERFORMANCE,
                       UNIT coordinate_units_ = UNIT_MILLIMETER,
                       COORDINATE_SYSTEM coordinate_system_ = COORDINATE_SYSTEM_IMAGE,
                       bool sdk_verbose_ = false,
                       int sdk_gpu_id_ = -1,
                       float depth_minimum_distance_ = -1.,
                       bool camera_disable_self_calib_ = false,
                       bool camera_image_flip_ = false,
                       bool enable_right_side_measure_ = false,
                       int camera_buffer_count_linux_ = 4,
                       sl::String sdk_verbose_log_file_ = sl::String(),
                       bool depth_stabilization_ = true)
            : camera_resolution(camera_resolution_)
            , camera_fps(camera_fps_)
            , camera_linux_id(camera_linux_id_)
            , svo_input_filename(svo_input_filename_)
            , svo_real_time_mode(svo_real_time_mode_)
            , depth_mode(depth_mode_)
            , coordinate_units(coordinate_units_)
            , coordinate_system(coordinate_system_)
            , sdk_verbose(sdk_verbose_)
            , sdk_gpu_id(sdk_gpu_id_)
            , depth_minimum_distance(depth_minimum_distance_)
            , camera_disable_self_calib(camera_disable_self_calib_)
            , camera_image_flip(camera_image_flip_)
            , enable_right_side_measure(enable_right_side_measure_)
            , camera_buffer_count_linux(camera_buffer_count_linux_)
            , sdk_verbose_log_file(sdk_verbose_log_file_)
            , depth_stabilization(depth_stabilization_) {}
```
 
### 2. RuntimeParameters
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
