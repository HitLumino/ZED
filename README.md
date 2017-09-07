# Namespace sl
1. Core.hpp
1. Mesh.hpp
1. define.hpp
1. Camera.hpp
1. type.hpp
----------------------
## Define.hpp
```
namespace sl {

    ///@{
    ///  @name Unavailable Values
    /**
    Defines an unavailable depth value that is above the depth Max value.
    */
    static const float TOO_FAR = INFINITY;
    /**
    Defines an unavailable depth value that is below the depth Min value.
    */
    static const float TOO_CLOSE = -INFINITY;
    /**
    Defines an unavailable depth value that is on an occluded image area.
    */
    static const float OCCLUSION_VALUE = NAN;
    ///@}

    //macro to detect wrong data measure
#define isValidMeasure(v) (std::isfinite(v))

    /// \defgroup Enumerations Public enumerations

    /**
    \enum RESOLUTION
    \ingroup Enumerations
    \brief Represents the available resolution defined in sl::cameraResolution.
    \note Since v1.0, RESOLUTION_VGA mode has been updated to WVGA (from 640*480 to 672*376) and requires a firmware update to function (>= 1142). Firmware can be updated in the ZED Explorer.
    \warning NVIDIA Jetson X1 only supports RESOLUTION_HD1080@15, RESOLUTION_HD720@30/15, and RESOLUTION_VGA@60/30/15.
    */
    enum RESOLUTION {
        RESOLUTION_HD2K, /**< 2208*1242, available framerates: 15 fps.*/
        RESOLUTION_HD1080, /**< 1920*1080, available framerates: 15, 30 fps.*/
        RESOLUTION_HD720, /**< 1280*720, available framerates: 15, 30, 60 fps.*/
        RESOLUTION_VGA, /**< 672*376, available framerates: 15, 30, 60, 100 fps.*/
        RESOLUTION_LAST
    };

    /**
    \enum CAMERA_SETTINGS
    \ingroup Enumerations
    \brief List available camera settings for the ZED camera (contrast, hue, saturation, gain...).
    \brief Each enum defines one of those settings.
    */
    enum CAMERA_SETTINGS {
        CAMERA_SETTINGS_BRIGHTNESS, /**< Defines the brightness control. Affected value should be between 0 and 8.*/
        CAMERA_SETTINGS_CONTRAST, /**< Defines the contrast control. Affected value should be between 0 and 8.*/
        CAMERA_SETTINGS_HUE, /**< Defines the hue control. Affected value should be between 0 and 11.*/
        CAMERA_SETTINGS_SATURATION, /**< Defines the saturation control. Affected value should be between 0 and 8.*/
        CAMERA_SETTINGS_GAIN, /**< Defines the gain control. Affected value should be between 0 and 100 for manual control. If ZED_EXPOSURE is set to -1, the gain is in auto mode too.*/
        CAMERA_SETTINGS_EXPOSURE, /**< Defines the exposure control. A -1 value enable the AutoExposure/AutoGain control,as the boolean parameter (default) does. Affected value should be between 0 and 100 for manual control.*/
        CAMERA_SETTINGS_WHITEBALANCE, /**< Defines the color temperature control. Affected value should be between 2800 and 6500 with a step of 100. A value of -1 set the AWB ( auto white balance), as the boolean parameter (default) does.*/
        CAMERA_SETTINGS_AUTO_WHITEBALANCE, /**< Defines the status of white balance (automatic or manual). A value of 0 disable the AWB, while 1 activate it.*/
        CAMERA_SETTINGS_LAST
    };

    /**
    \enum SELF_CALIBRATION_STATE
    \ingroup Enumerations
    \brief Status for self calibration. Since v0.9.3, self-calibration is done in background and start in the sl::Camera::open or Reset function.
    \brief You can follow the current status for the self-calibration any time once ZED object has been construct.
    */
    enum SELF_CALIBRATION_STATE {
        SELF_CALIBRATION_STATE_NOT_STARTED, /**< Self calibration has not run yet (no sl::Camera::open or sl::Camera::resetSelfCalibration called).*/
        SELF_CALIBRATION_STATE_RUNNING, /**< Self calibration is currently running.*/
        SELF_CALIBRATION_STATE_FAILED, /**< Self calibration has finished running but did not manage to get accurate values. Old parameters are taken instead.*/
        SELF_CALIBRATION_STATE_SUCCESS, /**< Self calibration has finished running and did manage to get accurate values. New parameters are set.*/
        SELF_CALIBRATION_STATE_LAST
    };

    /**
    \enum DEPTH_MODE
    \ingroup Enumerations
    \brief List available depth computation modes.
    */
    enum DEPTH_MODE {
        DEPTH_MODE_NONE, /**< This mode does not compute any depth map. Only rectified stereo images will be available.*/
        DEPTH_MODE_PERFORMANCE, /**< Fastest mode for depth computation.*/
        DEPTH_MODE_MEDIUM, /**< Balanced quality mode. Depth map is robust in any environment and requires medium resources for computation.*/
        DEPTH_MODE_QUALITY, /**< Best quality mode. Requires more compute power.*/
        DEPTH_MODE_LAST
    };

    /**
    \enum SENSING_MODE
    \ingroup Enumerations
    \brief List available depth sensing modes.
    */
    enum SENSING_MODE {
        SENSING_MODE_STANDARD, /**< This mode outputs ZED standard depth map that preserves edges and depth accuracy.
                               * Applications example: Obstacle detection, Automated navigation, People detection, 3D reconstruction.*/
        SENSING_MODE_FILL, /**< This mode outputs a smooth and fully dense depth map.
                           * Applications example: AR/VR, Mixed-reality capture, Image post-processing.*/
        SENSING_MODE_LAST
    };

    /**
    \enum UNIT
    \ingroup Enumerations
    \brief List available unit for measures.
    */
    enum UNIT {
        UNIT_MILLIMETER, /**< International System, 1/1000 METER. */
        UNIT_CENTIMETER, /**< International System, 1/100 METER. */
        UNIT_METER, /**< International System, 1 METER */
        UNIT_INCH, /**< Imperial Unit, 1/12 FOOT */
        UNIT_FOOT, /**< Imperial Unit, 1 FOOT */
        UNIT_LAST
    };

    /**
    \enum COORDINATE_SYSTEM
    \ingroup Enumerations
    \brief List available coordinates systems for positional tracking and 3D measures.
    */
    enum COORDINATE_SYSTEM {
        COORDINATE_SYSTEM_IMAGE, /**< Standard coordinates system in computer vision. Used in OpenCV : see here : http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html */
        COORDINATE_SYSTEM_LEFT_HANDED_Y_UP, /**< Left-Handed with Y up and Z forward. Used in Unity with DirectX. */
        COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP, /**< Right-Handed with Y pointing up and Z backward. Used in OpenGL. */
        COORDINATE_SYSTEM_RIGHT_HANDED_Z_UP, /**< Right-Handed with Z pointing up and Y forward. Used in 3DSMax. */
        COORDINATE_SYSTEM_LEFT_HANDED_Z_UP, /**< Left-Handed with Z axis pointing up and X forward. Used in Unreal Engine. */
        COORDINATE_SYSTEM_LAST
    };

    /**
    \enum MEASURE
    \ingroup Enumerations
    \brief List retrievable measures.
    */
    enum MEASURE {
        MEASURE_DISPARITY, /**< Disparity map, sl::MAT_TYPE_32F_C1.*/
        MEASURE_DEPTH, /**< Depth map, sl::MAT_TYPE_32F_C1.*/
        MEASURE_CONFIDENCE, /**< Certainty/confidence of the disparity map, sl::MAT_TYPE_32F_C1.*/
        MEASURE_XYZ, /**< Point cloud, sl::MAT_TYPE_32F_C4, channel 4 is empty.*/
        MEASURE_XYZRGBA, /**< Colored point cloud,  sl::MAT_TYPE_32F_C4, channel 4 contains color in R-G-B-A order.*/
        MEASURE_XYZBGRA, /**< Colored point cloud,  sl::MAT_TYPE_32F_C4, channel 4 contains color in B-G-R-A order.*/
        MEASURE_XYZARGB, /**< Colored point cloud,  sl::MAT_TYPE_32F_C4, channel 4 contains color in A-R-G-B order.*/
        MEASURE_XYZABGR, /**< Colored point cloud,  sl::MAT_TYPE_32F_C4, channel 4 contains color in A-B-G-R order.*/
        MEASURE_NORMALS, /**< Normals vector,  sl::MAT_TYPE_32F_C4, channel 4 is empty (set to 0)*/
        MEASURE_DISPARITY_RIGHT, /**< Disparity map for right sensor, sl::MAT_TYPE_32F_C1.*/
        MEASURE_DEPTH_RIGHT, /**< Depth map for right sensor, sl::MAT_TYPE_32F_C1.*/
        MEASURE_XYZ_RIGHT, /**< Point cloud for right sensor, sl::MAT_TYPE_32F_C1, channel 4 is empty.*/
        MEASURE_XYZRGBA_RIGHT, /**< Colored point cloud for right sensor, sl::MAT_TYPE_32F_C4, channel 4 contains color in R-G-B-A order.*/
        MEASURE_XYZBGRA_RIGHT, /**< Colored point cloud for right sensor, sl::MAT_TYPE_32F_C4, channel 4 contains color in B-G-R-A order.*/
        MEASURE_XYZARGB_RIGHT, /**< Colored point cloud for right sensor, sl::MAT_TYPE_32F_C4, channel 4 contains color in A-R-G-B order.*/
        MEASURE_XYZABGR_RIGHT, /**< Colored point cloud for right sensor, sl::MAT_TYPE_32F_C4, channel 4 contains color in A-B-G-R order.*/
        MEASURE_NORMALS_RIGHT, /**< Normals vector for right view, sl::MAT_TYPE_32F_C4, channel 4 is empty (set to 0)*/
        MEASURE_LAST
    };

    /**
    \enum VIEW
    \ingroup Enumerations
    \brief List available views.
    */
    enum VIEW {
        VIEW_LEFT, /**< Left RGBA image, sl::MAT_TYPE_8U_C4. */
        VIEW_RIGHT, /**< Right RGBA image, sl::MAT_TYPE_8U_C4. */
        VIEW_LEFT_GRAY, /**< Left GRAY image, sl::MAT_TYPE_8U_C1. */
        VIEW_RIGHT_GRAY, /**< Right GRAY image, sl::MAT_TYPE_8U_C1. */
        VIEW_LEFT_UNRECTIFIED, /**< Left RGBA unrectified image, sl::MAT_TYPE_8U_C4. */
        VIEW_RIGHT_UNRECTIFIED,/**< Right RGBA unrectified image, sl::MAT_TYPE_8U_C4. */
        VIEW_LEFT_UNRECTIFIED_GRAY, /**< Left GRAY unrectified image, sl::MAT_TYPE_8U_C1. */
        VIEW_RIGHT_UNRECTIFIED_GRAY,/**< Right GRAY unrectified image, sl::MAT_TYPE_8U_C1. */
        VIEW_SIDE_BY_SIDE, /**< Left and right image (the image width is therefore doubled). RGBA image, sl::MAT_TYPE_8U_C4. */
        VIEW_DEPTH, /**< Color rendering of the depth, sl::MAT_TYPE_8U_C4. */
        VIEW_CONFIDENCE, /**< Color rendering of the depth confidence, sl::MAT_TYPE_8U_C4. */
        VIEW_NORMALS, /**< Color rendering of the normals, sl::MAT_TYPE_8U_C4. */
        VIEW_DEPTH_RIGHT, /**< Color rendering of the right depth mapped on right sensor, sl::MAT_TYPE_8U_C4. */
        VIEW_NORMALS_RIGHT, /**< Color rendering of the normals mapped on right sensor, sl::MAT_TYPE_8U_C4. */
        VIEW_LAST
    };

    /**
    \enum DEPTH_FORMAT
    \ingroup Enumerations
    \brief List available file formats for saving depth maps.
    */
    enum DEPTH_FORMAT {
        DEPTH_FORMAT_PNG, /**< PNG image format in 16bits. 32bits depth is mapped to 16bits color image to preserve the consistency of the data range.*/
        DEPTH_FORMAT_PFM, /**< stream of bytes, graphic image file format.*/
        DEPTH_FORMAT_PGM, /**< gray-scale image format.*/
        DEPTH_FORMAT_LAST
    };

    /**
    \enum POINT_CLOUD_FORMAT
    \ingroup Enumerations
    \brief List available file formats for saving point clouds. Stores the spatial coordinates (x,y,z) of each pixel and optionally its RGB color.
    */
    enum POINT_CLOUD_FORMAT {
        POINT_CLOUD_FORMAT_XYZ_ASCII, /**< Generic point cloud file format, without color information.*/
        POINT_CLOUD_FORMAT_PCD_ASCII, /**< Point Cloud Data file, with color information.*/
        POINT_CLOUD_FORMAT_PLY_ASCII, /**< PoLYgon file format, with color information.*/
        POINT_CLOUD_FORMAT_VTK_ASCII, /**< Visualization ToolKit file, without color information.*/
        POINT_CLOUD_FORMAT_LAST
    };

    /**
    \enum TRACKING_STATE
    \ingroup Enumerations
    \brief List the different states of positional tracking.
    */
    enum TRACKING_STATE {
        TRACKING_STATE_SEARCHING, /**< The camera is searching for a previously known position to locate itself.*/
        TRACKING_STATE_OK, /**< Positional tracking is working normally.*/
        TRACKING_STATE_OFF, /**< Positional tracking is not enabled.*/
        TRACKING_STATE_FPS_TOO_LOW, /**< Effective FPS is too low to give proper results for motion tracking. Consider using PERFORMANCES parameters (DEPTH_MODE_PERFORMANCE, low camera resolution (VGA,HD720))*/
        TRACKING_STATE_LAST
    };

    /**
    \enum AREA_EXPORT_STATE
    \ingroup Enumerations
    \brief List the different states of spatial memory area export.
    */
    enum AREA_EXPORT_STATE {
        AREA_EXPORT_STATE_SUCCESS, /**< The spatial memory file has been successfully created.*/
        AREA_EXPORT_STATE_RUNNING, /**< The spatial memory is currently written.*/
        AREA_EXPORT_STATE_NOT_STARTED, /**< The spatial memory file exportation has not been called.*/
        AREA_EXPORT_STATE_FILE_EMPTY, /**< The spatial memory contains no data, the file is empty.*/
        AREA_EXPORT_STATE_FILE_ERROR, /**< The spatial memory file has not been written because of a wrong file name.*/
        AREA_EXPORT_STATE_SPATIAL_MEMORY_DISABLED, /**< The spatial memory learning is disable, no file can be created.*/
        AREA_EXPORT_STATE_LAST
    };

    /**
    \enum REFERENCE_FRAME
    \ingroup Enumerations
    \brief Define which type of position matrix is used to store camera path and pose.
    */
    enum REFERENCE_FRAME {
        REFERENCE_FRAME_WORLD, /**< The transform of sl::Pose will contains the motion with reference to the world frame (previously called PATH).*/
        REFERENCE_FRAME_CAMERA, /**< The transform of sl::Pose will contains the motion with reference to the previous camera frame (previously called POSE).*/
        REFERENCE_FRAME_LAST
    };

    /**
    \enum SPATIAL_MAPPING_STATE
    \ingroup Enumerations
    \brief Gives the spatial mapping state.
    */
    enum SPATIAL_MAPPING_STATE {
        SPATIAL_MAPPING_STATE_INITIALIZING, /**< The spatial mapping is initializing.*/
        SPATIAL_MAPPING_STATE_OK, /**< The depth and tracking data were correctly integrated in the fusion algorithm.*/
        SPATIAL_MAPPING_STATE_NOT_ENOUGH_MEMORY, /**< The maximum memory dedicated to the scanning has been reach, the mesh will no longer be updated.*/
        SPATIAL_MAPPING_STATE_NOT_ENABLED, /**< Camera::enableSpatialMapping() wasn't called (or the scanning was stopped and not relaunched).*/
        SPATIAL_MAPPING_STATE_FPS_TOO_LOW, /**< Effective FPS is too low to give proper results for spatial mapping. Consider using PERFORMANCES parameters (DEPTH_MODE_PERFORMANCE, low camera resolution (VGA,HD720), spatial mapping low resolution)*/
        SPATIAL_MAPPING_STATE_LAST
    };

    /**
    \enum SVO_COMPRESSION_MODE
    \ingroup Enumerations
    \brief List available compression modes for SVO recording.
    \brief sl::SVO_COMPRESSION_MODE_LOSSLESS is an improvement of previous lossless compression (used in ZED Explorer), even if size may be bigger, compression time is much faster.
    */
    enum SVO_COMPRESSION_MODE {
        SVO_COMPRESSION_MODE_RAW, /**< RAW images, no compression.*/
        SVO_COMPRESSION_MODE_LOSSLESS, /**< new Lossless, with PNG/ZSTD based compression : avg size = 42% of RAW).*/
        SVO_COMPRESSION_MODE_LOSSY, /**< new Lossy, with JPEG based compression : avg size = 22% of RAW).*/
        SVO_COMPRESSION_MODE_LAST
    };

    /**
    \struct RecordingState
    \brief Recording structure that contains information about SVO.
    */
    struct RecordingState {
        bool status; /**< status of current frame. May be true for success or false if frame could not be written in the SVO file.*/
        double current_compression_time; /**< compression time for the current frame in ms.*/
        double current_compression_ratio; /**< compression ratio (% of raw size) for the current frame.*/
        double average_compression_time; /**< average compression time in ms since beginning of recording.*/
        double average_compression_ratio; /**< compression ratio (% of raw size) since beginning of recording.*/
    };

    ///@{
    ///  @name ZED Camera Resolution
    /**
    Available video modes for the ZED camera.
    */
    static const std::vector<std::pair<int, int>> cameraResolution = {
        std::make_pair(2208, 1242), /**< sl::RESOLUTION_HD2K */
        std::make_pair(1920, 1080), /**< sl::RESOLUTION_HD1080 */
        std::make_pair(1280, 720), /**< sl::RESOLUTION_HD720 */
        std::make_pair(672, 376) /**< sl::RESOLUTION_VGA */
    };
    ///@}

    ///////////////////////////////////////////////////////////////////////////////////////////////////////

    ///@{
    ///  @name Enumeration conversion
    /*!
    \ingroup Functions
    \brief Converts the given RESOLUTION into a string
    \param res : a specific RESOLUTION
    \return The corresponding string
    */
    static inline std::string resolution2str(RESOLUTION res) {
        std::string output;
        switch (res) {
            case RESOLUTION::RESOLUTION_HD2K:
            output = "HD2K";
            break;
            case RESOLUTION::RESOLUTION_HD1080:
            output = "HD1080";
            break;
            case RESOLUTION::RESOLUTION_HD720:
            output = "HD720";
            break;
            case RESOLUTION::RESOLUTION_VGA:
            output = "VGA";
            break;
            case RESOLUTION::RESOLUTION_LAST:
            output = "Unknown";
            break;
            default:
            output = "Unknown";
            break;
        }
        return output;
    }

    /*!
    \ingroup Functions
    \brief Converts the given SELF_CALIBRATION_STATE into a string
    \param state : a specific SELF_CALIBRATION_STATE
    \return The corresponding string
    */
    static inline std::string statusCode2str(SELF_CALIBRATION_STATE state) {
        std::string output;
        switch (state) {
            case SELF_CALIBRATION_STATE::SELF_CALIBRATION_STATE_NOT_STARTED:
            output = "Self calibration:  Not Started";
            break;
            case SELF_CALIBRATION_STATE::SELF_CALIBRATION_STATE_RUNNING:
            output = "Self calibration:  Running";
            break;
            case SELF_CALIBRATION_STATE::SELF_CALIBRATION_STATE_FAILED:
            output = "Self calibration:  Failed";
            break;
            case SELF_CALIBRATION_STATE::SELF_CALIBRATION_STATE_SUCCESS:
            output = "Self calibration:  Success";
            break;
            case SELF_CALIBRATION_STATE::SELF_CALIBRATION_STATE_LAST:
            output = "Unknown";
            break;
            default:
            output = "Unknown";
            break;
        }
        return output;
    }

    /*!
    \ingroup Functions
    \brief Converts the given string into a DEPTH_MODE
    \param mode : a specific depth
    \return The corresponding DEPTH_MODE
    */
    static inline DEPTH_MODE str2mode(std::string mode) {
        DEPTH_MODE output = DEPTH_MODE_PERFORMANCE;
        if (!mode.compare("None"))
            output = DEPTH_MODE_NONE;
        if (!mode.compare("Performance"))
            output = DEPTH_MODE_PERFORMANCE;
        if (!mode.compare("Medium"))
            output = DEPTH_MODE_MEDIUM;
        if (!mode.compare("Quality"))
            output = DEPTH_MODE_QUALITY;
        return output;
    }

    /*!
    \ingroup Functions
    \brief Converts the given DEPTH_MODE into a string
    \param mode : a specific DEPTH_MODE
    \return The corresponding string
    */
    static inline std::string depthMode2str(DEPTH_MODE mode) {
        std::string output;
        switch (mode) {
            case DEPTH_MODE::DEPTH_MODE_NONE:
            output = "None";
            break;
            case DEPTH_MODE::DEPTH_MODE_PERFORMANCE:
            output = "Performance";
            break;
            case DEPTH_MODE::DEPTH_MODE_MEDIUM:
            output = "Medium";
            break;
            case DEPTH_MODE::DEPTH_MODE_QUALITY:
            output = "Quality";
            break;
            case DEPTH_MODE::DEPTH_MODE_LAST:
            output = "Unknown";
            break;
            default:
            output = "Unknown";
            break;
        }
        return output;
    }

    /*!
    \ingroup Functions
    \brief Converts the given SENSING_MODE into a string
    \param mode : a specific SENSING_MODE
    \return The corresponding string
    */
    static inline std::string sensingMode2str(SENSING_MODE mode) {
        std::string output;
        switch (mode) {
            case SENSING_MODE::SENSING_MODE_STANDARD:
            output = "Standard";
            break;
            case SENSING_MODE::SENSING_MODE_FILL:
            output = "Fill";
            break;
            case SENSING_MODE::SENSING_MODE_LAST:
            output = "Unknown";
            break;
            default:
            output = "Unknown";
            break;
        }
        return output;
    }

    /*!
    \ingroup Functions
    \brief Converts the given UNIT into a string
    \param unit : a specific UNIT
    \return The corresponding string
    */
    static inline std::string unit2str(UNIT unit) {
        std::string output;
        switch (unit) {
            case UNIT::UNIT_MILLIMETER:
            output = "Millimeter";
            break;
            case UNIT::UNIT_CENTIMETER:
            output = "Centimeter";
            break;
            case UNIT::UNIT_METER:
            output = "Meter";
            break;
            case UNIT::UNIT_INCH:
            output = "Inch";
            break;
            case UNIT::UNIT_FOOT:
            output = "Feet";
            break;
            case UNIT::UNIT_LAST:
            output = "Unknown";
            break;
            default:
            output = "Unknown";
            break;
        }
        return output;
    }

    /*!
    \ingroup Functions
    \brief Converts the given string into a UNIT
    \param unit : a specific unit string
    \return The corresponding UNIT
    */
    static inline UNIT str2unit(std::string unit) {
        UNIT output = UNIT_MILLIMETER;
        if (!unit.compare("Millimeter"))
            output = UNIT_MILLIMETER;
        if (!unit.compare("Centimeter"))
            output = UNIT_CENTIMETER;
        if (!unit.compare("Meter"))
            output = UNIT_METER;
        if (!unit.compare("Inch"))
            output = UNIT_INCH;
        if (!unit.compare("Feet"))
            output = UNIT_FOOT;
        return output;
    }

    /*!
    \ingroup Functions
    \brief Converts the given TRACKING_STATE into a string
    \param state : a specific TRACKING_STATE
    \return The corresponding string
    */
    static inline std::string trackingState2str(TRACKING_STATE state) {
        std::string output;
        switch (state) {
            case TRACKING_STATE_SEARCHING:
            output = "Tracking state: Searching";
            break;
            case TRACKING_STATE_OK:
            output = "Tracking state: OK";
            break;
            case TRACKING_STATE_OFF:
            output = "Tracking state: OFF";
            break;
            case TRACKING_STATE_FPS_TOO_LOW:
            output = "Tracking state: FPS too low";
            break;
            case TRACKING_STATE_LAST:
            output = "Unknown";
            break;
            default:
            output = "Unknown";
            break;
        }
        return output;
    }

    /*!
    \ingroup Functions
    \brief Converts the given SPATIAL_MAPPING_STATE into a string
    \param state : a specific SPATIAL_MAPPING_STATE
    \return The corresponding string
    */
    static inline std::string spatialMappingState2str(SPATIAL_MAPPING_STATE state) {
        std::string output;
        switch (state) {
            case SPATIAL_MAPPING_STATE_INITIALIZING:
            output = "Spatial Mapping state: Initializing";
            break;
            case SPATIAL_MAPPING_STATE_OK:
            output = "Spatial Mapping state: OK";
            break;
            case SPATIAL_MAPPING_STATE_NOT_ENOUGH_MEMORY:
            output = "Spatial Mapping state: Not Enough Memory";
            break;
            case SPATIAL_MAPPING_STATE_NOT_ENABLED:
            output = "Spatial Mapping state: Not Enabled";
            break;
            case SPATIAL_MAPPING_STATE_FPS_TOO_LOW:
            output = "Spatial Mapping state: FPS too low";
            break;
            case SPATIAL_MAPPING_STATE_LAST:
            output = "Unknown";
            break;
            default:
            output = "Unknown";
            break;
        }
        return output;
    }
}
```
## Camera.hpp
### 1. InitParameters
 ___class SL_SDK_EXPORT InitParameters:___

*  RESOLUTION camera_resolution;//默认 RESOLUTION_HD720。
*  int camera_fps;//set 0 表示按照默认。
*  int camera_image_flip;//是否水平翻转，默认false。
*  bool camera_disable_self_calib;//默认false,默认自我标定且能够优化。
*  bool enable_right_side_measure;//默认flase,Defines if right MEASURE should be computed (needed for MEASURE_<XXX>_RIGHT)。
*  __int camera_buffer_count_linux;__//只有linux有.减少这个值会减少延迟，但是会增加坏帧。默认为4。
*  int camera_linux_id;//这个用于多个ZED，只有linux。
*  sl::String svo_input_filename;//路径去记录SVO文件，默认空。
*  bool svo_real_time_mode;//This mode simulates the live camera and consequently skipped frames if the computation framerate is too slow. default : false
*  DEPTH_MODE depth_mode;//影响深度图的质量。default : sl::DEPTH_MODE::DEPTH_MODE_PERFORMANCE
*  bool depth_stabilization;//深度度是否需要稳定化，需要positional tracking 数据，他会自动使能如果需要的话。
*  float depth_minimum_distance;//根据电脑性能，适当提高。当前默认70cm(-1).
*  UNIT coordinate_units;//定义度量单位（深度/点云/跟踪/网格），默认米  sl::UNIT::UNIT_MILLIMETER。
*  COORDINATE_SYSTEM coordinate_system;//定义坐标系，顺序，轴的方向default : sl::COORDINATE_SYSTEM::COORDINATE_SYSTEM_IMAGE。
*  `int sdk_gpu_id`;//默认使用gpu(-1)
*  bool sdk_verbose;//提供文字反馈在终端，默认false.
*  sl::String sdk_verbose_log_file;//log
*  bool save(sl::String filename);//保存当前配置文件到文档中，路径名
*  bool load(sl::String filename);//加载配置文件
------------------------------------------------
**InitParameters默认构造函数如下:**
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
-----------------------------------------------

### 2. RuntimeParameter
___class SL_SDK_EXPORT RuntimeParameters___
*  SENSING_MODE sensing_mode;//定义深度地图计算方法，更多的参考sl::SENSING_MODE definition。默认：__sl::SENSING_MODE::SENSING_MODE_STANDARD__
*  REFERENCE_FRAME measure3D_reference_frame;//提供3D测量（点云）的参考坐标系，默认是相机坐标系default is REFERENCE_FRAME_CAMERA
*  bool enable_depth;//是否需要计算深度图，默认true。否则只有图像信息
*  bool enable_point_cloud;//默认计算点云，(including XYZRGBA)目前sdk2.1没有这个选项。
*  bool save(sl::String filename);//保存配置文件
*  bool load(sl::String filename);//加载配置文件
```
RuntimeParameters(SENSING_MODE sensing_mode_ = SENSING_MODE::**SENSING_MODE_STANDARD**,
                          bool enable_depth_ = **true**, bool enable_point_cloud_ = **true**, REFERENCE_FRAME measure3D_reference_frame_ = **REFERENCE_FRAME_CAMERA**)
            : sensing_mode(sensing_mode_)
            , enable_depth(enable_depth_)
            , enable_point_cloud(enable_point_cloud_)
            , measure3D_reference_frame(measure3D_reference_frame_) {}
```
------------------------------------------------
### 3. TrackingParameters
**class SL_SDK_EXPORT TrackingParameters**
*  sl::Transform initial_world_transform;//相机一开始运行时的在世界坐标系下的位置默认认为单位矩阵
*  bool enable_spatial_memory;//使相机能够学习和记住他的环境，有利于纠正运动漂移，和位置。这需要一些资源去跑，但能够有效的改善跟踪精度。建议打开
*  sl::String area_file_path;//Area localization mode 可以记录和加载一个描述环境的文件
        `\note Loading an area file will start a searching phase during which the camera will try to position itself in the previously learned area.
        \warning : The area file describes a specific location. If you are using an area file describing a different location, the tracking function will continuously search for a position and may not find a correct one.
        \warning The '.area' file can only be used with the same depth mode (sl::MODE) as the one used during area recording.`
*  bool save(sl::String filename);
*  bool load(sl::String filename);
```
TrackingParameters(sl::Transform init_pos = sl::Transform(), bool _enable_memory = true, sl::String _area_path = sl::String())
            : initial_world_transform(init_pos)
            , enable_spatial_memory(_enable_memory)
            , area_file_path(_area_path) {}
```
-------------------------------------------------
### 4. SpatialMappingParameters
*  typedef std::pair<float, float> interval;
*  enum RESOLUTION {  
            1. RESOLUTION_HIGH//0.02米      
            2. RESOLUTION_MEDIUM//0.05米，默认   
            3. RESOLUTION_LOW//0.08米   
        };
*  enum RANGE {  
            RANGE_NEAR, // Only depth close to the camera will be used by the spatial mapping.3.5米 \n 
            RANGE_MEDIUM, //Medium depth range 0.5米 默认选项 \n
            RANGE_FAR //useful outdoor.10米   \n
        }；
```
void set(RESOLUTION resolution = RESOLUTION_HIGH) {   //设置函数默认选项是default : RESOLUTION_HIGH
            resolution_meter = get(resolution);   
        }
void set(RANGE range = RANGE_MEDIUM) {
            range_meter.second = get(range);
        }
```
*  float resolution_meter = 0.03f;//Spatial mapping resolution in meters
*  const interval allowed_resolution = std::make_pair(0.01f, 0.2f);//mapping允许的分辨率
*  interval range_meter = std::make_pair(0.7f, 5.f);//0.7~5米 range_meter.first（min)  range_meter.second（max）
*  const interval allowed_min = std::make_pair(0.3f, 10.f);//允许的最小深度取值范围
*  const interval allowed_max = std::make_pair(2.f, 20.f);//允许的大深度取值范围
*  bool save_texture = true;//纹理
*  bool keep_mesh_consistent = true;//使网格连续
*  int max_memory_usage = 2048;//最大允许CPU内存2048M
*  bool inverse_triangle_vertices_order = false;
*  bool save(sl::String filename);
*  bool load(sl::String filename);

**SpatialMappingParameters构造函数
```
SpatialMappingParameters(RESOLUTION resolution = RESOLUTION_HIGH,
                                 RANGE range = RANGE_MEDIUM,
                                 int max_memory_usage_ = 2048,
                                 bool save_texture_ = true,
                                 bool keep_mesh_consistent_ = true,
                                 bool inverse_triangle_vertices_order_ = false) {
            max_memory_usage = max_memory_usage_;
            save_texture = save_texture_;
            keep_mesh_consistent = keep_mesh_consistent_;
            inverse_triangle_vertices_order = inverse_triangle_vertices_order_;
            set(resolution);
            set(range);
```
------------------------------------------------------------------
### 5. Pose
**class SL_SDK_EXPORT Pose**

*  Pose(const sl::Transform &pose_data, unsigned long long mtimestamp = 0, int mconfidence = 0);//默认构造函数
*  sl::Translation getTranslation();//或得平移向量（x,y,z）
*  sl::Orientation getOrientation();//获取方向向量orientation vector
*  sl::Rotation getRotationMatrix();//获取旋转矩阵（3x3)
*  inline sl::Rotation getRotation() return getRotationMatrix()//弃用
*  sl::float3 getRotationVector();//获取旋转向量（3x1）
*  sl::float3 getEulerAngles(bool radian = true);//欧拉角，默认不是角度制。sl::float3 (x=roll, y=pitch, z=yaw)
*  unsigned long long timestamp;//时间戳
*  sl::Transform pose_data;//变换矩阵（T）4X4
*  int pose_confidence;//位姿估计置信度
*  bool valid;//boolean that indicates if tracking is activated or not
-------------------------------------------------------------------
### 6. class CameraMemberHandler
**class SL_SDK_EXPORT Camera**   
       `friend CameraMemberHandler;`     
`public:`
*  Camera();
*  ERROR_CODE open(InitParameters init_parameters = InitParameters());//init_parameters : a structure containing all the individual parameters.一个结构体包含所有初始化参数。//Opens the ZED camera in the desired mode (live/SVO), sets all the defined parameters, checks hardware requirements and launch internal self calibration.检查硬件要求，启动内部自我标定
//return An error code given informations about the internal process, if sl::ERROR_CODE::SUCCESS is returned, the camera is ready to us.Every other code indicates an error and the program should be stoppe。必须返回success。

*  inline bool isOpened() {return opene;}
void close();//关闭相机并且释放内存！

*  ERROR_CODE grab(RuntimeParameters rt_parameters = RuntimeParameters());
//抓图->矫正->计算深度，retrieve函数应当在grab之后使用。
*  ERROR_CODE retrieveImage(Mat &mat, VIEW view = VIEW_LEFT, MEM type = MEM_CPU, int width = 0, int height = 0);//默认左眼图像，内存类型为CPU。MEM_CPU=1,MEM_GPU=2。
*  Resolution getResolution();//获取图像分辨率
*  CUcontext getCUDAContext();//The CUDA context created by the inner process.
*  CameraInformation getCameraInformation(Resolution resizer = Resolution(0, 0));//返回相机内参/外参 calibration_parameters。
*  int getCameraSettings(CAMERA_SETTINGS setting);//相机设置，比如饱和度/亮度/曝光等 -1表示有错误
*  void setCameraSettings(**CAMERA_SETTINGS settings**, int value, bool use_default = false);//value 是相应的控制模式
*  float getCameraFPS();//-1表有错误
*  void setCameraFPS(int desired_fps);
*  float getCurrentFPS();//获取当前帧率，比如回调函数
*  timeStamp getCameraTimestamp();
*  timeStamp getCurrentTimestamp();
*  unsigned int getFrameDroppedCount();//返回丢失的帧数
*  int getSVOPosition();//返回当前SVO文件位置，只用于调用SVO文件时
*  void setSVOPosition(int frame_number);//设置SV0的位置，参数是所希望的帧数
*  int getSVONumberOfFrames();//返回当前文件里的帧数

**有关深度函数：**
*  ERROR_CODE retrieveMeasure(Mat &mat, MEASURE measure = MEASURE_DEPTH, MEM type = MEM_CPU, int width = 0, int height = 0);//取回数据函数，默认深度信息，默认CPU内存占用。
*  float getDepthMaxRangeValue();//返回可以计算最大深度值
*  void setDepthMaxRangeValue(float depth_max_range);//Sets the maximum distance of depth/disparity estimation 
*  float getDepthMinRangeValue();
*  int getConfidenceThreshold();//自信度阈值，0-100
*  void setConfidenceThreshold(int conf_threshold_value);

**Positional Tracking functions:**
*  ERROR_CODE enableTracking(TrackingParameters tracking_parameters = TrackingParameters());//开始跟踪
*  sl::TRACKING_STATE getPosition(**sl::Pose &camera_pose**, REFERENCE_FRAME reference_frame = sl::REFERENCE_FRAME_WORLD);//返回跟踪状态；The camera frame在左眼后面；camera_pose [out]：包含相机位置以及时间戳和自信度；reference_frame :默认世界坐标系
*  sl::AREA_EXPORT_STATE getAreaExportState();//Returns the state of the spatial memory export process
*  void disableTracking(sl::String area_file_path = "");
*  ERROR_CODE resetTracking(sl::Transform &path);//重置，重新初始化路径，和变换矩阵。UI有个按钮。

**Spatial Mapping functions:**
The spatial mapping will create a geometric representation of the scene based on both tracking data and 3D point clouds.
*  ERROR_CODE enableSpatialMapping(SpatialMappingParameters spatial_mapping_parameters = SpatialMappingParameters());
*  SPATIAL_MAPPING_STATE getSpatialMappingState();//The current state of the spatial mapping process
*  void requestMeshAsync();//
*  ERROR_CODE getMeshRequestStatusAsync();
*  ERROR_CODE retrieveMeshAsync(sl::Mesh &mesh);//Retrieves the generated mesh after calling requestMeshAsync.输出网格
*  ERROR_CODE extractWholeMesh(sl::Mesh &mesh);
*  void pauseSpatialMapping(bool status);
*  void disableSpatialMapping();

**Self calibration :**
*  SELF_CALIBRATION_STATE getSelfCalibrationState();//返回自标定状态，有未开始/已经运行/结束但未处理好数据，用的上一次数据/成功
*  void resetSelfCalibration();//It will reset and calculate again correction for misalignment, convergence and color mismatch.这将重置并重新计算校正偏差，收敛和颜色不匹配。

**Specific When Recording mode is activated**
*  ERROR_CODE enableRecording(sl::String video_filename, SVO_COMPRESSION_MODE compression_mode = SVO_COMPRESSION_MODE_LOSSLESS);// video_filename : can be a *.svo* file or a *.avi* file;compression_mode : can be one of the sl::SVO_COMPRESSION_MODE enum
*  sl::RecordingState record();
*  void disableRecording();
**(static)**
*  static sl::String getSDKVersion();
*  static int isZEDconnected();
*  static sl::ERROR_CODE sticktoCPUCore(int cpu_core);//only jetson
```
private:
        ERROR_CODE openCamera(bool);
        bool nextImage(bool);
        int initMemory();
        bool initRectifier();
        CameraMemberHandler *h = 0;
        bool opened = false;
```
*  SL_SDK_EXPORT bool saveDepthAs(sl::Camera &zed, sl::DEPTH_FORMAT format, sl::String name, float factor = 1.);//zed : the current camera object.
*  SL_SDK_EXPORT bool saveDepthAs(sl::Mat &depth, sl::DEPTH_FORMAT format, sl::String name, float factor = 1.);//depth : the depth map to record (CPU 32F_C1 sl::Mat)
*  SL_SDK_EXPORT bool savePointCloudAs(sl::Camera &zed, sl::POINT_CLOUD_FORMAT format, sl::String name, bool with_color = false, bool keep_occluded_point = false);//cloud : the point cloud to record (CPU 32F_C4 sl::Mat)
*  SL_SDK_EXPORT bool savePointCloudAs(sl::Mat &cloud, sl::POINT_CLOUD_FORMAT format, sl::String name, bool with_color = false, bool keep_occluded_point = false);
`#endif /* __CAMERA_HPP__ */`








