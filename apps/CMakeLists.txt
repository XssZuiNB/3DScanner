# Orbbec camera test / live point cloud viewer
if(GCA_WITH_ORBBEC)
    add_executable(orbbec_viewer orbbec_viewer.cpp)
    target_link_libraries(
        orbbec_viewer
        PRIVATE
        camera
        geometry
        gpu
        opencv_core
        opencv_highgui
        opencv_imgcodecs
        opencv_imgproc
    )
    set_target_properties(orbbec_viewer PROPERTIES
        BUILD_RPATH "${OrbbecSDK_DIR}")
endif()
