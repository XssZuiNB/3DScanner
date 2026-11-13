# Orbbec camera test / live point cloud viewer
if(GCA_WITH_ORBBEC)
    add_executable(orbbec_viewer orbbec_viewer.cpp)
    target_link_libraries(orbbec_viewer PRIVATE gca_lib)
    set_target_properties(orbbec_viewer PROPERTIES
        BUILD_RPATH "${OrbbecSDK_DIR}")
endif()

# Original RealSense + PCL demo (comparison code)
if(GCA_WITH_REALSENSE AND GCA_WITH_PCL)
    add_executable(realsense_demo app.cpp)
    target_include_directories(realsense_demo PRIVATE ${PCL_INCLUDE_DIRS})
    target_link_libraries(realsense_demo PRIVATE gca_lib ${PCL_LIBRARIES} OpenMP::OpenMP_CXX)
endif()
