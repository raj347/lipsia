PROJECT(vROIonesample_ttest)

ADD_EXECUTABLE(vROIonesample_ttest vROIonesample_ttest.c)
TARGET_LINK_LIBRARIES(vROIonesample_ttest lipsia ${VIA_LIBRARY})



INSTALL(TARGETS vROIonesample_ttest
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
