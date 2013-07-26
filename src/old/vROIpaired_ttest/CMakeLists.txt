PROJECT(vROIpaired_ttest)

ADD_EXECUTABLE(vROIpaired_ttest vROIpaired_ttest.c)
TARGET_LINK_LIBRARIES(vROIpaired_ttest lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vROIpaired_ttest
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
