PROJECT(vROItwosample_ttest)

ADD_EXECUTABLE(vROItwosample_ttest vROItwosample_ttest.c)
TARGET_LINK_LIBRARIES(vROItwosample_ttest lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vROItwosample_ttest
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
