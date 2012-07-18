PROJECT(vtwosample_ttest)

ADD_EXECUTABLE(vtwosample_ttest vtwosample_ttest.c Gaussianize.c)
TARGET_LINK_LIBRARIES(vtwosample_ttest lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vtwosample_ttest
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
