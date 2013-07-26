PROJECT(vimagetimecourse)

ADD_EXECUTABLE(vimagetimecourse vimagetimecourse.c)
TARGET_LINK_LIBRARIES(vimagetimecourse lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vimagetimecourse
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
