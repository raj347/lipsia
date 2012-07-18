PROJECT(vshowpts)

ADD_EXECUTABLE(vshowpts vshowpts.c)
TARGET_LINK_LIBRARIES(vshowpts lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vshowpts
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
