PROJECT(vbrainmask)

ADD_EXECUTABLE(vbrainmask vbrainmask.c)
TARGET_LINK_LIBRARIES(vbrainmask lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vbrainmask
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
