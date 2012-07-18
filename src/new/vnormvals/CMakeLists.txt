PROJECT(vnormvals)

ADD_EXECUTABLE(vnormvals vnormvals.c)
TARGET_LINK_LIBRARIES(vnormvals m lipsia ${VIAIO_LIBRARY} ${LISPIA_LIBRARY})


INSTALL(TARGETS vnormvals
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
