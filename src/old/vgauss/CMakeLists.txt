PROJECT(vgauss)

ADD_EXECUTABLE(vgauss vgauss.c)
TARGET_LINK_LIBRARIES(vgauss lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vgauss
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
