PROJECT(vwhitematter)

ADD_EXECUTABLE(vwhitematter
               LocalContrast.c TopSmooth.c vwhitematter.c Lee3d.c VCheckPlane.c)
TARGET_LINK_LIBRARIES(vwhitematter lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vwhitematter
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
