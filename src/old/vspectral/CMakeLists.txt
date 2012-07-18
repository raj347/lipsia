PROJECT(vspectral)

ADD_EXECUTABLE(vspectral vspectral.c)
TARGET_LINK_LIBRARIES(vspectral lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vspectral
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
