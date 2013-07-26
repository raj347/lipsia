PROJECT(vdomulticomp)

ADD_EXECUTABLE(vdomulticomp vdomulticomp.c DoMulticomp2d.c DoMulticomp3d.c)
TARGET_LINK_LIBRARIES(vdomulticomp lipsia ${VIA_LIBRARY})

SET_TARGET_PROPERTIES(vdomulticomp PROPERTIES
                      COMPILE_FLAGS -ansi)

INSTALL(TARGETS vdomulticomp
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
