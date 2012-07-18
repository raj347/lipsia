PROJECT(vdistortionCorrection)

ADD_EXECUTABLE(vdistortionCorrection
               DistortionCorrection.c  QuickMorph2d.c  vdistortionCorrection.c)
TARGET_LINK_LIBRARIES(vdistortionCorrection lipsia ${VIA_LIBRARY})

SET_TARGET_PROPERTIES(vdistortionCorrection PROPERTIES
                      COMPILE_FLAGS -ansi)

INSTALL(TARGETS vdistortionCorrection
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
