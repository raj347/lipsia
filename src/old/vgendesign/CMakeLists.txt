PROJECT(vgendesign)

ADD_EXECUTABLE(vgendesign vgendesign.c)
TARGET_LINK_LIBRARIES(vgendesign lipsia ${VIA_LIBRARY} ${FFTW3_FFTW3_LIBRARY})

SET_TARGET_PROPERTIES(vgendesign PROPERTIES
                      COMPILE_FLAGS -ansi)

INSTALL(TARGETS vgendesign
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
