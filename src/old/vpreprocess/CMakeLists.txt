PROJECT(vpreprocess)

ADD_EXECUTABLE(vpreprocess vpreprocess.c MinVals.c SpatialFilter.c FreqFilter.c)
TARGET_LINK_LIBRARIES(vpreprocess lipsia ${VIA_LIBRARY} ${FFTW3_FFTW3_LIBRARY})


INSTALL(TARGETS vpreprocess
        RUNTIME DESTINATION bin
                COMPONENT RuntimeLibraries)
