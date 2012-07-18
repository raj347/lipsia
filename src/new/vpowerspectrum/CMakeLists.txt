PROJECT(vpowerspectrum)

ADD_EXECUTABLE(vpowerspectrum vpowerspectrum.c)
TARGET_LINK_LIBRARIES(vpowerspectrum lipsia ${VIAIO_LIBRARY} ${FFTW3_FFTW3_LIBRARY} ${LIPSIA_LIBRARY})


INSTALL(TARGETS vpowerspectrum
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
