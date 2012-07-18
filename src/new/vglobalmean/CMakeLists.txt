PROJECT(vglobalmean)

ADD_EXECUTABLE(vglobalmean vglobalmean.c)
TARGET_LINK_LIBRARIES(vglobalmean lipsia ${GSL_GSL_LIBRARY} ${GSL_CBLAS_LIBRARY} ${VIAIO_LIBRARY} ${LISPIA_LIBRARY})


INSTALL(TARGETS vglobalmean
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
