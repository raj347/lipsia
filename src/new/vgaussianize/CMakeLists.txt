PROJECT(vgaussianize)

ADD_EXECUTABLE(vgaussianize vgaussianize.c)
TARGET_LINK_LIBRARIES(vgaussianize lipsia ${GSL_GSL_LIBRARY} ${GSL_CBLAS_LIBRARY} ${VIAIO_LIBRARY} ${LISPIA_LIBRARY})


INSTALL(TARGETS vgaussianize
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
