PROJECT(vspectralecm)

ADD_EXECUTABLE(vspectralecm vspectralecm.c)
TARGET_LINK_LIBRARIES(vspectralecm  ${VIAIO_LIBRARY} ${LIPSIA_LIBRARY}
    ${GSL_GSL_LIBRARY} ${GSL_CBLAS_LIBRARY})

INSTALL(TARGETS vspectralecm
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
