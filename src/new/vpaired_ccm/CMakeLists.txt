PROJECT(vpaired_ccm)


ADD_EXECUTABLE(vpaired_ccm vpaired_ccm.c CCC.c)
TARGET_LINK_LIBRARIES(vpaired_ccm ${VIAIO_LIBRARY}  ${LIPSIA_LIBRARY}
    ${GSL_GSL_LIBRARY} ${GSL_CBLAS_LIBRARY})


INSTALL(TARGETS vpaired_ccm
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
