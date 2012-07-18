PROJECT(vsymmetric)


ADD_EXECUTABLE(vsymmetric vsymmetric.c)
TARGET_LINK_LIBRARIES(vsymmetric lipsia ${VIAIO_LIBRARY} ${GSL_CBLAS_LIBRARY} ${LIPSIA_LIBRARY})

INSTALL(TARGETS vsymmetric
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
