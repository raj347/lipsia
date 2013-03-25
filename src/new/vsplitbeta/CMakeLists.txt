PROJECT(vsplitbeta)

ADD_EXECUTABLE(vsplitbeta vsplitbeta.cpp)

TARGET_LINK_LIBRARIES(vsplitbeta ${VIAIO_LIBRARY}  ${LIPSIA_LIBRARY}
    ${GSL_GSL_LIBRARY} ${GSL_CBLAS_LIBRARY})

SET_TARGET_PROPERTIES(vsplitbeta PROPERTIES
                      LINK_FLAGS -Wl)

INSTALL(TARGETS vsplitbeta 
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
