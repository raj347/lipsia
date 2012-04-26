PROJECT(vsvm)

ADD_EXECUTABLE(vsvm vsvm.cpp MriSvm.cpp SearchLight.cpp libsvm-dense/svm.o)
TARGET_LINK_LIBRARIES(vsvm ${VIAIO_LIBRARY}  ${LIPSIA_LIBRARY}
    ${GSL_GSL_LIBRARY} ${GSL_CBLAS_LIBRARY})

SET_TARGET_PROPERTIES(vsvm PROPERTIES
                      LINK_FLAGS -Wl)

INSTALL(TARGETS vsvm
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)

