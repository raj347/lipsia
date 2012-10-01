PROJECT(vsvm)

ADD_EXECUTABLE(vsvm vsvm.cpp MriSvm.cpp libsvm-dense/svm.o)
ADD_EXECUTABLE(vsearchlightsvm vsearchlightsvm.cpp MriSvm.cpp SearchLight.cpp libsvm-dense/svm.o)
ADD_EXECUTABLE(vpca vpca.cpp PCA.cpp MriSvm.cpp libsvm-dense/svm.cpp SearchLight.cpp)
ADD_EXECUTABLE(vsvr vsvr.cpp PCA.cpp MriSvm.cpp libsvm-dense/svm.cpp SearchLight.cpp)

TARGET_LINK_LIBRARIES(vsvm ${VIAIO_LIBRARY}  ${LIPSIA_LIBRARY}
    ${GSL_GSL_LIBRARY} ${GSL_CBLAS_LIBRARY})

TARGET_LINK_LIBRARIES(vsearchlightsvm ${VIAIO_LIBRARY}  ${LIPSIA_LIBRARY}
    ${GSL_GSL_LIBRARY} ${GSL_CBLAS_LIBRARY})

TARGET_LINK_LIBRARIES(vpca ${VIAIO_LIBRARY} ${LIPSIA_LIBRARY}
    ${GSL_GSL_LIBRARY} ${GSL_CBLAS_LIBRARY})

TARGET_LINK_LIBRARIES(vsvr ${VIAIO_LIBRARY} ${LIPSIA_LIBRARY}
    ${GSL_GSL_LIBRARY} ${GSL_CBLAS_LIBRARY})

SET_TARGET_PROPERTIES(vsvm PROPERTIES
                      LINK_FLAGS -Wl)

SET_TARGET_PROPERTIES(vsearchlightsvm PROPERTIES
                      LINK_FLAGS -Wl)

SET_TARGET_PROPERTIES(vpca PROPERTIES
                      LINK_FLAGS -Wl)

SET_TARGET_PROPERTIES(vsvr PROPERTIES
                      LINK_FLAGS -Wl)

INSTALL(TARGETS vsvm
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)

INSTALL(TARGETS vsearchlightsvm
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)


INSTALL(TARGETS vsearchlightsvm
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)

