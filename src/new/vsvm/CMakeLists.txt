PROJECT(vsvm)

ADD_EXECUTABLE(vsl vsl.cpp MriSvm.cpp SearchLight.cpp libsvm-dense/svm.o)
ADD_EXECUTABLE(vsvm vsvm.cpp PCA.cpp MriSvm.cpp libsvm-dense/svm.cpp SearchLight.cpp /home/raid1/tibu/PRIVATE/hexenkueche/atlas-allgemein/lib/libptcblas.a /home/raid1/tibu/PRIVATE/hexenkueche/atlas-allgemein/lib/libptf77blas.a /home/raid1/tibu/PRIVATE/hexenkueche/atlas-allgemein/lib/libptlapack.a /home/raid1/tibu/PRIVATE/hexenkueche/atlas-allgemein/lib/libatlas.a)
ADD_EXECUTABLE(vsvr vsvr.cpp PCA.cpp MriSvm.cpp libsvm-dense/svm.cpp SearchLight.cpp)

TARGET_LINK_LIBRARIES(vsl ${VIAIO_LIBRARY}  ${LIPSIA_LIBRARY} ${GSL_GSL_LIBRARY} ${GSL_CBLAS_LIBRARY})
#TARGET_LINK_LIBRARIES(vsvm ${VIAIO_LIBRARY} ${GSL_GSL_LIBRARY} ${GSL_CBLAS_LIBRARY})
TARGET_LINK_LIBRARIES(vsvm ${VIAIO_LIBRARY} ${GSL_GSL_LIBRARY})
TARGET_LINK_LIBRARIES(vsvr ${VIAIO_LIBRARY} ${LIPSIA_LIBRARY} ${GSL_GSL_LIBRARY} ${GSL_CBLAS_LIBRARY})


#SET_TARGET_PROPERTIES(vsl PROPERTIES LINK_FLAGS -Wl)

#SET_TARGET_PROPERTIES(vsvm PROPERTIES LINK_FLAGS -Wl)

#SET_TARGET_PROPERTIES(vsvr PROPERTIES LINK_FLAGS -Wl)
