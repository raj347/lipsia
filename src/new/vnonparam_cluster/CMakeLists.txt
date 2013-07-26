PROJECT(vnonparam_cluster)

ADD_EXECUTABLE(vthreshold vthreshold.cpp Threshold.cpp )
ADD_EXECUTABLE(vclusterhist vclusterhist.cpp Label3d.o )
ADD_EXECUTABLE(vclusterassess vclusterassess.cpp Label3d.o )

TARGET_LINK_LIBRARIES(vthreshold ${VIAIO_LIBRARY} ${LIPSIA_LIBRARY} ${GSL_GSL_LIBRARY} ${GSL_CBLAS_LIBRARY} )
TARGET_LINK_LIBRARIES(vclusterhist ${VIAIO_LIBRARY} ${GSL_GSL_LIBRARY} ${GSL_CBLAS_LIBRARY} )
TARGET_LINK_LIBRARIES(vclusterassess ${VIAIO_LIBRARY} ${LIPSIA_LIBRARY} ${GSL_GSL_LIBRARY} ${GSL_CBLAS_LIBRARY} )

