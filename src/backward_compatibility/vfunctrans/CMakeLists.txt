PROJECT(vfunctrans)

ADD_EXECUTABLE(vfunctrans
               Functrans.c  SliceOrientation.c  vfunctrans.c FuncCompress.c
               TransTimeStep.c)
TARGET_LINK_LIBRARIES(vfunctrans lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vfunctrans
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
