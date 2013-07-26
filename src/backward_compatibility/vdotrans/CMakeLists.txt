PROJECT(vdotrans)

ADD_EXECUTABLE(vdotrans SliceOrientation.c DoTrans.c  Shuffle.c  vdotrans.c)
TARGET_LINK_LIBRARIES(vdotrans lipsia ${VIA_LIBRARY})

INSTALL(TARGETS vdotrans
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
