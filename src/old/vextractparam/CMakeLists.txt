PROJECT(vextractparam)

ADD_EXECUTABLE(vextractparam vextractparam.c)
TARGET_LINK_LIBRARIES(vextractparam lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vextractparam
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
