PROJECT(vbayesgroup)

ADD_EXECUTABLE(vbayesgroup vbayesgroup.c)
TARGET_LINK_LIBRARIES(vbayesgroup lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vbayesgroup
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
