PROJECT(vbayes)

ADD_EXECUTABLE(vbayes vbayes.c)
TARGET_LINK_LIBRARIES(vbayes lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vbayes
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
