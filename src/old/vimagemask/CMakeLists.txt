PROJECT(vimagemask)

ADD_EXECUTABLE(vimagemask vimagemask.c)
TARGET_LINK_LIBRARIES(vimagemask lipsia ${VIA_LIBRARY})

INSTALL(TARGETS vimagemask
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
