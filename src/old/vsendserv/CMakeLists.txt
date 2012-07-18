PROJECT(vsendserv)

ADD_EXECUTABLE(vsendserv vsendserv.C)
TARGET_LINK_LIBRARIES(vsendserv lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vsendserv
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
