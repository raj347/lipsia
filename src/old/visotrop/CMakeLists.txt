PROJECT(visotrop)

ADD_EXECUTABLE(visotrop visotrop.c)
TARGET_LINK_LIBRARIES(visotrop lipsia ${VIA_LIBRARY})


INSTALL(TARGETS visotrop
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
