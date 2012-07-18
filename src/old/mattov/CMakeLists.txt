PROJECT(mattov)

ADD_EXECUTABLE(mattov mattov.c)
TARGET_LINK_LIBRARIES(mattov lipsia ${VIA_LIBRARY})


INSTALL(TARGETS mattov
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
