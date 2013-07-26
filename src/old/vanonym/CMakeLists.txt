PROJECT(vanonym)

ADD_EXECUTABLE(vanonym vanonym.c)
TARGET_LINK_LIBRARIES(vanonym lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vanonym
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
