PROJECT(vnumave)

ADD_EXECUTABLE(vnumave vnumave.c)
TARGET_LINK_LIBRARIES(vnumave lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vnumave
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
