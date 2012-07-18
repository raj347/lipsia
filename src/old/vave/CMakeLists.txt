PROJECT(vave)

ADD_EXECUTABLE(vave vave.c)
TARGET_LINK_LIBRARIES(vave lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vave
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
