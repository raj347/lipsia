PROJECT(vtomat)

ADD_EXECUTABLE(vtomat vtomat.c)
TARGET_LINK_LIBRARIES(vtomat lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vtomat
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
