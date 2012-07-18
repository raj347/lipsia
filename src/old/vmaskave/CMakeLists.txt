PROJECT(vmaskave)

ADD_EXECUTABLE(vmaskave vmaskave.c)
TARGET_LINK_LIBRARIES(vmaskave lipsia ${VIA_LIBRARY})

INSTALL(TARGETS vmaskave
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
