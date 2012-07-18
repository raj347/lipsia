PROJECT(vconvcoord)

ADD_EXECUTABLE(vconvcoord vconvcoord.c)
TARGET_LINK_LIBRARIES(vconvcoord lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vconvcoord
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
