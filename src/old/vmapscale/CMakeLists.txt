PROJECT(vmapscale)

ADD_EXECUTABLE(vmapscale vmapscale.c)
TARGET_LINK_LIBRARIES(vmapscale lipsia ${VIA_LIBRARY})

INSTALL(TARGETS vmapscale
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
