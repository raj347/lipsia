PROJECT(vtimestep)

ADD_EXECUTABLE(vtimestep vtimestep.c)
TARGET_LINK_LIBRARIES(vtimestep lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vtimestep
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
