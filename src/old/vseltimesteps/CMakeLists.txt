PROJECT(vseltimesteps)

ADD_EXECUTABLE(vseltimesteps vseltimesteps.c)
TARGET_LINK_LIBRARIES(vseltimesteps lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vseltimesteps
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
