PROJECT(vmaskedit)

ADD_EXECUTABLE(vmaskedit vmaskedit.c)
TARGET_LINK_LIBRARIES(vmaskedit lipsia ${VIA_LIBRARY})

INSTALL(TARGETS vmaskedit
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
