PROJECT(vattredit)

ADD_EXECUTABLE(vattredit vattredit.c)
TARGET_LINK_LIBRARIES(vattredit lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vattredit
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
