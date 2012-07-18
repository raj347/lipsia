PROJECT(vdemon)

ADD_EXECUTABLE(vdemon maxwell.C vdemon.C)
TARGET_LINK_LIBRARIES(vdemon lipsia ${VIA_LIBRARY})

INSTALL(TARGETS vdemon
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
