PROJECT(vdelcereb)

ADD_EXECUTABLE(vdelcereb vdelcereb.c)
TARGET_LINK_LIBRARIES(vdelcereb lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vdelcereb
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
