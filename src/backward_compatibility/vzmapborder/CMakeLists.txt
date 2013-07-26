PROJECT(vzmapborder)

ADD_EXECUTABLE(vzmapborder vzmapborder.c)
TARGET_LINK_LIBRARIES(vzmapborder lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vzmapborder
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
