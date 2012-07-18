PROJECT(vspm2lipsia)

ADD_EXECUTABLE(vspm2lipsia vspm2lipsia.c)
TARGET_LINK_LIBRARIES(vspm2lipsia lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vspm2lipsia
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
