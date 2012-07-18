PROJECT(vmetropolis)

ADD_EXECUTABLE(vmetropolis vmetropolis.c)
TARGET_LINK_LIBRARIES(vmetropolis lipsia ${VIA_LIBRARY})

INSTALL(TARGETS vmetropolis
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
