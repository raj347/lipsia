PROJECT(vattrcopy)

ADD_EXECUTABLE(vattrcopy vattrcopy.c)
TARGET_LINK_LIBRARIES(vattrcopy lipsia ${VIA_LIBRARY})

SET_TARGET_PROPERTIES(vattrcopy PROPERTIES
                      COMPILE_FLAGS -ansi)

INSTALL(TARGETS vattrcopy
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
